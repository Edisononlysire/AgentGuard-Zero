"""Recovery reward with zero positive format reward and public-state shaping."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


REQUIRED_RECOVERY_SIGNALS = frozenset(
    {
        "parse_ok",
        "schema_ok",
        "safe_success",
        "attack_success",
        "counterfactual_advantage",
        "teacher_advantage",
        "new_evidence",
        "uncertainty_reduced",
        "probe_grounded_state_update",
        "probe_grounded_authorization",
        "probe_counterfactual_improvement",
        "correct_state_transitions",
        "false_state_transitions",
        "business_cost",
        "verification_cost",
        "delay",
        "action",
        "core_utility",
    }
)


@dataclass(frozen=True)
class RecoveryRewardConfig:
    invalid_json_penalty: float = -1.0
    core_utility_weight: float = 1.0
    counterfactual_weight: float = 0.15
    probe_reward_cap: float = 0.05
    correct_state_transition_reward: float = 0.05
    false_state_transition_penalty: float = -0.20
    blanket_quarantine_penalty: float = -0.10
    noop_opportunity_penalty: float = -0.20
    teacher_advantage_delta: float = 0.05
    shaping_absolute_cap: float = 0.25


def _float(signals: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = signals.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric reward signal: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite reward signal: {name}")
    return result


def compute_recovery_reward(
    signals: Mapping[str, Any],
    config: RecoveryRewardConfig | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Compute a transparent recovery reward.

    Hidden-state values may be supplied only by the offline environment scorer.
    They are returned as aggregate components and must never be serialized into
    an SFT prompt or target.
    """

    cfg = config or RecoveryRewardConfig()
    if strict:
        missing = REQUIRED_RECOVERY_SIGNALS.difference(signals)
        if missing:
            raise RuntimeError(f"missing recovery reward signals: {sorted(missing)}")
    parse_ok = bool(signals.get("parse_ok", False))
    schema_ok = bool(signals.get("schema_ok", False))
    if not (parse_ok and schema_ok):
        return {
            "overall": cfg.invalid_json_penalty,
            "format_component": cfg.invalid_json_penalty,
            "core_component": 0.0,
            "terminal_component": 0.0,
            "counterfactual_component": 0.0,
            "probe_component": 0.0,
            "state_component": 0.0,
            "noop_component": 0.0,
            "shaping_component": 0.0,
            "shaping_was_capped": False,
            "cost_component": 0.0,
            "normal_environment_action_allowed": False,
            "config": asdict(cfg),
        }

    # A valid packet receives exactly zero format reward.
    format_component = 0.0
    core_component = cfg.core_utility_weight * _float(
        signals,
        "core_utility",
        0.0,
    )

    counterfactual_advantage = max(
        -1.0,
        min(1.0, _float(signals, "counterfactual_advantage", 0.0)),
    )
    counterfactual_component = cfg.counterfactual_weight * counterfactual_advantage

    information_events = sum(
        int(bool(signals.get(name, False)))
        for name in (
            "new_evidence",
            "uncertainty_reduced",
            "probe_grounded_state_update",
            "probe_grounded_authorization",
            "probe_counterfactual_improvement",
        )
    )
    probe_component = min(
        cfg.probe_reward_cap,
        cfg.probe_reward_cap * information_events / 3.0,
    )

    correct_transitions = max(
        0,
        int(_float(signals, "correct_state_transitions", 0.0)),
    )
    false_transitions = max(
        0,
        int(_float(signals, "false_state_transitions", 0.0)),
    )
    state_component = (
        cfg.correct_state_transition_reward * correct_transitions
        + cfg.false_state_transition_penalty * false_transitions
    )
    if bool(signals.get("blanket_quarantine", False)):
        state_component += cfg.blanket_quarantine_penalty

    teacher_advantage = _float(signals, "teacher_advantage", 0.0)
    noop_component = 0.0
    if (
        str(signals.get("action", "")) == "Observe"
        and teacher_advantage > cfg.teacher_advantage_delta
    ):
        noop_component = cfg.noop_opportunity_penalty

    # Business, verification, delay, poison, and overresponse costs already
    # live in the shared U_core.  They remain required above for plumbing and
    # audit, but are not counted a second time here.
    cost_component = 0.0
    shaping_uncapped = (
        counterfactual_component + probe_component + state_component + noop_component
    )
    shaping_component = max(
        -cfg.shaping_absolute_cap,
        min(cfg.shaping_absolute_cap, shaping_uncapped),
    )
    overall = format_component + core_component + shaping_component + cost_component
    return {
        "overall": float(overall),
        "format_component": format_component,
        "core_component": float(core_component),
        "terminal_component": float(core_component),
        "counterfactual_component": float(counterfactual_component),
        "probe_component": float(probe_component),
        "state_component": float(state_component),
        "noop_component": float(noop_component),
        "shaping_component": float(shaping_component),
        "shaping_was_capped": abs(shaping_uncapped) > cfg.shaping_absolute_cap,
        "cost_component": float(cost_component),
        "normal_environment_action_allowed": True,
        "config": asdict(cfg),
    }


def validate_recovery_signal_batch(
    rows: Iterable[Mapping[str, Any]],
    *,
    minimum_trajectories: int = 32,
) -> dict[str, Any]:
    """Fail-closed smoke gate for real Gate-B rollout signal payloads."""

    samples = [dict(row) for row in rows]
    failures: list[str] = []
    if len(samples) < minimum_trajectories:
        failures.append("trajectory_count_below_minimum")
    rewards: list[dict[str, Any]] = []
    valid_samples: list[dict[str, Any]] = []
    for index, row in enumerate(samples):
        try:
            rewards.append(compute_recovery_reward(row, strict=True))
            valid_samples.append(row)
        except Exception as exc:
            failures.append(f"trajectory_{index}:{exc}")
    if rewards:
        teacher_values = {
            round(_float(row, "teacher_advantage"), 12) for row in valid_samples
        }
        counterfactual_values = {
            round(_float(row, "counterfactual_advantage"), 12) for row in valid_samples
        }
        overall = [float(item["overall"]) for item in rewards]
        if len(teacher_values) < 2:
            failures.append("teacher_advantage_constant")
        if len(counterfactual_values) < 2:
            failures.append("counterfactual_advantage_constant")
        if not (min(overall) < 0.0 < max(overall)):
            failures.append("reward_does_not_cover_positive_and_negative")
        if not any(abs(float(item["probe_component"])) > 0.0 for item in rewards):
            failures.append("probe_component_never_nonzero")
        if not any(abs(float(item["state_component"])) > 0.0 for item in rewards):
            failures.append("state_component_never_nonzero")
    unique_failures = list(dict.fromkeys(failures))
    return {
        "accepted": not unique_failures,
        "status": "accepted" if not unique_failures else "rejected",
        "failures": unique_failures,
        "trajectory_count": len(samples),
        "minimum_trajectories": minimum_trajectories,
        "required_signals": sorted(REQUIRED_RECOVERY_SIGNALS),
        "strict_reward_mode": True,
    }
