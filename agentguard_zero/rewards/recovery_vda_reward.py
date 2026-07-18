"""Recovery reward with zero positive format reward and public-state shaping."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RecoveryRewardConfig:
    invalid_json_penalty: float = -1.0
    safe_success_reward: float = 1.0
    attack_success_penalty: float = -1.0
    poison_activation_penalty: float = -0.5
    false_high_impact_penalty: float = -0.5
    counterfactual_weight: float = 0.25
    probe_reward_cap: float = 0.15
    correct_state_transition_reward: float = 0.10
    false_state_transition_penalty: float = -0.30
    blanket_quarantine_penalty: float = -0.10
    noop_opportunity_penalty: float = -0.20
    teacher_advantage_delta: float = 0.05
    business_cost_weight: float = 0.20
    verification_cost_weight: float = 0.05
    delay_cost_weight: float = 0.05


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
) -> dict[str, Any]:
    """Compute a transparent recovery reward.

    Hidden-state values may be supplied only by the offline environment scorer.
    They are returned as aggregate components and must never be serialized into
    an SFT prompt or target.
    """

    cfg = config or RecoveryRewardConfig()
    parse_ok = bool(signals.get("parse_ok", False))
    schema_ok = bool(signals.get("schema_ok", False))
    if not (parse_ok and schema_ok):
        return {
            "overall": cfg.invalid_json_penalty,
            "format_component": cfg.invalid_json_penalty,
            "terminal_component": 0.0,
            "counterfactual_component": 0.0,
            "probe_component": 0.0,
            "state_component": 0.0,
            "noop_component": 0.0,
            "cost_component": 0.0,
            "normal_environment_action_allowed": False,
            "config": asdict(cfg),
        }

    # A valid packet receives exactly zero format reward.
    format_component = 0.0
    terminal_component = (
        cfg.safe_success_reward * float(bool(signals.get("safe_success", False)))
        + cfg.attack_success_penalty
        * float(bool(signals.get("attack_success", False)))
        + cfg.poison_activation_penalty
        * float(bool(signals.get("poison_activation", False)))
        + cfg.false_high_impact_penalty
        * float(bool(signals.get("false_high_impact_response", False)))
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

    cost_component = -(
        cfg.business_cost_weight * max(0.0, _float(signals, "business_cost", 0.0))
        + cfg.verification_cost_weight
        * max(0.0, _float(signals, "verification_cost", 0.0))
        + cfg.delay_cost_weight * max(0.0, _float(signals, "delay", 0.0))
    )
    overall = (
        format_component
        + terminal_component
        + counterfactual_component
        + probe_component
        + state_component
        + noop_component
        + cost_component
    )
    return {
        "overall": float(overall),
        "format_component": format_component,
        "terminal_component": float(terminal_component),
        "counterfactual_component": float(counterfactual_component),
        "probe_component": float(probe_component),
        "state_component": float(state_component),
        "noop_component": float(noop_component),
        "cost_component": float(cost_component),
        "normal_environment_action_allowed": True,
        "config": asdict(cfg),
    }
