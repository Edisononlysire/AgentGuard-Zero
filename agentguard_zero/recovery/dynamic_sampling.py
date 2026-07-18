"""Adaptive G=2→4 sampling and DAPO-style zero-advantage filtering."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import pstdev
from typing import Any, Iterable, Mapping, Sequence


MALFORMED_TRAJECTORY_REASONS = frozenset(
    {
        "json_incomplete",
        "tool_call_unterminated",
        "empty_turn",
        "repeated_identical_action",
        "tool_result_mismatch",
        "terminal_reward_missing",
    }
)


@dataclass(frozen=True)
class RolloutSample:
    scenario_id: str
    reward: float
    action_class: str
    safe_success: bool
    trajectory_valid: bool = True
    invalid_reason: str = ""
    terminal_reward_present: bool = True

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "RolloutSample":
        reward = row.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError("rollout reward must be numeric")
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError("rollout reward must be finite")
        return cls(
            scenario_id=str(row.get("scenario_id", "")).strip(),
            reward=value,
            action_class=str(row.get("action_class", "")).strip(),
            safe_success=bool(row.get("safe_success", False)),
            trajectory_valid=bool(row.get("trajectory_valid", True)),
            invalid_reason=str(row.get("invalid_reason", "")).strip(),
            terminal_reward_present=bool(row.get("terminal_reward_present", True)),
        )


@dataclass(frozen=True)
class GroupDecision:
    scenario_id: str
    sample_count: int
    reward_std: float
    all_observe: bool
    all_failed: bool
    all_succeeded: bool
    malformed_count: int
    action: str
    additional_rollouts: int
    usable_for_policy_gradient: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _same_scenario(samples: Sequence[RolloutSample]) -> str:
    ids = {item.scenario_id for item in samples if item.scenario_id}
    if len(ids) != 1:
        raise ValueError("rollout group must contain one non-empty scenario_id")
    return next(iter(ids))


def evaluate_rollout_group(
    rows: Iterable[RolloutSample | Mapping[str, Any]],
    *,
    initial_rollouts: int = 2,
    adaptive_rollouts: int = 4,
    reward_epsilon: float = 1.0e-8,
) -> GroupDecision:
    samples = [
        item if isinstance(item, RolloutSample) else RolloutSample.from_mapping(item)
        for item in rows
    ]
    scenario_id = _same_scenario(samples)
    if len(samples) not in {initial_rollouts, adaptive_rollouts}:
        raise ValueError(
            f"unexpected rollout group size {len(samples)}; "
            f"expected {initial_rollouts} or {adaptive_rollouts}"
        )

    malformed = [
        item
        for item in samples
        if (
            not item.trajectory_valid
            or not item.terminal_reward_present
            or item.invalid_reason in MALFORMED_TRAJECTORY_REASONS
        )
    ]
    if malformed:
        return GroupDecision(
            scenario_id=scenario_id,
            sample_count=len(samples),
            reward_std=0.0,
            all_observe=False,
            all_failed=False,
            all_succeeded=False,
            malformed_count=len(malformed),
            action="exclude",
            additional_rollouts=0,
            usable_for_policy_gradient=False,
            reason="malformed_multiturn_trajectory",
        )

    rewards = [item.reward for item in samples]
    reward_std = float(pstdev(rewards)) if len(rewards) > 1 else 0.0
    all_observe = all(item.action_class == "observe" for item in samples)
    all_failed = all(not item.safe_success for item in samples)
    all_succeeded = all(item.safe_success for item in samples)
    zero_advantage = reward_std <= reward_epsilon
    # all_failed/all_succeeded are diagnostics, not exclusion criteria.  The
    # recovery reward deliberately represents partial progress and cost, so a
    # group can be useful even when every trajectory shares the same terminal
    # success flag.
    degenerate = zero_advantage or all_observe

    if len(samples) == initial_rollouts and degenerate:
        return GroupDecision(
            scenario_id=scenario_id,
            sample_count=len(samples),
            reward_std=reward_std,
            all_observe=all_observe,
            all_failed=all_failed,
            all_succeeded=all_succeeded,
            malformed_count=0,
            action="resample",
            additional_rollouts=adaptive_rollouts - initial_rollouts,
            usable_for_policy_gradient=False,
            reason="initial_group_degenerate",
        )

    if degenerate:
        return GroupDecision(
            scenario_id=scenario_id,
            sample_count=len(samples),
            reward_std=reward_std,
            all_observe=all_observe,
            all_failed=all_failed,
            all_succeeded=all_succeeded,
            malformed_count=0,
            action="exclude",
            additional_rollouts=0,
            usable_for_policy_gradient=False,
            reason="adaptive_group_still_degenerate",
        )

    return GroupDecision(
        scenario_id=scenario_id,
        sample_count=len(samples),
        reward_std=reward_std,
        all_observe=False,
        all_failed=all_failed,
        all_succeeded=all_succeeded,
        malformed_count=0,
        action="use",
        additional_rollouts=0,
        usable_for_policy_gradient=True,
        reason="nonzero_advantage_group",
    )


def summarize_update_batch(
    groups: Iterable[GroupDecision],
    *,
    minimum_nonzero_advantage_rate: float = 0.50,
) -> dict[str, Any]:
    decisions = list(groups)
    if not decisions:
        raise ValueError("dynamic-sampling batch is empty")
    usable = sum(item.usable_for_policy_gradient for item in decisions)
    resample = sum(item.action == "resample" for item in decisions)
    excluded = sum(item.action == "exclude" for item in decisions)
    rate = usable / len(decisions)
    accepted = rate + 1.0e-12 >= minimum_nonzero_advantage_rate and resample == 0
    return {
        "accepted": accepted,
        "status": "accepted" if accepted else "replenish_before_update",
        "group_count": len(decisions),
        "usable_group_count": usable,
        "resample_group_count": resample,
        "excluded_group_count": excluded,
        "additional_rollouts_requested": sum(
            item.additional_rollouts for item in decisions
        ),
        "replacement_scenarios_required": max(
            0,
            math.ceil(minimum_nonzero_advantage_rate * len(decisions)) - usable,
        ),
        "nonzero_advantage_group_rate": rate,
        "minimum_nonzero_advantage_group_rate": minimum_nonzero_advantage_rate,
        "policy_update_allowed": accepted,
    }
