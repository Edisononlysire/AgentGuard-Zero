"""Shared outcome utility for recovery teachers, gates, and RL rewards.

The core utility is deliberately small and outcome centred.  Public-state
teachers and policy-gradient rewards may add bounded shaping, but they must not
replace or outweigh this common objective.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RecoveryCoreUtilityConfig:
    mitigation_reward: float = 1.00
    safe_success_reward: float = 0.50
    attack_success_penalty: float = -1.00
    poison_success_penalty: float = -0.50
    business_cost_weight: float = 0.25
    verification_cost_weight: float = 0.10
    delay_cost_weight: float = 0.10
    overresponse_penalty: float = -0.50
    unauthorized_high_impact_weight: float = 0.25
    invalid_state_operation_weight: float = 0.25
    teacher_shaping_cap: float = 0.25


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric core utility signal: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite core utility signal: {name}")
    return result


def recovery_core_components(
    env: Any,
    score: Mapping[str, Any],
    config: RecoveryCoreUtilityConfig | None = None,
) -> dict[str, float]:
    """Return normalized, auditable core-utility components for one trajectory."""

    cfg = config or RecoveryCoreUtilityConfig()
    constraints = env.scenario.get("defense_constraints", {}) or {}
    business_budget = max(1.0, float(constraints.get("business_budget", 5.0)))
    max_steps = max(1.0, float(getattr(env, "max_steps", 1)))
    business = min(
        1.0,
        max(
            0.0,
            _finite(score.get("business_cost", 0.0), "business_cost") / business_budget,
        ),
    )
    verification = min(
        1.0,
        max(
            0.0,
            _finite(score.get("verification_cost", 0.0), "verification_cost")
            / max_steps,
        ),
    )
    delay = min(1.0, max(0.0, _finite(score.get("delay", 0.0), "delay")))
    unauthorized = min(
        1.0,
        max(
            0.0,
            _finite(
                score.get("unauthorized_high_impact", 0.0),
                "unauthorized_high_impact",
            )
            / max_steps,
        ),
    )
    invalid_state = min(
        1.0,
        max(
            0.0,
            _finite(
                score.get("invalid_state_operations", 0.0),
                "invalid_state_operations",
            )
            / max_steps,
        ),
    )
    poison_success = bool(
        score.get("poison_success", score.get("poison_activation", False))
    )
    return {
        "mitigation": cfg.mitigation_reward
        * float(bool(score.get("attack_mitigated", False))),
        "safe_success": cfg.safe_success_reward
        * float(bool(score.get("safe_success", False))),
        "attack_success": cfg.attack_success_penalty
        * float(bool(score.get("attack_success", False))),
        "poison_success": cfg.poison_success_penalty * float(poison_success),
        "business_cost": -cfg.business_cost_weight * business,
        "verification_cost": -cfg.verification_cost_weight * verification,
        "delay_cost": -cfg.delay_cost_weight * delay,
        "overresponse": cfg.overresponse_penalty
        * float(bool(score.get("overresponse", False))),
        "unauthorized_high_impact": -cfg.unauthorized_high_impact_weight * unauthorized,
        "invalid_state_operations": -cfg.invalid_state_operation_weight * invalid_state,
    }


def recovery_core_utility(
    env: Any,
    score: Mapping[str, Any],
    config: RecoveryCoreUtilityConfig | None = None,
) -> float:
    return float(sum(recovery_core_components(env, score, config).values()))


def core_utility_manifest(
    config: RecoveryCoreUtilityConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RecoveryCoreUtilityConfig()
    return {
        "name": "recovery_core_utility_v1",
        "config": asdict(cfg),
        "teacher_shaping_rule": "absolute_total_at_most_teacher_shaping_cap",
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def spearman_rank_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left_ranks, right_ranks)
    )
    left_scale = sqrt(sum((x - left_mean) ** 2 for x in left_ranks))
    right_scale = sqrt(sum((y - right_mean) ** 2 for y in right_ranks))
    if left_scale <= 1.0e-12 or right_scale <= 1.0e-12:
        return 1.0 if left_ranks == right_ranks else 0.0
    return numerator / (left_scale * right_scale)
