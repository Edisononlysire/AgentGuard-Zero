"""Frontier-focused DCA reward for candidate-level VDA feedback."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CandidateDCARewardConfig:
    frontier_weight: float = 0.35
    novelty_weight: float = 0.20
    skill_gap_weight: float = 0.20
    vda_regret_weight: float = 0.25
    invalid_penalty: float = 1.0
    unreachable_penalty: float = 1.0


def frontier_score(safe_success_samples: Sequence[bool | int | float]) -> float:
    if not safe_success_samples:
        return 0.0
    probability = sum(float(bool(item)) for item in safe_success_samples) / len(
        safe_success_samples
    )
    return float(4.0 * probability * (1.0 - probability))


def compute_candidate_dca_reward(
    feedback: Mapping[str, Any],
    config: CandidateDCARewardConfig | None = None,
) -> dict[str, float | bool]:
    cfg = config or CandidateDCARewardConfig()
    parser_or_compiler_failure = bool(
        feedback.get("parser_failure")
        or feedback.get("compiler_failure")
        or feedback.get("candidate_generation_failure")
    )
    solvable = bool(feedback.get("teacher_solvable", False))
    if parser_or_compiler_failure:
        return {
            "reward": -cfg.invalid_penalty,
            "frontier": 0.0,
            "safe_success_probability": 0.0,
            "admitted_frontier": False,
            "infrastructure_failure": True,
        }
    if not solvable:
        return {
            "reward": -cfg.unreachable_penalty,
            "frontier": 0.0,
            "safe_success_probability": 0.0,
            "admitted_frontier": False,
            "infrastructure_failure": False,
        }
    samples = list(feedback.get("safe_success_samples", []) or [])
    probability = sum(float(bool(item)) for item in samples) / max(1, len(samples))
    frontier = frontier_score(samples)
    novelty = min(1.0, max(0.0, float(feedback.get("novelty", 0.0))))
    skill_gap = min(1.0, max(0.0, float(feedback.get("skill_gap", 0.0))))
    regret = min(1.0, max(0.0, float(feedback.get("vda_regret", 0.0))))
    reward = (
        cfg.frontier_weight * frontier
        + cfg.novelty_weight * novelty
        + cfg.skill_gap_weight * skill_gap
        + cfg.vda_regret_weight * regret
    )
    return {
        "reward": float(reward),
        "frontier": frontier,
        "safe_success_probability": probability,
        "admitted_frontier": 0.20 <= probability <= 0.80,
        "infrastructure_failure": False,
    }


def summarize_candidate_dca_feedback(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rewards = [float(row.get("reward", 0.0)) for row in rows]
    return {
        "scenario_count": len(rows),
        "teacher_solvability": sum(
            int(bool(row.get("teacher_solvable", False))) for row in rows
        )
        / max(1, len(rows)),
        "vda_action_validity": sum(
            float(row.get("vda_action_validity", 0.0)) for row in rows
        )
        / max(1, len(rows)),
        "frontier_scenario_rate": sum(
            int(bool(row.get("admitted_frontier", False))) for row in rows
        )
        / max(1, len(rows)),
        "reward_variance": statistics.pvariance(rewards) if len(rewards) >= 2 else 0.0,
        "parser_or_compiler_exploit_count": sum(
            int(bool(row.get("infrastructure_failure", False))) for row in rows
        ),
        "reward_config": asdict(CandidateDCARewardConfig()),
    }
