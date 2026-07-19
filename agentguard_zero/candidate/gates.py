"""Fail-closed gates for candidate-level VDA co-evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CandidateGateVerdict:
    gate: str
    accepted: bool
    failures: tuple[str, ...]
    metrics: dict[str, Any]
    thresholds: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rate(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric metric: {key}")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"rate outside [0,1]: {key}={number}")
    return number


def evaluate_compiler_gate(metrics: Mapping[str, Any]) -> CandidateGateVerdict:
    thresholds = {
        "compiler_validity": 1.0,
        "public_reference_validity": 1.0,
        "teacher_best_candidate_recall_min": 0.95,
        "near_optimal_candidate_recall_min": 0.99,
        "core_regret_p95_max": 0.02,
        "semantic_duplicate_rate_max": 0.05,
        "candidate_permutation_consistency": 1.0,
        "semantic_target_conflict_rate": 0.0,
    }
    failures = []
    for key in ("compiler_validity", "public_reference_validity"):
        if _rate(metrics, key) != 1.0:
            failures.append(key)
    if _rate(metrics, "teacher_best_candidate_recall") < 0.95:
        failures.append("teacher_best_candidate_recall")
    if _rate(metrics, "near_optimal_candidate_recall") < 0.99:
        failures.append("near_optimal_candidate_recall")
    if float(metrics.get("core_regret_p95", float("inf"))) > 0.02:
        failures.append("core_regret_p95")
    if _rate(metrics, "semantic_duplicate_rate") > 0.05:
        failures.append("semantic_duplicate_rate")
    if _rate(metrics, "candidate_permutation_consistency") != 1.0:
        failures.append("candidate_permutation_consistency")
    if _rate(metrics, "semantic_target_conflict_rate") != 0.0:
        failures.append("semantic_target_conflict_rate")
    return CandidateGateVerdict(
        "candidate_compiler", not failures, tuple(failures), dict(metrics), thresholds
    )


def evaluate_vda_gate_a(metrics: Mapping[str, Any]) -> CandidateGateVerdict:
    thresholds = {
        "action_validity_min": 0.995,
        "invalid_noop_rate_max": 0.005,
        "actionable_observe_rate_max": 0.70,
        "active_probe_rate_min": 0.10,
        "probe_yield_min": 0.10,
        "trust_rate_min": 0.05,
        "memory_operation_rate_min": 0.05,
        "memory_use_rate_min": 0.03,
        "mitigation_rate_min": 0.15,
        "attack_mitigation_min": 0.15,
        "safe_success_min": 0.10,
    }
    checks = {
        "action_validity": _rate(metrics, "action_validity") >= 0.995,
        "invalid_noop_rate": _rate(metrics, "invalid_noop_rate") <= 0.005,
        "actionable_observe_rate": _rate(metrics, "actionable_observe_rate") <= 0.70,
        "active_probe_rate": _rate(metrics, "active_probe_rate") >= 0.10,
        "probe_yield": _rate(metrics, "probe_yield") >= 0.10,
        "trust_rate": _rate(metrics, "trust_rate") >= 0.05,
        "memory_operation_rate": _rate(metrics, "memory_operation_rate") >= 0.05,
        "memory_use_rate": _rate(metrics, "memory_use_rate") >= 0.03,
        "mitigation_rate": _rate(metrics, "mitigation_rate") >= 0.15,
        "attack_mitigation": _rate(metrics, "attack_mitigation") >= 0.15,
        "safe_success": _rate(metrics, "safe_success") >= 0.10,
    }
    failures = tuple(key for key, passed in checks.items() if not passed)
    return CandidateGateVerdict(
        "candidate_gate_a", not failures, failures, dict(metrics), thresholds
    )


def evaluate_dca_feedback_gate(metrics: Mapping[str, Any]) -> CandidateGateVerdict:
    thresholds = {
        "teacher_solvability_min": 0.95,
        "vda_action_validity_min": 0.995,
        "frontier_scenario_rate_min": 0.30,
        "reward_variance_exclusive": 0.0,
        "parser_or_compiler_exploit_count": 0,
    }
    checks = {
        "teacher_solvability": _rate(metrics, "teacher_solvability") >= 0.95,
        "vda_action_validity": _rate(metrics, "vda_action_validity") >= 0.995,
        "frontier_scenario_rate": _rate(metrics, "frontier_scenario_rate") >= 0.30,
        "reward_variance": float(metrics.get("reward_variance", 0.0)) > 0.0,
        "parser_or_compiler_exploit_count": int(
            metrics.get("parser_or_compiler_exploit_count", -1)
        )
        == 0,
    }
    failures = tuple(key for key, passed in checks.items() if not passed)
    return CandidateGateVerdict(
        "candidate_dca_feedback", not failures, failures, dict(metrics), thresholds
    )


def evaluate_round_gate(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    *,
    round_index: int,
) -> CandidateGateVerdict:
    thresholds = {
        "fixed_safe_utility_non_decreasing": True,
        "fixed_safe_success_non_decreasing": True,
        "fixed_action_validity_non_decreasing": True,
        "one_improvement_required": {
            "safe_success": 1.0 / 32.0,
            "attack_mitigation": 1.0 / 32.0,
            "safe_utility": 0.01,
            "mean_candidate_regret_reduction": 0.01,
        },
    }
    failures = []
    for key in ("safe_utility", "safe_success", "action_validity"):
        if float(end.get(key, float("-inf"))) + 1.0e-12 < float(start.get(key, 0.0)):
            failures.append(f"fixed_{key}_decreased")
    improvements = {
        "safe_success": float(end.get("safe_success", 0.0))
        - float(start.get("safe_success", 0.0))
        >= 1.0 / 32.0,
        "attack_mitigation": float(end.get("attack_mitigation", 0.0))
        - float(start.get("attack_mitigation", 0.0))
        >= 1.0 / 32.0,
        "safe_utility": float(end.get("safe_utility", 0.0))
        - float(start.get("safe_utility", 0.0))
        >= 0.01,
        "mean_candidate_regret": (
            isinstance(start.get("mean_candidate_regret"), (int, float))
            and isinstance(end.get("mean_candidate_regret"), (int, float))
            and float(start["mean_candidate_regret"])
            - float(end["mean_candidate_regret"])
            >= 0.01
        ),
    }
    if not any(improvements.values()):
        failures.append("no_minimum_pilot_improvement")
    for task in ("T1", "T2", "T3", "T4"):
        task_metrics = (end.get("by_task_terminal") or {}).get(task)
        if not isinstance(task_metrics, Mapping) or int(task_metrics.get("scenario_count", 0)) == 0:
            failures.append(f"missing_task:{task}")
    packed = {"round_index": round_index, "start": dict(start), "end": dict(end), "improvements": improvements}
    return CandidateGateVerdict(
        f"candidate_round_{round_index}",
        not failures,
        tuple(failures),
        packed,
        thresholds,
    )
