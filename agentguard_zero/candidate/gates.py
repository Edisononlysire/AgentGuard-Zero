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


def evaluate_vda_learnability_gate(
    metrics: Mapping[str, Any],
) -> CandidateGateVerdict:
    """Fail closed on the frozen 256-state head-overfit protocol."""

    thresholds = {
        "record_count": 256,
        "acceptable_candidate_accuracy_min": 0.95,
        "state_conditioned_family_accuracy_min": 0.95,
        "mean_teacher_regret_max": 0.02,
        "probe_target_accuracy_min": 0.90,
        "mitigation_target_accuracy_min": 0.90,
        "memory_lifecycle_accuracy_min": 0.90,
        "teacher_probability_argmax_rate": 1.0,
        "core_ineligible_probability_mass": 0.0,
        "semantic_target_conflict_rate": 0.0,
        "teacher_target_recall": 1.0,
        "teacher_final_target_alignment_rate": 1.0,
        "semantic_public_state_duplicate_count": 0,
        "head_only_optimizer_steps_min": 500,
        "head_only_optimizer_steps_max": 1000,
        "head_learning_rate_min": 3.0e-4,
        "head_learning_rate_max": 1.0e-3,
        "critical_field_truncated_count": 0,
        "utility_variance_relative_gain_min": 0.25,
        "utility_variance_absolute_gain_min": 1.0e-4,
        "target_margin_gain_min": 0.05,
        "argmax_decision_change_count_min": 1,
        "formal_selection_modes": [
            "candidate_argmax",
            "hierarchical_family_then_utility",
        ],
    }
    checks = {
        "record_count": int(metrics.get("record_count", -1)) == 256,
        "acceptable_candidate_accuracy": _rate(
            metrics, "acceptable_candidate_accuracy"
        )
        >= 0.95,
        "state_conditioned_family_accuracy": _rate(
            metrics, "state_conditioned_family_accuracy"
        )
        >= 0.95,
        "mean_teacher_regret": float(
            metrics.get("mean_teacher_regret", float("inf"))
        )
        <= 0.02,
        "probe_target_accuracy": _rate(metrics, "probe_target_accuracy") >= 0.90,
        "mitigation_target_accuracy": _rate(metrics, "mitigation_target_accuracy")
        >= 0.90,
        "memory_lifecycle_accuracy": _rate(metrics, "memory_lifecycle_accuracy")
        >= 0.90,
        "teacher_probability_argmax_rate": _rate(
            metrics, "teacher_probability_argmax_rate"
        )
        == 1.0,
        "core_ineligible_probability_mass": float(
            metrics.get("core_ineligible_probability_mass", float("inf"))
        )
        == 0.0,
        "semantic_target_conflict_rate": _rate(
            metrics, "semantic_target_conflict_rate"
        )
        == 0.0,
        "teacher_target_recall": _rate(metrics, "teacher_target_recall") == 1.0,
        "teacher_final_target_alignment_rate": _rate(
            metrics, "teacher_final_target_alignment_rate"
        )
        == 1.0,
        "semantic_public_state_duplicate_count": int(
            metrics.get("semantic_public_state_duplicate_count", -1)
        )
        == 0,
        "hidden_state_in_model_input": metrics.get("hidden_state_in_model_input")
        is False,
        "teacher_q_in_model_input": metrics.get("teacher_q_in_model_input") is False,
        "protocol_role": metrics.get("protocol_role") == "learnability_256",
        "head_only_backbone_frozen": metrics.get("head_only_backbone_frozen") is True,
        "head_only_optimizer_steps": 500
        <= int(metrics.get("head_only_optimizer_steps", -1))
        <= 1000,
        "head_learning_rate": 3.0e-4
        <= float(metrics.get("head_learning_rate", float("inf")))
        <= 1.0e-3,
        "critical_field_truncated_count": int(
            metrics.get("critical_field_truncated_count", -1)
        )
        == 0,
        "utility_variance_relative_gain": float(
            metrics.get("utility_variance_relative_gain", float("-inf"))
        )
        >= 0.25,
        "utility_variance_absolute_gain": float(
            metrics.get("utility_variance_absolute_gain", float("-inf"))
        )
        >= 1.0e-4,
        "target_margin_gain": float(
            metrics.get("target_margin_gain", float("-inf"))
        )
        >= 0.05,
        "argmax_decision_change_count": int(
            metrics.get("argmax_decision_change_count", 0)
        )
        >= 1,
        "formal_selection_mode": metrics.get("selection_mode")
        in {"candidate_argmax", "hierarchical_family_then_utility"},
        "required_target_families_present": metrics.get(
            "required_target_families_present"
        )
        is True,
    }
    failures = tuple(key for key, passed in checks.items() if not passed)
    return CandidateGateVerdict(
        "candidate_learnability_256",
        not failures,
        failures,
        dict(metrics),
        thresholds,
    )


def evaluate_vda_gate_a_three_seed(
    metrics: Mapping[str, Any],
) -> CandidateGateVerdict:
    """Evaluate the frozen learned-only 200-scenario x 3-seed Gate A."""

    thresholds = {
        "seed_count": 3,
        "scenarios_per_seed": 200,
        "scenarios_per_task_per_seed": 50,
        "worst_seed_action_validity_min": 0.995,
        "worst_seed_invalid_noop_rate_max": 0.005,
        "worst_seed_actionable_observe_rate_max": 0.70,
        "worst_seed_safe_success_min": 0.10,
        "worst_seed_attack_mitigation_min": 0.15,
        "worst_seed_active_probe_rate_min": 0.10,
        "worst_seed_t1_probe_yield_min": 0.10,
        "worst_seed_trust_operation_rate_min": 0.05,
        "worst_seed_t2_betrayal_detection_exclusive": 0.0,
        "worst_seed_memory_operation_rate_min": 0.05,
        "worst_seed_memory_use_rate_min": 0.03,
        "worst_seed_mitigation_rate_min": 0.15,
        "max_t4_overresponse_delta_vs_warmstart": 0.0,
        "max_t4_business_cost_delta_vs_warmstart": 0.0,
        "seed_task_collapse_count": 0,
        "probe_budget_1_safe_utility_gain_exclusive": 0.0,
        "probe_gain_minus_verification_cost_exclusive": 0.0,
        "memory_causal_utility_gain_exclusive": 0.0,
        "benign_overresponse_delta_max": 0.0,
    }
    checks = {
        "seed_count": int(metrics.get("seed_count", -1)) == 3,
        "distinct_seed_count": int(metrics.get("distinct_seed_count", -1)) == 3,
        "scenarios_per_seed": metrics.get("scenarios_per_seed_valid") is True,
        "scenarios_per_task_per_seed": metrics.get("task_balance_valid") is True,
        "learned_only_selection": metrics.get("learned_only_selection") is True,
        "dca_frozen": metrics.get("dca_frozen") is True,
        "ecrg_disabled": metrics.get("ecrg_disabled") is True,
        "paired_warmstart_suites": metrics.get("paired_warmstart_suites") is True,
        "worst_seed_action_validity": float(
            metrics.get("worst_seed_action_validity", float("-inf"))
        )
        >= 0.995,
        "worst_seed_invalid_noop_rate": float(
            metrics.get("worst_seed_invalid_noop_rate", float("inf"))
        )
        <= 0.005,
        "worst_seed_actionable_observe_rate": float(
            metrics.get("worst_seed_actionable_observe_rate", float("inf"))
        )
        <= 0.70,
        "worst_seed_safe_success": float(
            metrics.get("worst_seed_safe_success", float("-inf"))
        )
        >= 0.10,
        "worst_seed_attack_mitigation": float(
            metrics.get("worst_seed_attack_mitigation", float("-inf"))
        )
        >= 0.15,
        "worst_seed_active_probe_rate": float(
            metrics.get("worst_seed_active_probe_rate", float("-inf"))
        )
        >= 0.10,
        "worst_seed_t1_probe_yield": float(
            metrics.get("worst_seed_t1_probe_yield", float("-inf"))
        )
        >= 0.10,
        "worst_seed_trust_operation_rate": float(
            metrics.get("worst_seed_trust_operation_rate", float("-inf"))
        )
        >= 0.05,
        "worst_seed_t2_betrayal_detection": float(
            metrics.get("worst_seed_t2_betrayal_detection", 0.0)
        )
        > 0.0,
        "worst_seed_memory_operation_rate": float(
            metrics.get("worst_seed_memory_operation_rate", float("-inf"))
        )
        >= 0.05,
        "worst_seed_memory_use_rate": float(
            metrics.get("worst_seed_memory_use_rate", float("-inf"))
        )
        >= 0.03,
        "worst_seed_mitigation_rate": float(
            metrics.get("worst_seed_mitigation_rate", float("-inf"))
        )
        >= 0.15,
        "t4_overresponse_non_regression": float(
            metrics.get("max_t4_overresponse_delta_vs_warmstart", float("inf"))
        )
        <= 0.0,
        "t4_business_cost_non_regression": float(
            metrics.get("max_t4_business_cost_delta_vs_warmstart", float("inf"))
        )
        <= 0.0,
        "seed_task_collapse_count": int(
            metrics.get("seed_task_collapse_count", -1)
        )
        == 0,
        "probe_budget_causality": float(
            metrics.get("probe_budget_1_safe_utility_gain", float("-inf"))
        )
        > 0.0,
        "probe_cost_adjusted_gain": float(
            metrics.get("probe_gain_minus_verification_cost", float("-inf"))
        )
        > 0.0,
        "memory_causality": float(
            metrics.get("memory_causal_utility_gain", float("-inf"))
        )
        > 0.0,
        "benign_overresponse_non_regression": float(
            metrics.get("benign_overresponse_delta", float("inf"))
        )
        <= 0.0,
    }
    failures = tuple(key for key, passed in checks.items() if not passed)
    return CandidateGateVerdict(
        "candidate_gate_a_200x3",
        not failures,
        failures,
        dict(metrics),
        thresholds,
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
        "fixed_task_safe_success_non_decreasing": True,
        "fixed_task_safe_utility_non_decreasing": True,
        "fixed_t1_probe_yield_non_decreasing": True,
        "fixed_t2_betrayal_detection_non_decreasing": True,
        "fixed_t3_poison_success_non_increasing": True,
        "fixed_t3_memory_operation_non_decreasing": True,
        "fixed_t3_memory_use_non_decreasing": True,
        "fixed_t4_overresponse_non_increasing": True,
        "reported_improvement_thresholds": {
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
    for task in ("T1", "T2", "T3", "T4"):
        start_task_metrics = (start.get("by_task_terminal") or {}).get(task)
        task_metrics = (end.get("by_task_terminal") or {}).get(task)
        if not isinstance(task_metrics, Mapping) or int(task_metrics.get("scenario_count", 0)) == 0:
            failures.append(f"missing_task:{task}")
            continue
        if not isinstance(start_task_metrics, Mapping):
            failures.append(f"missing_start_task:{task}")
            continue
        if float(task_metrics.get("safe_success", 0.0)) + 1.0e-12 < float(
            start_task_metrics.get("safe_success", 0.0)
        ):
            failures.append(f"fixed_task_safe_success_decreased:{task}")
        if float(task_metrics.get("safe_utility", float("-inf"))) + 1.0e-12 < float(
            start_task_metrics.get("safe_utility", 0.0)
        ):
            failures.append(f"fixed_task_safe_utility_decreased:{task}")
    if float(end.get("t1_probe_yield", float("-inf"))) + 1.0e-12 < float(
        start.get("t1_probe_yield", 0.0)
    ):
        failures.append("fixed_t1_probe_yield_decreased")
    if float(end.get("t2_betrayal_detection", float("-inf"))) + 1.0e-12 < float(
        start.get("t2_betrayal_detection", 0.0)
    ):
        failures.append("fixed_t2_betrayal_detection_decreased")
    if float(end.get("t3_poison_success", float("inf"))) > float(
        start.get("t3_poison_success", 0.0)
    ) + 1.0e-12:
        failures.append("fixed_t3_poison_success_increased")
    start_execution = dict(start.get("by_task_execution") or {})
    end_execution = dict(end.get("by_task_execution") or {})
    for metric in ("memory_operation_rate", "memory_use_rate"):
        end_value = float(
            (end_execution.get("T3") or {}).get(metric, float("-inf"))
        )
        start_value = float((start_execution.get("T3") or {}).get(metric, 0.0))
        if end_value + 1.0e-12 < start_value:
            failures.append(f"fixed_t3_{metric}_decreased")
    if float(end.get("t4_overresponse", float("inf"))) > float(
        start.get("t4_overresponse", 0.0)
    ) + 1.0e-12:
        failures.append("fixed_t4_overresponse_increased")
    packed = {"round_index": round_index, "start": dict(start), "end": dict(end), "improvements": improvements}
    return CandidateGateVerdict(
        f"candidate_round_{round_index}",
        not failures,
        tuple(failures),
        packed,
        thresholds,
    )
