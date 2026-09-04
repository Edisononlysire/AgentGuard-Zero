"""Frozen candidate-native protocol for the final progressive ablation."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agentguard_zero.candidate.generator import ACTION_FAMILIES, CandidateGenerator
from agentguard_zero.candidate.semantic import semantic_digest
from agentguard_zero.candidate.supervision import policy_consistent_supervision
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.recovery.public_teacher import (
    PROBE_GROUNDING_TIE_TOLERANCE,
    public_state_digest,
)
from agentguard_zero.variants import ExperimentVariant, experiment_variant


PROGRESSIVE_ARM_ORDER = (
    "zero_shot",
    "static_training",
    "verification",
    "active_probing",
    "state_aware",
    "coevolution",
    "ecrg",
)
TRAINED_PROGRESSIVE_ARMS = PROGRESSIVE_ARM_ORDER[1:-1]
K6_PROFILE_QUOTAS = {
    "zero_shot": {"observe": 2, "mitigation": 4},
    "static_training": {"observe": 2, "mitigation": 4},
    "verification": {
        "observe": 1,
        "passive_verification": 2,
        "mitigation": 3,
    },
    "active_probing": {
        "observe": 1,
        "passive_verification": 1,
        "active_probe": 2,
        "mitigation": 2,
    },
    "state_aware": {family: 1 for family in ACTION_FAMILIES},
    "coevolution": {family: 1 for family in ACTION_FAMILIES},
    "ecrg": {family: 1 for family in ACTION_FAMILIES},
}
RUNTIME_VARIANTS = {
    "zero_shot": "static_train",
    "static_training": "static_train",
    "verification": "verification_tools",
    "active_probing": "active_probing",
    "state_aware": "state_aware",
    "coevolution": "state_aware",
    "ecrg": "state_aware",
}


@dataclass(frozen=True)
class ProgressiveArm:
    name: str
    runtime_variant: str
    quotas: Mapping[str, int]
    trained: bool
    ecrg_enabled: bool


def progressive_arm(name: str) -> ProgressiveArm:
    normalized = str(name)
    if normalized not in PROGRESSIVE_ARM_ORDER:
        raise ValueError(f"unsupported progressive arm: {normalized}")
    return ProgressiveArm(
        name=normalized,
        runtime_variant=RUNTIME_VARIANTS[normalized],
        quotas=K6_PROFILE_QUOTAS[normalized],
        trained=normalized in TRAINED_PROGRESSIVE_ARMS,
        ecrg_enabled=normalized == "ecrg",
    )


def candidate_allowed(
    candidate: CandidateOption,
    variant: ExperimentVariant | str,
) -> bool:
    """Return whether an action uses only capabilities enabled in one arm."""

    resolved = experiment_variant(variant) if isinstance(variant, str) else variant
    flags = candidate.action_flags
    if flags.active_probe and not resolved.active_probing:
        return False
    if flags.passive_verification and not resolved.passive_verification:
        return False
    if flags.trust and not resolved.trust_recalibration:
        return False
    if (flags.memory_operation or flags.memory_use) and not resolved.state_layer:
        return False
    return True


def progressive_generator(arm_name: str) -> CandidateGenerator:
    """Create the exact-K=6, capability-filtered runtime generator for an arm."""

    arm = progressive_arm(arm_name)
    return CandidateGenerator(
        min_candidates=6,
        max_candidates=6,
        quotas=arm.quotas,
        experiment_variant=arm.runtime_variant,
    )


def ablate_public_observation(
    observation: Mapping[str, Any],
    variant_name: str,
) -> dict[str, Any]:
    """Project a full public state onto the fields available to one variant."""

    variant = experiment_variant(variant_name)
    projected = copy.deepcopy(dict(observation))
    if variant.state_layer:
        return projected

    defender = dict(projected.get("defender_state") or {})
    history = []
    active_actions = {
        "SourceChallenge",
        "CanaryProbe",
        "DecoyProbe",
        "ShadowActionProbe",
    }
    passive_actions = {"CrossCheck", "QueryProvenance"}
    for row in defender.get("response_history", []) or []:
        if not isinstance(row, Mapping):
            continue
        action = str(row.get("action", ""))
        if action in active_actions and not variant.active_probing:
            continue
        if action in passive_actions and not variant.passive_verification:
            continue
        history.append(copy.deepcopy(dict(row)))
    projected["available_evidence"] = copy.deepcopy(
        list(projected.get("observed_events") or [])
    )
    projected["defender_state"] = {
        "trust": {},
        "memory": {},
        "probe_state": [],
        "response_history": history,
    }
    last = projected.get("last_tool_result") or {}
    tool = str(last.get("tool", "")) if isinstance(last, Mapping) else ""
    if (
        tool in active_actions
        and not variant.active_probing
        or tool in {"LogQuery", "CrossCheck", "ProvenanceCheck", "GraphQuery"}
        and not variant.passive_verification
    ):
        projected["last_tool_result"] = None
    return projected


def _stable_candidate_indices(row: Mapping[str, Any], arm_name: str) -> list[int]:
    arm = progressive_arm(arm_name)
    variant = experiment_variant(arm.runtime_variant)
    candidates = [CandidateOption.from_record(item) for item in row["candidates"]]
    permitted = [
        index
        for index, candidate in enumerate(candidates)
        if candidate_allowed(candidate, variant)
    ]
    by_family = {
        family: sorted(
            (
                index
                for index in permitted
                if candidates[index].action_family == family
            ),
            key=lambda index: candidates[index].semantic_id,
        )
        for family in ACTION_FAMILIES
    }
    selected: list[int] = []
    for family in ACTION_FAMILIES:
        selected.extend(
            by_family[family][: max(0, int(arm.quotas.get(family, 0)))]
        )
    selected_set = set(selected)
    top_up = sorted(
        (index for index in permitted if index not in selected_set),
        key=lambda index: (
            candidates[index].action_family,
            candidates[index].semantic_id,
        ),
    )
    selected.extend(top_up[: 6 - len(selected)])
    selected = selected[:6]
    if len(selected) != 6:
        raise RuntimeError(
            f"{arm_name} record {row.get('record_id')} has only "
            f"{len(selected)} permitted candidates"
        )
    if not any(candidates[index].action_flags.observe_only for index in selected):
        raise RuntimeError("progressive candidate profile lacks Observe")
    return selected


def _variant_target_index(
    row: Mapping[str, Any],
    selected: Sequence[int],
) -> int:
    candidates = [CandidateOption.from_record(item) for item in row["candidates"]]
    q_values = list(map(float, row["teacher_q_values"]))
    core_values = list(map(float, row["teacher_core_q_values"]))
    observe = max(
        (index for index in selected if candidates[index].action_flags.observe_only),
        key=lambda index: (q_values[index], candidates[index].semantic_id),
    )
    eligible = [
        index
        for index in selected
        if core_values[index] >= core_values[observe] - 0.02 - 1.0e-12
    ]
    best = max(
        eligible,
        key=lambda index: (q_values[index], candidates[index].semantic_id),
    )
    last = dict((row.get("public_observation") or {}).get("last_tool_result") or {})
    if bool(last.get("active_probe", False)):
        evidence_id = str(last.get("evidence_id", ""))
        grounded = [
            index
            for index in eligible
            if evidence_id
            and candidates[index].action_flags.trust
            and evidence_id in candidates[index].referenced_ids
            and core_values[index] + 0.02 >= core_values[best]
            and q_values[index] + PROBE_GROUNDING_TIE_TOLERANCE >= q_values[best]
        ]
        if grounded:
            grounded_best = max(
                grounded,
                key=lambda index: (q_values[index], candidates[index].semantic_id),
            )
            if q_values[grounded_best] + 1.0e-12 >= q_values[observe]:
                return grounded_best
    if q_values[best] <= q_values[observe] + 0.05 + 1.0e-12:
        return observe
    return best


def transform_candidate_record(
    row: Mapping[str, Any],
    arm_name: str,
) -> dict[str, Any]:
    """Build a label-blind K=6 view and recompute aligned offline supervision."""

    arm = progressive_arm(arm_name)
    if arm.name in {"zero_shot", "ecrg"}:
        raise ValueError(f"{arm.name} does not define an independent training set")
    selected = _stable_candidate_indices(row, arm_name)
    source_candidates = list(row["candidates"])
    candidates = [CandidateOption.from_record(source_candidates[index]) for index in selected]
    target_source_index = _variant_target_index(row, selected)
    target_index = selected.index(target_source_index)
    q_values = [float(row["teacher_q_values"][index]) for index in selected]
    core_values = [float(row["teacher_core_q_values"][index]) for index in selected]
    observe_index = next(
        index
        for index, candidate in enumerate(candidates)
        if candidate.action_flags.observe_only
    )
    supervision = policy_consistent_supervision(
        q_values,
        core_values,
        target_index=target_index,
        observe_index=observe_index,
        core_tolerance=0.02,
        temperature=0.1,
    )
    target = candidates[target_index]
    same_family = sorted(
        (
            index
            for index, candidate in enumerate(candidates)
            if index != target_index
            and candidate.action_family == target.action_family
        ),
        key=lambda index: (-supervision.policy_scores[index], candidates[index].semantic_id),
    )
    other = sorted(
        (index for index in range(6) if index != target_index and index not in same_family),
        key=lambda index: (-supervision.policy_scores[index], candidates[index].semantic_id),
    )
    hard_indices = (same_family + other)[:4]
    source_chain = copy.deepcopy(dict(row.get("probe_chain_target") or {}))
    probe_ids = {
        str(value)
        for value in (
            source_chain.get("probe_evidence_ids")
            or [source_chain.get("probe_evidence_id")]
        )
        if value
    }
    source_chain.update(
        {
            "evidence_used_by_target": bool(probe_ids & set(target.referenced_ids)),
            "followup_flags": target.action_flags.to_dict(),
            "followup_nonobserve": not target.action_flags.observe_only,
        }
    )
    transformed = copy.deepcopy(dict(row))
    transformed["record_id"] = hashlib.sha256(
        f"progressive:{arm_name}:{row['record_id']}".encode("utf-8")
    ).hexdigest()
    transformed["public_observation"] = ablate_public_observation(
        dict(row["public_observation"]), arm.runtime_variant
    )
    transformed["public_state_digest"] = public_state_digest(
        transformed["public_observation"]
    )
    transformed["semantic_public_state_digest"] = semantic_digest(
        transformed["public_observation"]
    )
    transformed["candidates"] = [source_candidates[index] for index in selected]
    transformed["teacher_q_values"] = q_values
    transformed["teacher_core_q_values"] = core_values
    transformed["teacher_probabilities"] = list(supervision.probabilities)
    transformed["teacher_policy_scores"] = list(supervision.policy_scores)
    transformed["teacher_eligible_mask"] = list(supervision.eligible)
    transformed["teacher_retained_mask"] = list(supervision.retained)
    transformed["teacher_acceptable_mask"] = list(supervision.acceptable)
    transformed["outcome_targets"] = [
        row["outcome_targets"][index] for index in selected
    ]
    transformed["target_candidate_key"] = target.candidate_key
    transformed["target_candidate_id"] = target.candidate_key
    transformed["target_semantic_id"] = target.semantic_id
    transformed["target_family"] = target.action_family
    transformed["teacher_full_q_best_candidate_id"] = candidates[
        max(
            range(6),
            key=lambda index: (q_values[index], candidates[index].semantic_id),
        )
    ].semantic_id
    transformed["teacher_full_q_best_value"] = max(q_values)
    transformed["hard_negative_candidate_ids"] = [
        candidates[index].candidate_id for index in hard_indices
    ]
    transformed["hard_negative_reasons"] = {
        candidates[index].candidate_id: [
            "progressive_same_family"
            if candidates[index].action_family == target.action_family
            else "progressive_high_policy_negative"
        ]
        for index in hard_indices
    }
    allowed_keys = {candidate.candidate_id for candidate in candidates}
    transformed["trajectory_negative_candidate_ids"] = [
        value
        for value in row.get("trajectory_negative_candidate_ids", [])
        if value in allowed_keys and value != target.candidate_id
    ]
    transformed["probe_chain_target"] = source_chain
    transformed["protocol_role"] = "progressive_ablation_train"
    audit = copy.deepcopy(dict(row.get("audit") or {}))
    audit.update(
        {
            "progressive_arm": arm.name,
            "runtime_variant": arm.runtime_variant,
            "source_record_id": str(row["record_id"]),
            "candidate_selection_uses_teacher_labels": False,
            "target_recomputed_after_capability_filter": True,
        }
    )
    transformed["audit"] = audit
    return transformed
