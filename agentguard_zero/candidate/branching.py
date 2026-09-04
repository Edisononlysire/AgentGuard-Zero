"""Supervision contract for the V10 short/long-horizon candidate ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agentguard_zero.candidate.types import CandidateOption


SHARED_LOCAL_LONG_ARCHITECTURE = "shared_local_long_v1"
SHARED_LOCAL_LONG_ARCHITECTURE_V2 = "shared_local_long_v2"
SHARED_LOCAL_LONG_ARCHITECTURES = (
    SHARED_LOCAL_LONG_ARCHITECTURE,
    SHARED_LOCAL_LONG_ARCHITECTURE_V2,
)
BRANCH_NAMES = ("local", "t3_memory", "t4_budget")
T1_LOCAL_PHASES = (
    "low_cost_passive_anchor",
    "probe_needed",
    "probe_grounding",
    "mitigation_ready",
)
T2_LOCAL_PHASES = (
    "low_cost_passive_anchor",
    "betrayal_probe_needed",
    "trust_update_needed",
    "mitigation_ready",
)
T3_MEMORY_PHASES = (
    "ltm_risk_claim_wait",
    "ltm_ingest_needed",
    "ltm_verification_needed",
    "ltm_evidence_grounding",
    "ltm_transition_needed",
    "ltm_grounded_mitigation_ready",
    "ltm_recovery_verification_needed",
    "ltm_recovery_grounding",
    "ltm_rejected_or_recovery",
    "ltm_recovery_promote",
)
T3_MEMORY_PHASES_V2 = (
    "ltm_risk_claim_wait",
    "ltm_ingest_needed",
    "ltm_retrieval_wait",
    "ltm_verification_needed",
    "ltm_evidence_grounding",
    "ltm_transition_needed",
    "ltm_grounded_mitigation_ready",
    "post_mitigation_stop",
    "ltm_recovery_verification_needed",
    "ltm_recovery_grounding",
    "ltm_rejected_or_recovery",
    "ltm_recovery_promote",
)
T4_BUDGET_PHASES = (
    "impact_probe_needed",
    "impact_accumulation_budget_check",
    "profile_verification_needed",
    "profile_update_needed",
    "profile_grounded_mitigation_ready",
    "post_mitigation_stop",
)
T4_BUDGET_PHASES_V2 = (
    "impact_probe_needed",
    "impact_accumulation_budget_check",
    "profile_history_accumulation",
    "profile_verification_needed",
    "profile_update_needed",
    "profile_grounded_mitigation_ready",
    "post_mitigation_stop",
)
PHASE_NAMES = (
    "local_action",
    *(f"t3:{name}" for name in T3_MEMORY_PHASES),
    *(f"t4:{name}" for name in T4_BUDGET_PHASES),
)
PHASE_NAMES_V2 = (
    *(f"t1:{name}" for name in T1_LOCAL_PHASES),
    *(f"t2:{name}" for name in T2_LOCAL_PHASES),
    *(f"t3:{name}" for name in T3_MEMORY_PHASES_V2),
    *(f"t4:{name}" for name in T4_BUDGET_PHASES_V2),
)
LOCAL_PHASE_INDICES_V2 = tuple(
    index
    for index, name in enumerate(PHASE_NAMES_V2)
    if name.startswith(("t1:", "t2:"))
)
LONG_PHASE_INDICES_V2 = tuple(
    index
    for index, name in enumerate(PHASE_NAMES_V2)
    if name.startswith(("t3:", "t4:"))
)
MEMORY_OPERATION_NAMES = (
    "observe",
    "ingest",
    "verify",
    "trust_support",
    "trust_contradict",
    "trust_recover",
    "trust_other",
    "promote",
    "reject",
    "reopen",
    "memory_use",
    "mitigation",
    "memory_use_mitigation",
)
BUDGET_TARGET_NAMES = (
    "cumulative_estimated_cost",
    "remaining_business_budget",
    "budget_checked",
    "mitigation_ready",
    "stop_required",
)


@dataclass(frozen=True)
class BranchSupervision:
    """One record's public, non-oracle routing and auxiliary targets."""

    branch_index: int
    phase_index: int
    memory_operation_index: int
    budget_targets: tuple[float, ...]
    budget_mask: tuple[bool, ...]


def _bounded(value: Any) -> float:
    return min(1.0, max(0.0, float(value or 0.0)))


def public_contract_branch(observation: Mapping[str, Any]) -> str:
    """Route from public execution requirements, never hidden task labels.

    V2 uses this deterministic outer route so long-horizon gradients cannot
    contaminate the original T1/T2 local decision heads.  The learned phase
    gate remains responsible for the internal T3/T4 phase.
    """

    requirements = (
        ((observation.get("defense_context") or {}).get("response_requirements"))
        or {}
    )
    if not isinstance(requirements, Mapping):
        return "local"
    if requirements.get("impact_probe_required_for_mitigation") or requirements.get(
        "profile_memory_use_required_for_mitigation"
    ):
        return "t4_budget"
    if requirements.get("memory_use_required_for_mitigation"):
        return "t3_memory"
    return "local"


def _public_tool_names(observation: Mapping[str, Any]) -> set[str]:
    names = {str((observation.get("last_tool_result") or {}).get("tool", ""))}
    names.update(
        str((row.get("content") or {}).get("tool", ""))
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, Mapping)
    )
    return names - {""}


def _public_trust_is_grounded(observation: Mapping[str, Any]) -> bool:
    trust = ((observation.get("defender_state") or {}).get("trust") or {})

    def assessed(value: Any) -> bool:
        if isinstance(value, Mapping):
            if str(value.get("status", "unassessed")) not in {
                "",
                "unassessed",
                "uncertain",
            }:
                return True
            return any(assessed(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(assessed(item) for item in value)
        return False

    return assessed(trust)


def public_local_phase(observation: Mapping[str, Any], task_id: str) -> str:
    """Derive T1/T2 phase labels entirely from the public pre-state."""

    tools = _public_tool_names(observation)
    requirements = (
        ((observation.get("defense_context") or {}).get("response_requirements"))
        or {}
    )
    if (
        requirements.get("passive_check_required_before_active_probe")
        and not tools & {"CrossCheck", "ProvenanceCheck", "GraphQuery"}
    ):
        return "low_cost_passive_anchor"
    if task_id == "T1":
        if not tools & {
            "SourceChallenge",
            "CanaryProbe",
            "DecoyProbe",
            "ShadowActionProbe",
        }:
            return "probe_needed"
        return (
            "mitigation_ready"
            if _public_trust_is_grounded(observation)
            else "probe_grounding"
        )
    if task_id == "T2":
        if not tools & {
            "SourceChallenge",
            "CanaryProbe",
            "DecoyProbe",
            "ShadowActionProbe",
        }:
            return "betrayal_probe_needed"
        return (
            "mitigation_ready"
            if _public_trust_is_grounded(observation)
            else "trust_update_needed"
        )
    raise ValueError(f"local phase requires T1/T2, got {task_id}")


def memory_operation_name(candidate: CandidateOption) -> str:
    """Return the candidate's semantic T3 operation.

    Verification is intentionally checked before generic memory use.  The V1
    implementation treated every non-Observe response as mitigation, which
    mislabeled memory-backed CrossCheck/ProvenanceCheck candidates as
    ``memory_use_mitigation``.
    """

    packet = candidate.compiled_packet
    operations = packet.get("memory_operations") or packet.get("memory_operation") or []
    if isinstance(operations, Mapping):
        operations = [operations]
    if operations:
        operation = str(operations[0].get("op", ""))
        if operation in {"ingest", "promote", "reject", "reopen"}:
            return operation

    tool = packet.get("tool_call") or {}
    if candidate.action_flags.passive_verification or candidate.action_flags.active_probe:
        return "verify"
    if str(tool.get("name", "None")) != "None":
        return "verify"

    trust = packet.get("trust_operations") or packet.get("trust_operation") or []
    if trust:
        if isinstance(trust, Mapping):
            trust = [trust]
        operation = str(trust[0].get("op", ""))
        return (
            f"trust_{operation}"
            if operation in {"support", "contradict", "recover"}
            else "trust_other"
        )

    uses = packet.get("memory_usage") or packet.get("memory_use") or []
    if isinstance(uses, Mapping):
        uses = [uses]
    mitigates = bool(candidate.action_flags.mitigation)
    if uses:
        return "memory_use_mitigation" if mitigates else "memory_use"
    if mitigates:
        return "mitigation"
    return "observe"


# Backward-compatible private name used by existing imports/tests.
_memory_operation = memory_operation_name


def derive_branch_supervision(
    record: Mapping[str, Any],
    target: CandidateOption,
    *,
    architecture: str = SHARED_LOCAL_LONG_ARCHITECTURE,
) -> BranchSupervision:
    """Derive all labels from public state and the admitted causal target."""

    task_id = str(record.get("task_id", ""))
    causal_phase = str(record.get("causal_phase", ""))
    public_observation = record.get("public_observation") or {}
    if architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2:
        public_branch = public_contract_branch(public_observation)
        expected_branch = {
            "T1": "local",
            "T2": "local",
            "T3": "t3_memory",
            "T4": "t4_budget",
        }.get(task_id)
        if public_branch != expected_branch:
            raise ValueError(
                "task supervision disagrees with the public execution contract: "
                f"task={task_id} public_branch={public_branch}"
            )
        branch_index = BRANCH_NAMES.index(public_branch)
    if task_id in {"T1", "T2"}:
        if architecture != SHARED_LOCAL_LONG_ARCHITECTURE_V2:
            branch_index = BRANCH_NAMES.index("local")
        phase_name = (
            f"{task_id.lower()}:"
            f"{public_local_phase(record.get('public_observation') or {}, task_id)}"
            if architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
            else "local_action"
        )
    elif task_id == "T3":
        valid_t3_phases = (
            T3_MEMORY_PHASES_V2
            if architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
            else T3_MEMORY_PHASES
        )
        if causal_phase not in valid_t3_phases:
            raise ValueError(f"unknown T3 causal phase: {causal_phase}")
        if architecture != SHARED_LOCAL_LONG_ARCHITECTURE_V2:
            branch_index = BRANCH_NAMES.index("t3_memory")
        phase_name = f"t3:{causal_phase}"
    elif task_id == "T4":
        valid_t4_phases = (
            T4_BUDGET_PHASES_V2
            if architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
            else T4_BUDGET_PHASES
        )
        if causal_phase not in valid_t4_phases:
            raise ValueError(f"unknown T4 causal phase: {causal_phase}")
        if architecture != SHARED_LOCAL_LONG_ARCHITECTURE_V2:
            branch_index = BRANCH_NAMES.index("t4_budget")
        phase_name = f"t4:{causal_phase}"
    else:
        raise ValueError(f"unknown task for branch supervision: {task_id}")

    memory_index = -100
    if task_id == "T3":
        memory_index = MEMORY_OPERATION_NAMES.index(memory_operation_name(target))

    budget_targets = (0.0,) * len(BUDGET_TARGET_NAMES)
    budget_mask = (False,) * len(BUDGET_TARGET_NAMES)
    if task_id == "T4":
        observation = record.get("public_observation") or {}
        defender = observation.get("defender_state") or {}
        impact = defender.get("business_impact_memory") or {}
        defense = observation.get("defense_context") or {}
        estimate_count = int(impact.get("impact_estimate_count", 0) or 0)
        budget_targets = (
            _bounded(impact.get("cumulative_estimated_cost", 0.0)),
            _bounded(
                defense.get(
                    "remaining_business_budget",
                    impact.get("remaining_business_budget", 0.0),
                )
            ),
            float(estimate_count > 0),
            float(causal_phase == "profile_grounded_mitigation_ready"),
            float(causal_phase == "post_mitigation_stop"),
        )
        budget_mask = (True,) * len(BUDGET_TARGET_NAMES)

    phase_names = (
        PHASE_NAMES_V2
        if architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
        else PHASE_NAMES
    )
    return BranchSupervision(
        branch_index=branch_index,
        phase_index=phase_names.index(phase_name),
        memory_operation_index=memory_index,
        budget_targets=budget_targets,
        budget_mask=budget_mask,
    )


def phase_branch_probabilities(
    phase_logits: Any, *, phase_names: tuple[str, ...] = PHASE_NAMES
) -> Any:
    """Aggregate learned phase probabilities into local/T3/T4 routing weights."""

    import torch

    probabilities = torch.softmax(phase_logits, dim=-1)
    if int(probabilities.shape[-1]) != len(phase_names):
        raise ValueError("phase logits do not match the V10 phase contract")
    local_indices = [
        index
        for index, name in enumerate(phase_names)
        if name == "local_action" or name.startswith(("t1:", "t2:"))
    ]
    t3_indices = [
        index for index, name in enumerate(phase_names) if name.startswith("t3:")
    ]
    t4_indices = [
        index for index, name in enumerate(phase_names) if name.startswith("t4:")
    ]
    return torch.stack(
        (
            probabilities[..., local_indices].sum(dim=-1),
            probabilities[..., t3_indices].sum(dim=-1),
            probabilities[..., t4_indices].sum(dim=-1),
        ),
        dim=-1,
    )
