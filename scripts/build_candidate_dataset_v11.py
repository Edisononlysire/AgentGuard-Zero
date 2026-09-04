#!/usr/bin/env python3
"""Build public-state listwise candidate supervision from the robust Teacher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.generator import CandidateGenerator, DEFAULT_QUOTAS
from agentguard_zero.candidate.semantic import semantic_digest
from agentguard_zero.candidate.supervision import policy_consistent_supervision
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.canonical_scenarios import (
    canonical_recovery_group,
    canonical_recovery_suite,
)
from agentguard_zero.recovery.public_teacher import (
    ActionCandidate,
    PublicStateRobustTeacher,
    probe_evidence_trust_operation,
    public_state_digest,
)
from agentguard_zero.recovery.source_counterfactuals import (
    counterfactual_groups,
    load_source_scenarios,
)
from agentguard_zero.runtime_policy import HIGH_IMPACT_ACTIONS
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now
from agentguard_zero.world.public_projector import (
    assert_public,
    project_event,
    project_public,
)


ACTIVE_PROBE_TOOLS = {
    "SourceChallenge",
    "CanaryProbe",
    "DecoyProbe",
    "ShadowActionProbe",
}
IMPACT_PROBE_TOOLS = {"BusinessImpactEstimator", "ShadowActionProbe"}
SOURCE_PROFILE_PROBE_TOOLS = {"SourceChallenge", "CanaryProbe"}
MEMORY_VERIFICATION_TOOLS = {
    "CrossCheck",
    "ProvenanceCheck",
    *SOURCE_PROFILE_PROBE_TOOLS,
}
LOW_COST_PASSIVE_TOOLS = {"CrossCheck", "ProvenanceCheck", "GraphQuery"}
MITIGATING_ACTIONS = {
    "DeployDecoy",
    "LimitSession",
    "ShadowBlock",
    "Isolate",
    "Restore",
    "Remove",
}


def _public_tool_names(observation: dict[str, Any]) -> set[str]:
    names = {str((observation.get("last_tool_result") or {}).get("tool", ""))}
    names.update(
        str((row.get("content") or {}).get("tool", ""))
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, dict)
    )
    return {name for name in names if name}


def _memory_statuses(observation: dict[str, Any]) -> dict[str, str]:
    memory = ((observation.get("defender_state") or {}).get("memory") or {})
    statuses: dict[str, str] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            memory_id = str(value.get("memory_id", ""))
            if memory_id:
                statuses[memory_id] = str(value.get("status", "quarantined"))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(memory)
    return statuses


def _memory_rows(observation: dict[str, Any]) -> list[dict[str, Any]]:
    memory = ((observation.get("defender_state") or {}).get("memory") or {})
    rows: dict[str, dict[str, Any]] = {}
    for bucket in (
        "retrieved_confirmed",
        "retrieved_quarantined",
        "rejected_warnings",
    ):
        for row in memory.get(bucket, []) or []:
            if not isinstance(row, dict):
                continue
            memory_id = str(row.get("memory_id", ""))
            if memory_id and not memory_id.startswith("profile:"):
                rows[memory_id] = row
    return list(rows.values())


def _is_memory_update_candidate(
    observation: dict[str, Any], candidate: CandidateOption
) -> bool:
    operations = candidate.compiled_packet.get("memory_operations") or []
    if not operations or str(operations[0].get("op", "")) != "ingest":
        return False
    operation = operations[0]
    claim = operation.get("claim") or {}
    refs = set(map(str, operation.get("evidence_refs", []) or []))
    for row in _memory_rows(observation):
        if claim != (row.get("claim") or {}):
            continue
        if refs - set(map(str, row.get("evidence_refs", []) or [])):
            return True
    return False


def _trust_is_grounded(observation: dict[str, Any]) -> bool:
    trust = ((observation.get("defender_state") or {}).get("trust") or {})

    def assessed(value: Any) -> bool:
        if isinstance(value, dict):
            if str(value.get("status", "unassessed")) not in {
                "",
                "unassessed",
                "uncertain",
            }:
                return True
            return any(assessed(item) for item in value.values())
        if isinstance(value, list):
            return any(assessed(item) for item in value)
        return False

    return assessed(trust)


def _claim_trust_statuses(observation: dict[str, Any]) -> dict[str, str]:
    trust = ((observation.get("defender_state") or {}).get("trust") or {})
    statuses: dict[str, str] = {
        str(event_id): str(row.get("status", ""))
        for event_id, row in (trust.get("current_claim_trust") or {}).items()
        if isinstance(row, dict) and str(row.get("status", ""))
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            event_id = str(value.get("event_id", ""))
            status = str(value.get("status", ""))
            if event_id and status:
                statuses[event_id] = status
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(trust)
    return statuses


def _probe_evidence_ids(observation: dict[str, Any]) -> set[str]:
    return {
        str(row.get("evidence_id", ""))
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, dict)
        and str((row.get("content") or {}).get("tool", ""))
        in ACTIVE_PROBE_TOOLS | IMPACT_PROBE_TOOLS
        and str(row.get("evidence_id", ""))
    }


def _verification_evidence_ids(observation: dict[str, Any]) -> set[str]:
    return {
        str(row.get("evidence_id", ""))
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, dict)
        and str((row.get("content") or {}).get("tool", ""))
        in MEMORY_VERIFICATION_TOOLS
        and str(row.get("evidence_id", ""))
    }


def _required_probe_trust_operation(observation: dict[str, Any]) -> str:
    evidence = [
        row
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, dict)
        and str((row.get("content") or {}).get("tool", ""))
        in MEMORY_VERIFICATION_TOOLS
    ]
    if not evidence:
        return "challenge"
    latest = max(
        evidence,
        key=lambda row: (
            int(row.get("available_at", -1)),
            str(row.get("evidence_id", "")),
        ),
    )
    return probe_evidence_trust_operation(latest)


def _memory_event_ids(observation: dict[str, Any]) -> set[str]:
    memory = _memory_statuses(observation)
    evidence_to_event = {
        str(row.get("evidence_id", "")): str(row.get("event_id", ""))
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, dict)
    }
    event_ids: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if str(value.get("memory_id", "")) in memory:
                event_ids.update(
                    evidence_to_event.get(str(reference), "")
                    for reference in value.get("evidence_refs", []) or []
                )
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(((observation.get("defender_state") or {}).get("memory") or {}))
    return {event_id for event_id in event_ids if event_id}


def _authorized_mitigation_seen(observation: dict[str, Any]) -> bool:
    return any(
        isinstance(row, dict)
        and bool(row.get("authorized", False))
        and str(row.get("action", "")) in MITIGATING_ACTIONS
        for row in (
            ((observation.get("defender_state") or {}).get("response_history") or [])
        )
    )


def _impact_estimates(observation: dict[str, Any]) -> list[dict[str, Any]]:
    memory = (
        ((observation.get("defender_state") or {}).get("business_impact_memory") or {})
    )
    return [
        row
        for row in memory.get("impact_estimates", []) or []
        if isinstance(row, dict)
    ]


def _evidence_root_count(
    observation: dict[str, Any], evidence_ids: set[str]
) -> int:
    roots: set[str] = set()
    for row in observation.get("available_evidence", []) or []:
        if not isinstance(row, dict) or str(row.get("evidence_id", "")) not in evidence_ids:
            continue
        roots.update(
            map(
                str,
                row.get("root_source_ids") or [row.get("source_id", "")],
            )
        )
    return len(roots - {""})


def _evidence_covers_memory_event(
    observation: dict[str, Any], evidence_ids: set[str], memory_event_ids: set[str]
) -> bool:
    if not memory_event_ids:
        return True
    return any(
        isinstance(row, dict)
        and str(row.get("evidence_id", "")) in evidence_ids
        and str(row.get("event_id", "")) in memory_event_ids
        for row in observation.get("available_evidence", []) or []
    )


def _memory_has_independent_corroboration(
    observation: dict[str, Any]
) -> bool:
    evidence = observation.get("available_evidence", []) or []
    for memory_row in _memory_rows(observation):
        claim = memory_row.get("claim") or {}
        matching_ids = {
            str(row.get("evidence_id", ""))
            for row in evidence
            if isinstance(row, dict)
            and ((row.get("content") or {}).get("claim_semantics") or {}) == claim
            and str(row.get("evidence_id", ""))
        }
        if _evidence_root_count(observation, matching_ids) >= 2:
            return True
    return False


def _fresh_recovery_verification_evidence_ids(
    observation: dict[str, Any]
) -> set[str]:
    memory_event_ids = _memory_event_ids(observation)
    return {
        str(row.get("evidence_id", ""))
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, dict)
        and str((row.get("content") or {}).get("tool", ""))
        in MEMORY_VERIFICATION_TOOLS
        and str(row.get("event_id", "")) not in memory_event_ids
        and str(row.get("evidence_id", ""))
    }


def _latest_recovery_verification(
    observation: dict[str, Any]
) -> dict[str, Any] | None:
    fresh_ids = _fresh_recovery_verification_evidence_ids(observation)
    rows = [
        row
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, dict) and str(row.get("evidence_id", "")) in fresh_ids
    ]
    # Evidence snapshots preserve append order while deliberately omitting
    # hidden/internal timestamps.  The last matching public record is therefore
    # the newest recovery verification available to the policy.
    return rows[-1] if rows else None


def _latest_recovery_verification_is_grounded(
    observation: dict[str, Any], evidence: dict[str, Any]
) -> bool:
    event_id = str(evidence.get("event_id", ""))
    evidence_id = str(evidence.get("evidence_id", ""))
    claim = (
        (
            (
                ((observation.get("defender_state") or {}).get("trust") or {}).get(
                    "current_claim_trust", {}
                )
                or {}
            ).get(event_id, {})
        )
        or {}
    )
    references = {
        str(reference)
        for key in (
            "evidence_refs",
            "support_evidence_refs",
            "contradiction_evidence_refs",
        )
        for reference in claim.get(key, []) or []
    }
    return evidence_id in references


def _causal_phase(observation: dict[str, Any], task_id: str) -> str:
    tools = _public_tool_names(observation)
    memory = _memory_statuses(observation)
    trust_grounded = _trust_is_grounded(observation)
    requirements = (
        (observation.get("defense_context") or {}).get("response_requirements")
        or {}
    )
    if task_id in {"T1", "T2", "T3", "T4"} and _authorized_mitigation_seen(
        observation
    ):
        return "post_mitigation_stop"
    if (
        task_id in {"T1", "T2"}
        and requirements.get("passive_check_required_before_active_probe")
        and not tools & LOW_COST_PASSIVE_TOOLS
    ):
        return "low_cost_passive_anchor"
    if task_id == "T1":
        if not tools & ACTIVE_PROBE_TOOLS:
            return "probe_needed"
        if not trust_grounded:
            return "probe_grounding"
        return "mitigation_ready"
    if task_id == "T2":
        if not tools & ACTIVE_PROBE_TOOLS:
            return "betrayal_probe_needed"
        if not trust_grounded:
            return "trust_update_needed"
        return "mitigation_ready"
    if task_id == "T3":
        if _authorized_mitigation_seen(observation):
            return "post_mitigation_stop"
        if not memory:
            risk_objects = {
                "high",
                "active",
                "compromised",
                "credential_theft",
                "exfiltration",
                "lateral_movement",
                "persistence",
            }
            has_risk_claim = any(
                str((row.get("claim_semantics") or {}).get("object", ""))
                in risk_objects
                for row in observation.get("observed_events", []) or []
                if isinstance(row, dict)
            )
            return "ltm_ingest_needed" if has_risk_claim else "ltm_risk_claim_wait"
        if any(status == "rejected" for status in memory.values()):
            latest_recovery = _latest_recovery_verification(observation)
            if latest_recovery is None:
                return "ltm_recovery_verification_needed"
            if probe_evidence_trust_operation(latest_recovery) not in {
                "support",
                "recover",
            }:
                return "ltm_recovery_verification_needed"
            if not _latest_recovery_verification_is_grounded(
                observation, latest_recovery
            ):
                return "ltm_recovery_grounding"
            event_id = str(latest_recovery.get("event_id", ""))
            claim = (
                (
                    (
                        (
                            (observation.get("defender_state") or {}).get("trust")
                            or {}
                        ).get("current_claim_trust", {})
                        or {}
                    ).get(event_id, {})
                )
                or {}
            )
            if str(claim.get("status", "unassessed")) != "supported":
                return "ltm_recovery_grounding"
            return "ltm_rejected_or_recovery"
        if any(
            str(row.get("status", "")) == "quarantined"
            and any(
                str(transition.get("op", "")) == "reopen"
                for transition in row.get("transition_history", []) or []
                if isinstance(transition, dict)
            )
            for row in _memory_rows(observation)
        ):
            return "ltm_recovery_promote"
        if not _memory_has_independent_corroboration(observation):
            return "ltm_retrieval_wait"
        if not tools & MEMORY_VERIFICATION_TOOLS:
            return "ltm_verification_needed"
        if not trust_grounded:
            return "ltm_evidence_grounding"
        if not any(
            str(row.get("status", "")) == "confirmed"
            for row in _memory_rows(observation)
        ):
            return "ltm_transition_needed"
        if any(status == "confirmed" for status in memory.values()):
            return "ltm_grounded_mitigation_ready"
        return "ltm_rejected_or_recovery"
    if task_id == "T4":
        if _authorized_mitigation_seen(observation):
            return "post_mitigation_stop"
        impact_estimates = _impact_estimates(observation)
        if not impact_estimates:
            return "impact_probe_needed"
        last_estimate_time = max(
            int(row.get("time", -1)) for row in impact_estimates
        )
        if len(impact_estimates) == 1 and int(
            observation.get("time", 0)
        ) <= last_estimate_time + 1:
            return "impact_accumulation_budget_check"
        profile_rows = list(
            (((observation.get("defender_state") or {}).get("memory") or {}).get(
                "retrieved_profiles", []
            ))
            or []
        )
        if not profile_rows:
            return "profile_history_accumulation"
        if not tools & SOURCE_PROFILE_PROBE_TOOLS:
            return "profile_verification_needed"
        if not trust_grounded:
            return "profile_update_needed"
        return "profile_grounded_mitigation_ready"
    return "unknown"


def _causal_target(
    observation: dict[str, Any],
    task_id: str,
    candidates: list[CandidateOption],
    q_values: list[float],
) -> tuple[str, str] | None:
    """Return the public-chain family and best Teacher-scored member of it."""

    phase = _causal_phase(observation, task_id)
    family = {
        "low_cost_passive_anchor": "passive_verification",
        "probe_needed": "active_probe",
        "probe_grounding": "trust",
        "mitigation_ready": "mitigation",
        "betrayal_probe_needed": "active_probe",
        "trust_update_needed": "trust",
        "ltm_risk_claim_wait": "observe",
        "ltm_ingest_needed": "memory",
        "ltm_retrieval_wait": "observe",
        "ltm_verification_needed": "passive_verification",
        "ltm_evidence_grounding": "trust",
        "ltm_transition_needed": "memory",
        "ltm_recovery_verification_needed": "passive_verification",
        "ltm_recovery_grounding": "trust",
        "ltm_recovery_promote": "memory",
        "ltm_grounded_mitigation_ready": "mitigation",
        "ltm_rejected_or_recovery": "memory",
        "impact_probe_needed": "passive_verification",
        "impact_accumulation_budget_check": "observe",
        "profile_history_accumulation": "observe",
        "profile_verification_needed": "active_probe",
        "profile_update_needed": "trust",
        "profile_grounded_mitigation_ready": "mitigation",
        "post_mitigation_stop": "observe",
    }.get(phase)
    if family is None:
        return None
    indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.action_family == family
    ]
    if phase in {"probe_needed", "betrayal_probe_needed"}:
        specialized = [
            index
            for index in indices
            if str((candidates[index].compiled_packet.get("tool_call") or {}).get("name", ""))
            in {"SourceChallenge", "CanaryProbe"}
        ]
        indices = specialized or indices
    elif phase == "low_cost_passive_anchor":
        specialized = [
            index
            for index in indices
            if str(
                (candidates[index].compiled_packet.get("tool_call") or {}).get(
                    "name", ""
                )
            )
            in {"ProvenanceCheck", "GraphQuery"}
            and not candidates[index].compiled_packet.get("trust_operations")
        ]
        indices = specialized or indices
    elif phase in {
        "probe_grounding",
        "trust_update_needed",
        "ltm_evidence_grounding",
        "ltm_recovery_grounding",
        "profile_update_needed",
    }:
        probe_ids = (
            _fresh_recovery_verification_evidence_ids(observation)
            if phase == "ltm_recovery_grounding"
            else _verification_evidence_ids(observation)
            if phase == "ltm_evidence_grounding"
            else _probe_evidence_ids(observation)
        )
        latest_recovery = _latest_recovery_verification(observation)
        required_op = (
            (
                "recover"
                if probe_evidence_trust_operation(latest_recovery) == "support"
                else probe_evidence_trust_operation(latest_recovery)
            )
            if phase == "ltm_recovery_grounding"
            and latest_recovery is not None
            else _required_probe_trust_operation(observation)
        )
        specialized = [
            index
            for index in indices
            if bool(set(candidates[index].referenced_ids) & probe_ids)
            and any(
                str(operation.get("op", "")) == required_op
                for operation in candidates[index].compiled_packet.get(
                    "trust_operations", []
                )
                if isinstance(operation, dict)
            )
        ]
        indices = (
            specialized
            if phase
            in {
                "ltm_evidence_grounding",
                "ltm_recovery_grounding",
                "profile_update_needed",
            }
            else specialized or indices
        )
    elif phase == "ltm_ingest_needed":
        risk_objects = {
            "high",
            "active",
            "compromised",
            "credential_theft",
            "exfiltration",
            "lateral_movement",
            "persistence",
        }
        specialized = [
            index
            for index in indices
            if str(
                ((candidates[index].compiled_packet.get("memory_operations") or [{}])[0].get("claim") or {}).get(
                    "object", ""
                )
            )
            in risk_objects
        ]
        indices = specialized or indices
    elif phase in {"ltm_transition_needed", "ltm_recovery_promote"}:
        desired_operations = (
            {"promote"}
            if phase == "ltm_recovery_promote"
            else {"promote", "reject"}
        )
        specialized = [
            index
            for index in indices
            if str(
                (
                    candidates[index].compiled_packet.get("memory_operations")
                    or [{}]
                )[0].get("op", "")
            )
            in desired_operations
        ]
        indices = specialized
    elif phase == "ltm_rejected_or_recovery":
        specialized = [
            index
            for index in indices
            if str(
                (
                    candidates[index].compiled_packet.get("memory_operations")
                    or [{}]
                )[0].get("op", "")
            )
            == "reopen"
        ]
        indices = specialized
    elif phase in {
        "ltm_verification_needed",
        "ltm_recovery_verification_needed",
        "profile_verification_needed",
    }:
        memory_event_ids = _memory_event_ids(observation)
        specialized = [
            index
            for index in indices
            if str((candidates[index].compiled_packet.get("tool_call") or {}).get("name", ""))
            in (
                {"CrossCheck"}
                if phase
                in {"ltm_verification_needed", "ltm_recovery_verification_needed"}
                else {"SourceChallenge", "CanaryProbe"}
            )
            and not candidates[index].compiled_packet.get("trust_operations")
            and (
                phase
                not in {"ltm_verification_needed", "ltm_recovery_verification_needed"}
                or (
                    (
                        phase == "ltm_recovery_verification_needed"
                        or _evidence_covers_memory_event(
                            observation,
                            set(
                                map(
                                    str,
                                    (
                                        (
                                            candidates[index].compiled_packet.get(
                                                "tool_call"
                                            )
                                            or {}
                                        ).get("args", {})
                                        or {}
                                    ).get("evidence_ids", [])
                                    or [],
                                )
                            ),
                            memory_event_ids,
                        )
                    )
                    and _evidence_root_count(
                        observation,
                        set(
                            map(
                                str,
                                (
                                    (
                                        candidates[index].compiled_packet.get(
                                            "tool_call"
                                        )
                                        or {}
                                    ).get("args", {})
                                    or {}
                                ).get("evidence_ids", [])
                                or [],
                            )
                        ),
                    )
                    >= 2
                    and (
                        phase != "ltm_recovery_verification_needed"
                        or str(
                            (
                                candidates[index].compiled_packet.get("tool_call")
                                or {}
                            ).get("args", {}).get("event_id", "")
                        )
                        not in memory_event_ids
                    )
                )
            )
        ]
        indices = specialized
    elif phase in {
        "ltm_grounded_mitigation_ready",
        "profile_grounded_mitigation_ready",
    }:
        specialized = [
            index
            for index in indices
            if candidates[index].action_flags.memory_use
        ]
        if phase == "ltm_grounded_mitigation_ready":
            accepted = [
                index
                for index in specialized
                if str(
                    (
                        candidates[index].compiled_packet.get("memory_usage")
                        or [{}]
                    )[0].get("usage", "")
                )
                == "support"
            ]
            specialized = accepted
            low_impact = [
                index
                for index in specialized
                if str(
                    (
                        candidates[index].compiled_packet.get("response") or {}
                    ).get("action", "Observe")
                )
                not in HIGH_IMPACT_ACTIONS
            ]
            specialized = low_impact
        if phase == "profile_grounded_mitigation_ready":
            profile_specialized = [
                index
                for index in specialized
                if str(
                    (
                        candidates[index].compiled_packet.get("memory_usage")
                        or [{}]
                    )[0].get("memory_id", "")
                ).startswith("profile:")
            ]
            specialized = profile_specialized
            low_cost_specialized = [
                index
                for index in specialized
                if str(
                    (
                        candidates[index].compiled_packet.get("response") or {}
                    ).get("action", "")
                )
                == "DeployDecoy"
            ]
            specialized = low_cost_specialized
        indices = specialized
    elif phase == "impact_probe_needed":
        specialized = [
            index
            for index in indices
            if str((candidates[index].compiled_packet.get("tool_call") or {}).get("name", ""))
            == "BusinessImpactEstimator"
            and not candidates[index].compiled_packet.get("trust_operations")
        ]
        indices = specialized
    if not indices:
        return None
    index = max(indices, key=lambda item: (q_values[item], candidates[item].semantic_id))
    return family, candidates[index].candidate_key


def _balanced_chain_rollout_option(
    observation: dict[str, Any],
    options: list[CandidateOption],
    q_values: dict[str, float],
    task_id: str,
) -> CandidateOption | None:
    """Choose a Teacher-ranked action inside a public, task-causal chain stage."""

    if not options:
        return None

    def best(predicate: Any) -> CandidateOption | None:
        matches = [row for row in options if predicate(row)]
        return (
            max(
                matches,
                key=lambda row: (
                    row.semantic_id in q_values,
                    q_values.get(row.semantic_id, float("-inf")),
                    row.semantic_id,
                ),
            )
            if matches
            else None
        )

    def tool(row: CandidateOption) -> str:
        return str((row.compiled_packet.get("tool_call") or {}).get("name", ""))

    def memory_op(row: CandidateOption) -> str:
        operations = row.compiled_packet.get("memory_operations") or []
        return str((operations[0] if operations else {}).get("op", ""))

    phase = _causal_phase(observation, task_id)
    probe_ids = (
        _fresh_recovery_verification_evidence_ids(observation)
        if phase == "ltm_recovery_grounding"
        else _verification_evidence_ids(observation)
        if phase == "ltm_evidence_grounding"
        else _probe_evidence_ids(observation)
    )
    memory_event_ids = _memory_event_ids(observation)
    choice: CandidateOption | None = None
    if phase == "low_cost_passive_anchor":
        choice = best(
            lambda row: row.action_family == "passive_verification"
            and tool(row) in {"ProvenanceCheck", "GraphQuery"}
            and not row.compiled_packet.get("trust_operations")
        )
    elif phase in {"probe_needed", "betrayal_probe_needed"}:
        choice = best(
            lambda row: row.action_family == "active_probe"
            and tool(row) in {"SourceChallenge", "CanaryProbe"}
        )
    elif phase in {
        "ltm_verification_needed",
        "ltm_recovery_verification_needed",
        "profile_verification_needed",
    }:
        choice = best(
            lambda row: row.action_family
            == (
                "passive_verification"
                if phase
                in {"ltm_verification_needed", "ltm_recovery_verification_needed"}
                else "active_probe"
            )
            and tool(row)
            in (
                {"CrossCheck"}
                if phase
                in {"ltm_verification_needed", "ltm_recovery_verification_needed"}
                else {"SourceChallenge", "CanaryProbe"}
            )
            and not row.compiled_packet.get("trust_operations")
            and (
                phase
                not in {"ltm_verification_needed", "ltm_recovery_verification_needed"}
                or (
                    (
                        phase == "ltm_recovery_verification_needed"
                        or _evidence_covers_memory_event(
                            observation,
                            set(
                                map(
                                    str,
                                    (
                                        (row.compiled_packet.get("tool_call") or {}).get(
                                            "args", {}
                                        )
                                        or {}
                                    ).get("evidence_ids", [])
                                    or [],
                                )
                            ),
                            memory_event_ids,
                        )
                    )
                    and _evidence_root_count(
                        observation,
                        set(
                            map(
                                str,
                                (
                                    (row.compiled_packet.get("tool_call") or {}).get(
                                        "args", {}
                                    )
                                    or {}
                                ).get("evidence_ids", [])
                                or [],
                            )
                        ),
                    )
                    >= 2
                    and (
                        phase != "ltm_recovery_verification_needed"
                        or str(
                            (
                                row.compiled_packet.get("tool_call") or {}
                            ).get("args", {}).get("event_id", "")
                        )
                        not in memory_event_ids
                    )
                )
            )
        )
    elif phase in {
        "probe_grounding",
        "trust_update_needed",
        "ltm_evidence_grounding",
        "ltm_recovery_grounding",
        "profile_update_needed",
    }:
        latest_recovery = _latest_recovery_verification(observation)
        required_op = (
            (
                "recover"
                if probe_evidence_trust_operation(latest_recovery) == "support"
                else probe_evidence_trust_operation(latest_recovery)
            )
            if phase == "ltm_recovery_grounding"
            and latest_recovery is not None
            else _required_probe_trust_operation(observation)
        )
        choice = best(
            lambda row: row.action_family == "trust"
            and bool(set(row.referenced_ids) & probe_ids)
            and any(
                str(operation.get("op", "")) == required_op
                for operation in row.compiled_packet.get("trust_operations", [])
                if isinstance(operation, dict)
            )
        )
    elif phase in {"ltm_risk_claim_wait", "ltm_retrieval_wait"}:
        choice = best(lambda row: row.action_flags.observe_only)
    elif phase == "ltm_ingest_needed":
        choice = best(
            lambda row: row.action_family == "memory"
            and memory_op(row) == "ingest"
            and str(
                ((row.compiled_packet.get("memory_operations") or [{}])[0].get("claim") or {}).get(
                    "object", ""
                )
            )
            in {
                "high",
                "active",
                "compromised",
                "credential_theft",
                "exfiltration",
                "lateral_movement",
                "persistence",
            }
        )
    elif phase in {"ltm_transition_needed", "ltm_recovery_promote"}:
        choice = best(
            lambda row: row.action_family == "memory"
            and memory_op(row)
            in (
                {"promote"}
                if phase == "ltm_recovery_promote"
                else {"promote", "reject"}
            )
        )
    elif phase == "ltm_rejected_or_recovery":
        choice = best(
            lambda row: row.action_family == "memory" and memory_op(row) == "reopen"
        )
    elif phase == "profile_history_accumulation":
        choice = best(lambda row: row.action_flags.observe_only)
    elif phase in {"impact_accumulation_budget_check", "post_mitigation_stop"}:
        choice = best(lambda row: row.action_flags.observe_only)
    elif phase == "impact_probe_needed":
        choice = best(
            lambda row: row.action_family == "passive_verification"
            and tool(row) == "BusinessImpactEstimator"
            and not row.compiled_packet.get("trust_operations")
        )
    elif phase in {
        "ltm_grounded_mitigation_ready",
        "profile_grounded_mitigation_ready",
    }:
        choice = best(
            lambda row: row.action_family == "mitigation"
            and bool(row.compiled_packet.get("memory_usage"))
            and (
                phase != "ltm_grounded_mitigation_ready"
                or str(
                    ((row.compiled_packet.get("memory_usage") or [{}])[0]).get(
                        "usage", ""
                    )
                )
                == "support"
            )
            and (
                phase != "ltm_grounded_mitigation_ready"
                or str(
                    (row.compiled_packet.get("response") or {}).get(
                        "action", "Observe"
                    )
                )
                not in HIGH_IMPACT_ACTIONS
            )
            and (
                phase != "profile_grounded_mitigation_ready"
                or str(
                    ((row.compiled_packet.get("memory_usage") or [{}])[0]).get(
                        "memory_id", ""
                    )
                ).startswith("profile:")
            )
            and (
                phase != "profile_grounded_mitigation_ready"
                or str((row.compiled_packet.get("response") or {}).get("action", ""))
                == "DeployDecoy"
            )
        )
    elif phase in {"mitigation_ready", "cost_grounded_mitigation_ready"}:
        choice = best(lambda row: row.action_family == "mitigation")
    return choice or best(lambda row: not row.action_flags.observe_only)


def _public_scenario_fingerprint(scenario: dict[str, Any]) -> str:
    """Hash public scenario semantics while excluding all hidden labels."""

    profiles = [
        {
            "source_id": str(profile.get("source_id", "")),
            "public_prior": float(profile.get("public_prior", 0.5)),
        }
        for profile in scenario.get("source_profiles", []) or []
        if isinstance(profile, dict)
    ]
    payload = {
        "protocol_version": scenario.get("protocol_version"),
        "schema_version": scenario.get("schema_version"),
        "scenario_family": scenario.get("scenario_family"),
        "distribution": scenario.get("distribution"),
        "network_context": scenario.get("network_context") or {},
        "source_profiles": profiles,
        "event_schedule": [
            project_event(event)
            for event in scenario.get("event_schedule", []) or []
            if isinstance(event, dict)
        ],
        "defense_constraints": scenario.get("defense_constraints") or {},
    }
    return semantic_digest(payload)


def _hidden_world_signature(scenario: dict[str, Any]) -> str:
    return semantic_digest(
        {
            "true_attack": scenario.get("true_attack") or {},
            "source_profiles": scenario.get("source_profiles") or [],
            "event_truth": [
                {
                    "event_id": row.get("event_id"),
                    "truth_value": row.get("truth_value"),
                    "is_fake": row.get("is_fake"),
                    "spoofability": row.get("spoofability"),
                }
                for row in scenario.get("event_schedule", []) or []
                if isinstance(row, dict)
            ],
        }
    )


def _expand_public_equivalent_worlds(
    groups: list[list[dict[str, Any]]], multiplicity: int
) -> list[list[dict[str, Any]]]:
    if multiplicity < 1:
        raise ValueError("robust world multiplicity must be positive")
    if multiplicity == 1:
        return groups
    expanded = []
    for group in groups:
        worlds = []
        for scenario in group:
            base_pressure = float(
                (scenario.get("true_attack") or {}).get("initial_pressure", 1.0)
            )
            for replica in range(multiplicity):
                clone = copy.deepcopy(scenario)
                if replica:
                    direction = -1.0 if base_pressure >= 0.95 else 1.0
                    pressure = max(
                        0.05,
                        min(1.0, base_pressure + direction * 0.04 * replica),
                    )
                    clone.setdefault("true_attack", {})["initial_pressure"] = pressure
                    # Keep the scenario ID stable: qualitative probe noise is
                    # keyed by it, while initial pressure remains hidden and
                    # creates a genuinely distinct terminal world.
                    metadata = dict(clone.get("metadata") or {})
                    metadata["public_equivalent_hidden_replica"] = replica
                    clone["metadata"] = metadata
                worlds.append(clone)
        expanded.append(worlds)
    return expanded


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _select_scored(
    options: list[CandidateOption],
    q_values: dict[str, float],
    *,
    selected_id: str,
    minimum: int,
    maximum: int,
    required_ids: tuple[str, ...] = (),
) -> list[CandidateOption]:
    scored = [row for row in options if row.semantic_id in q_values]
    by_family: dict[str, list[CandidateOption]] = defaultdict(list)
    for row in sorted(
        scored,
        key=lambda item: (-q_values[item.semantic_id], item.semantic_id),
    ):
        by_family[row.action_family].append(row)
    selected: list[CandidateOption] = []
    for family, quota in DEFAULT_QUOTAS.items():
        selected.extend(by_family[family][:quota])
    selected_ids = {row.semantic_id for row in selected}
    target = next((row for row in scored if row.semantic_id == selected_id), None)
    if target is not None and target.semantic_id not in selected_ids:
        selected.append(target)
        selected_ids.add(target.semantic_id)
    for required_id in required_ids:
        required = next(
            (row for row in scored if row.semantic_id == required_id), None
        )
        if required is not None and required.semantic_id not in selected_ids:
            selected.append(required)
            selected_ids.add(required.semantic_id)
    for row in sorted(
        scored,
        key=lambda item: (-q_values[item.semantic_id], item.semantic_id),
    ):
        if len(selected) >= maximum:
            break
        if row.semantic_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row.semantic_id)
    protected_ids = {selected_id, *required_ids}
    if len(selected) > maximum:
        protected = [
            row for row in selected if row.semantic_id in protected_ids
        ]
        retained_ids = {row.semantic_id for row in protected}
        selected = protected + [
            row
            for row in selected
            if row.semantic_id not in retained_ids
        ][: max(0, maximum - len(protected))]
    if len(selected) < minimum or selected_id not in {row.semantic_id for row in selected}:
        return []
    return selected


def _record_id(digest: str, candidate_id: str, index: int) -> str:
    return hashlib.sha256(f"{digest}:{candidate_id}:{index}".encode()).hexdigest()


def _hard_negatives(
    candidates: list[CandidateOption], policy_scores: list[float], target_index: int
) -> list[int]:
    target = candidates[target_index]
    target_score = policy_scores[target_index]
    selected: list[int] = []

    def add(index: int | None) -> None:
        if index is not None and index != target_index and index not in selected:
            selected.append(index)

    def structural_priority(index: int) -> tuple[int, int, int, float, str]:
        row = candidates[index]
        same_family = int(row.action_family == target.action_family)
        reference_overlap = int(bool(set(row.referenced_ids) & set(target.referenced_ids)))
        target_flags = target.action_flags.to_dict()
        row_flags = row.action_flags.to_dict()
        flag_distance = sum(target_flags[key] != row_flags[key] for key in target_flags)
        policy_gap = max(0.0, target_score - policy_scores[index])
        return (
            -same_family,
            -reference_overlap,
            flag_distance,
            policy_gap,
            row.candidate_id,
        )

    # Same-family and wrong-target near misses carry the useful decision signal.
    for index in sorted(
        (i for i in range(len(candidates)) if i != target_index),
        key=structural_priority,
    ):
        add(index)
        if len(selected) >= 3:
            break
    # Preserve Observe as an explicit collapse negative on actionable states.
    if not target.action_flags.observe_only:
        add(next((i for i, row in enumerate(candidates) if row.action_flags.observe_only), None))
    # Fill from the highest-scored incorrect actions, never from trivial tail Q.
    for index in sorted(
        (i for i in range(len(candidates)) if i != target_index),
        key=lambda i: (-policy_scores[i], candidates[i].candidate_id),
    ):
        add(index)
        if len(selected) >= 4:
            break
    return selected[:4]


def _negative_reasons(
    candidate: CandidateOption,
    target: CandidateOption,
    *,
    latest_evidence_ids: set[str],
) -> list[str]:
    reasons: list[str] = []
    candidate_packet = candidate.compiled_packet
    target_packet = target.compiled_packet
    if candidate.action_family == target.action_family:
        reasons.append("same_family_near_miss")
    if set(candidate.referenced_ids) != set(target.referenced_ids):
        reasons.append("wrong_or_stale_reference")
    if latest_evidence_ids and not (latest_evidence_ids & set(candidate.referenced_ids)):
        reasons.append("ignores_latest_evidence")
    candidate_tool = candidate_packet.get("tool_call") or {}
    target_tool = target_packet.get("tool_call") or {}
    if candidate_tool.get("name") == target_tool.get("name") and candidate_tool != target_tool:
        reasons.append("right_tool_wrong_target")
    candidate_response = candidate_packet.get("response") or {}
    target_response = target_packet.get("response") or {}
    if candidate_response.get("action") == target_response.get("action") and (
        candidate_response.get("target") != target_response.get("target")
    ):
        reasons.append("right_mitigation_wrong_asset")
    if candidate_response.get("target") == target_response.get("target") and (
        candidate_response.get("action") != target_response.get("action")
    ):
        reasons.append("wrong_response_severity")
    if target.action_flags.memory_use and not candidate.action_flags.memory_use:
        reasons.append("ignores_required_memory")
    if target.action_flags.trust and not candidate.action_flags.trust:
        reasons.append("ignores_current_claim_trust")
    return reasons or ["high_utility_incorrect"]


def _trajectory_negatives(
    candidates: list[CandidateOption],
    target_index: int,
    *,
    latest_evidence_ids: set[str],
    response_history_present: bool,
) -> list[int]:
    target = candidates[target_index]
    selected: list[int] = []
    for index, candidate in enumerate(candidates):
        if index == target_index or candidate.action_family != target.action_family:
            continue
        ignores_evidence = bool(
            latest_evidence_ids
            and latest_evidence_ids & set(target.referenced_ids)
            and not (latest_evidence_ids & set(candidate.referenced_ids))
        )
        ignores_memory = target.action_flags.memory_use and not candidate.action_flags.memory_use
        ignores_history = response_history_present and (
            candidate.compiled_packet.get("response")
            != target.compiled_packet.get("response")
        )
        if ignores_evidence or ignores_memory or ignores_history:
            selected.append(index)
    return selected[:4]


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    injected_groups = getattr(args, "scenario_groups", None)
    if injected_groups is not None:
        groups = copy.deepcopy(list(injected_groups))
        task_group_indices = [
            int(
                ((group[0].get("metadata") or {}).get("generator_group_index", index))
            )
            for index, group in enumerate(groups)
        ]
    elif args.scenario_source is not None:
        groups = counterfactual_groups(load_source_scenarios(args.scenario_source))
    elif args.task_schedule:
        task_schedule = [
            item.strip().upper()
            for item in str(args.task_schedule).split(",")
            if item.strip()
        ]
        invalid = sorted(set(task_schedule) - {"T1", "T2", "T3", "T4"})
        if not task_schedule or invalid:
            raise ValueError(f"invalid task schedule: {invalid or task_schedule}")
        groups = []
        task_group_indices = []
        cursor = int(args.group_offset)
        for task_id in task_schedule:
            while True:
                try:
                    group = canonical_recovery_group(task_id, cursor)
                except ValueError:
                    cursor += 1
                    continue
                groups.append(group)
                task_group_indices.append(cursor)
                cursor += 1
                break
    else:
        groups = canonical_recovery_suite(
            scenario_count=args.scenario_count,
            group_offset=args.group_offset,
        )
    groups = _expand_public_equivalent_worlds(
        groups, args.robust_world_multiplicity
    )
    if args.public_action_flip:
        for group in groups:
            for scenario in group:
                events = scenario.get("event_schedule", []) or []
                if not events:
                    continue
                index = min(1, len(events) - 1)
                semantics = events[index].get("claim_semantics", {}) or {}
                current = str(semantics.get("object", ""))
                replacement = "credential_theft" if current != "credential_theft" else "persistence"
                semantics["object"] = replacement
                events[index]["claim_semantics"] = semantics
                events[index]["objective_hint"] = replacement
                events[index]["claim"] = f"public counterfactual claim suggests {replacement}"
    teacher = PublicStateRobustTeacher(
        beam_width=args.teacher_beam_width,
        max_candidates=args.teacher_max_candidates,
    )
    generator = CandidateGenerator(
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
    )
    records: list[dict[str, Any]] = []
    skipped = Counter()
    family_counts = Counter()
    task_counts = Counter()
    decision_index = 0
    rng = random.Random(args.rollout_seed)
    effective_group_limit = int(args.max_records_per_group)
    if effective_group_limit <= 0 and args.task_schedule:
        effective_group_limit = max(1, math.ceil(args.max_records / len(groups)))
    for initial_group_index, scenarios in enumerate(groups):
        group_record_start = len(records)
        live = [instantiate_scenario(copy.deepcopy(row)) for row in scenarios]
        while (
            live
            and len(records) < args.max_records
            and (
                effective_group_limit <= 0
                or len(records) - group_record_start < effective_group_limit
            )
        ):
            public_groups: dict[str, list[Any]] = defaultdict(list)
            for env in live:
                public_groups[public_state_digest(env.observe())].append(env)
            next_live: list[Any] = []
            for digest, worlds in sorted(public_groups.items()):
                prior_step = worlds[0].history[-1] if worlds[0].history else {}
                prior_packet = prior_step.get("action_packet", {}) or {}
                prior_tool = str((prior_packet.get("tool_call") or {}).get("name", ""))
                post_probe_singleton = len(worlds) == 1 and prior_tool in ACTIVE_PROBE_TOOLS
                hidden_world_signature_count = len(
                    {_hidden_world_signature(env.scenario) for env in worlds}
                )
                if len(worlds) < 2 and not (
                    args.allow_post_probe_singletons and post_probe_singleton
                ):
                    skipped["singleton_public_state"] += 1
                    continue
                observation = project_public(worlds[0].observe())
                assert_public(observation)
                options = generator.generate_all(observation)
                task_id = str(
                    (scenarios[0].get("metadata") or {}).get(
                        "task_id", "unknown"
                    )
                )
                public_chain_target = _causal_target(
                    observation,
                    task_id,
                    options,
                    [0.0] * len(options),
                )
                public_chain_option = next(
                    (
                        row
                        for row in options
                        if public_chain_target is not None
                        and row.semantic_id == public_chain_target[1]
                    ),
                    None,
                )
                priority_candidates = (
                    (
                        ActionCandidate(
                            category=public_chain_option.action_family,
                            packet=copy.deepcopy(public_chain_option.compiled_packet),
                            label=f"public_chain:{public_chain_option.public_summary}",
                        ),
                    )
                    if public_chain_option is not None
                    else ()
                )
                decision = teacher.decide(
                    worlds,
                    horizon=3,
                    enforce_min_worlds=len(worlds) >= 2,
                    priority_candidates=priority_candidates,
                )
                supervision_target_id = decision.selected_candidate_id
                prior_probe_result = observation.get("last_tool_result") or {}
                raw_probe_evidence_id = str(
                    prior_probe_result.get("evidence_id", "")
                )
                historical_probe_evidence_ids = sorted(
                    {
                        str(evidence.get("evidence_id", ""))
                        for evidence in observation.get("available_evidence", [])
                        if isinstance(evidence, dict)
                        and str((evidence.get("content") or {}).get("tool", ""))
                        in ACTIVE_PROBE_TOOLS
                        and str(evidence.get("evidence_id", ""))
                    }
                )
                probe_evidence_ids = (
                    [raw_probe_evidence_id]
                    if prior_tool in ACTIVE_PROBE_TOOLS and raw_probe_evidence_id
                    else []
                )
                if prior_tool in ACTIVE_PROBE_TOOLS and not probe_evidence_ids:
                    probe_evidence_ids = [
                        evidence_id
                        for evidence_id in historical_probe_evidence_ids
                        if any(
                            str(evidence.get("evidence_id", "")) == evidence_id
                            and int(evidence.get("available_at", -1))
                            == int(observation.get("time", -2))
                            for evidence in observation.get(
                                "available_evidence", []
                            )
                            if isinstance(evidence, dict)
                        )
                    ]
                grounding_reference_required = prior_tool in {
                    "SourceChallenge",
                    "CanaryProbe",
                }
                grounded_scored = [
                    row
                    for row in options
                    if row.semantic_id in decision.q_audit
                    and bool(set(probe_evidence_ids) & set(row.referenced_ids))
                ]
                required_grounded_ids: tuple[str, ...] = ()
                if grounding_reference_required and grounded_scored:
                    best_grounded = max(
                        grounded_scored,
                        key=lambda row: (
                            decision.q_audit[row.semantic_id], row.semantic_id
                        ),
                    )
                    required_grounded_ids = (best_grounded.semantic_id,)
                admitted_options = [
                    row for row in options if row.semantic_id in decision.q_audit
                ]
                preselected_causal_target = _causal_target(
                    observation,
                    task_id,
                    admitted_options,
                    [
                        float(decision.q_audit[row.semantic_id])
                        for row in admitted_options
                    ],
                )
                preselected_causal_semantic_id = (
                    str(preselected_causal_target[1])
                    if preselected_causal_target is not None
                    else ""
                )
                preselected_chain_option = next(
                    (
                        row
                        for row in admitted_options
                        if row.semantic_id == preselected_causal_semantic_id
                    ),
                    None,
                )
                required_causal_ids = (
                    (preselected_causal_semantic_id,)
                    if preselected_causal_target is not None
                    else ()
                )
                selected = _select_scored(
                    options,
                    decision.q_audit,
                    selected_id=supervision_target_id,
                    minimum=args.min_candidates,
                    maximum=args.max_candidates,
                    required_ids=required_grounded_ids + required_causal_ids,
                )
                if not selected:
                    skipped["insufficient_scored_candidates"] += 1
                else:
                    selected = generator._remap_keys(
                        observation,
                        selected,
                        permutation_seed=args.permutation_seed + decision_index,
                    )
                    q_values = [float(decision.q_audit[row.semantic_id]) for row in selected]
                    core_q_values = [
                        float(decision.core_q_audit[row.semantic_id]) for row in selected
                    ]
                    best_index = next(
                        index
                        for index, row in enumerate(selected)
                        if row.semantic_id == supervision_target_id
                    )
                    observe_index = next(
                        index
                        for index, row in enumerate(selected)
                        if row.action_flags.observe_only
                    )
                    supervision = policy_consistent_supervision(
                        q_values,
                        core_q_values,
                        target_index=best_index,
                        observe_index=observe_index,
                        core_tolerance=teacher.core_tolerance,
                        temperature=args.teacher_temperature,
                    )
                    probabilities = list(supervision.probabilities)
                    policy_scores = list(supervision.policy_scores)
                    negative_indices = _hard_negatives(
                        selected, policy_scores, best_index
                    )
                    record_id = _record_id(
                            digest, supervision_target_id, decision_index
                    )
                    target_option = selected[best_index]
                    grounded_candidate_indices = [
                        index
                        for index, option in enumerate(selected)
                        if bool(
                            set(probe_evidence_ids) & set(option.referenced_ids)
                        )
                    ]
                    ungrounded_candidate_indices = [
                        index
                        for index in range(len(selected))
                        if index not in grounded_candidate_indices
                    ]
                    grounded_teacher_q_gap = None
                    grounded_teacher_policy_gap = None
                    if grounded_candidate_indices and ungrounded_candidate_indices:
                        grounded_teacher_q_gap = max(
                            q_values[index] for index in grounded_candidate_indices
                        ) - max(
                            q_values[index] for index in ungrounded_candidate_indices
                        )
                        grounded_teacher_policy_gap = max(
                            policy_scores[index]
                            for index in grounded_candidate_indices
                        ) - max(
                            policy_scores[index]
                            for index in ungrounded_candidate_indices
                        )
                    available_evidence = list(observation.get("available_evidence") or [])
                    latest_time = max(
                        (int(item.get("available_at", -1)) for item in available_evidence),
                        default=-1,
                    )
                    latest_evidence_ids = {
                        str(item.get("evidence_id", ""))
                        for item in available_evidence
                        if int(item.get("available_at", -1)) == latest_time
                        and str(item.get("evidence_id", ""))
                    }
                    trajectory_negative_indices = _trajectory_negatives(
                        selected,
                        best_index,
                        latest_evidence_ids=latest_evidence_ids,
                        response_history_present=bool(
                            ((observation.get("defender_state") or {}).get("response_history"))
                        ),
                    )
                    probe_chain = {
                        "is_probe_followup_state": prior_tool in ACTIVE_PROBE_TOOLS,
                        "probe_tool": prior_tool if prior_tool in ACTIVE_PROBE_TOOLS else None,
                        "probe_evidence_id": (
                            probe_evidence_ids[0] if probe_evidence_ids else None
                        ),
                        "probe_evidence_ids": probe_evidence_ids,
                        "historical_probe_evidence_ids": (
                            historical_probe_evidence_ids
                        ),
                        "raw_tool_evidence_id": raw_probe_evidence_id or None,
                        "grounding_reference_required": (
                            grounding_reference_required
                        ),
                        "grounded_candidate_count": len(
                            grounded_candidate_indices
                        ),
                        "grounded_teacher_q_gap": grounded_teacher_q_gap,
                        "grounded_teacher_policy_gap": (
                            grounded_teacher_policy_gap
                        ),
                        "evidence_used_by_target": bool(
                            set(probe_evidence_ids)
                            & set(target_option.referenced_ids)
                        ),
                        "followup_flags": target_option.action_flags.to_dict(),
                        "followup_nonobserve": not target_option.action_flags.observe_only,
                    }
                    causal_target_option = next(
                        (
                            row
                            for row in selected
                            if row.semantic_id == preselected_causal_semantic_id
                        ),
                        None,
                    )
                    causal_target = (
                        (
                            str(preselected_causal_target[0]),
                            causal_target_option.candidate_key,
                        )
                        if preselected_causal_target is not None
                        and causal_target_option is not None
                        else None
                    )
                    records.append(
                        {
                            "schema_version": 2,
                            "data_source": args.data_source,
                            "protocol_role": args.protocol_role,
                            "task_id": task_id,
                            "record_id": record_id,
                            "public_state_digest": digest,
                            "semantic_public_state_digest": semantic_digest(observation),
                            "semantic_scenario_fingerprint": (
                                _public_scenario_fingerprint(scenarios[0])
                            ),
                            "public_observation": observation,
                            "candidates": [
                                row.public_record(include_packet=True) for row in selected
                            ],
                            "teacher_probabilities": probabilities,
                            "teacher_policy_scores": policy_scores,
                            "teacher_eligible_mask": list(supervision.eligible),
                            "teacher_retained_mask": list(supervision.retained),
                            "teacher_acceptable_mask": list(supervision.acceptable),
                            "teacher_q_values": q_values,
                            "teacher_core_q_values": core_q_values,
                            "outcome_targets": [
                                dict(decision.outcome_audit[row.semantic_id])
                                for row in selected
                            ],
                            "teacher_full_q_best_candidate_id": max(
                                decision.q_audit,
                                key=lambda candidate_id: (
                                    decision.q_audit[candidate_id], candidate_id
                                ),
                            ),
                            "teacher_full_q_best_value": max(decision.q_audit.values()),
                            "target_candidate_key": selected[best_index].candidate_key,
                            "target_semantic_id": supervision_target_id,
                            "target_candidate_id": selected[best_index].candidate_key,
                            "target_family": selected[best_index].action_family,
                            "causal_target_family": (
                                causal_target[0] if causal_target is not None else None
                            ),
                            "causal_target_candidate_id": (
                                causal_target[1] if causal_target is not None else None
                            ),
                            "hard_negative_candidate_ids": [
                                selected[index].candidate_key for index in negative_indices
                            ],
                            "hard_negative_reasons": {
                                selected[index].candidate_key: _negative_reasons(
                                    selected[index],
                                    target_option,
                                    latest_evidence_ids=latest_evidence_ids,
                                )
                                for index in negative_indices
                            },
                            "trajectory_negative_candidate_ids": [
                                selected[index].candidate_key
                                for index in trajectory_negative_indices
                            ],
                            "trajectory_id": semantic_digest(scenarios[0]),
                            "parent_public_state_digest": getattr(
                                worlds[0],
                                "_candidate_dataset_parent_public_state_digest",
                                None,
                            ),
                            "trajectory_step": int(observation.get("time", 0)),
                            "causal_phase": _causal_phase(observation, task_id),
                            "probe_chain_target": probe_chain,
                            "scenario_credibility": copy.deepcopy(
                                (scenarios[0].get("metadata") or {}).get(
                                    "cyber_grounding"
                                )
                            ),
                            "audit": {
                                "world_count": len(worlds),
                                "hidden_world_signature_count": (
                                    hidden_world_signature_count
                                ),
                                "label_scope": (
                                    "post_probe_singleton"
                                    if post_probe_singleton
                                    else "robust_public_equivalence_class"
                                ),
                                "teacher_robust_value": decision.robust_value,
                                "teacher_observe_value": decision.observe_value,
                                "teacher_advantage": decision.advantage_over_observe,
                                "teacher_selected_core_value": core_q_values[best_index],
                                "teacher_original_selected_candidate_id": (
                                    decision.selected_candidate_id
                                ),
                                "supervision_target_override": None,
                                "teacher_observe_core_value": next(
                                    core_q_values[index]
                                    for index, row in enumerate(selected)
                                    if row.action_flags.observe_only
                                ),
                                "core_first_tolerance": teacher.core_tolerance,
                                "candidate_permutation_seed": args.permutation_seed
                                + decision_index,
                                "initial_group_index": initial_group_index,
                                "decision_index": decision_index,
                                "hidden_state_in_model_input": False,
                                "teacher_q_in_model_input": False,
                                "public_chain_priority_candidate_id": (
                                    public_chain_option.semantic_id
                                    if public_chain_option is not None
                                    else None
                                ),
                                "public_chain_priority_scored": bool(
                                    public_chain_option is not None
                                    and public_chain_option.semantic_id
                                    in decision.q_audit
                                ),
                            },
                        }
                    )
                    family_counts[selected[best_index].action_family] += 1
                    task_counts[task_id] += 1
                decision_index += 1
                rollout_packet = decision.selected_packet
                if args.trajectory_policy == "noop":
                    observe = next(
                        row for row in options if row.action_flags.observe_only
                    )
                    rollout_packet = observe.compiled_packet
                elif args.trajectory_policy in {"random", "scripted", "balanced_chain"}:
                    admitted = [
                        row for row in options if row.semantic_id in decision.q_audit
                    ]
                    if args.trajectory_policy == "balanced_chain":
                        chain_option = preselected_chain_option
                        if chain_option is not None:
                            admitted = [chain_option]
                    elif args.trajectory_policy == "scripted":
                        task_id = str(
                            (scenarios[0].get("metadata") or {}).get("task_id", "")
                        )
                        preferred = {
                            "T1": "active_probe",
                            "T2": "trust",
                            "T3": "memory",
                            "T4": "passive_verification",
                        }.get(task_id, "mitigation")
                        family_rows = [
                            row for row in admitted if row.action_family == preferred
                        ]
                        if family_rows:
                            admitted = family_rows
                    if admitted:
                        rollout_packet = copy.deepcopy(rng.choice(admitted).compiled_packet)
                post_mitigation_audit_steps = max(
                    0, int(getattr(args, "post_mitigation_audit_steps", 0) or 0)
                )
                for env in worlds:
                    pending_audit_steps = int(
                        getattr(
                            env,
                            "_candidate_dataset_post_mitigation_audit_steps",
                            0,
                        )
                        or 0
                    )
                    env.step(copy.deepcopy(rollout_packet))
                    env._candidate_dataset_parent_public_state_digest = digest
                    if pending_audit_steps > 0:
                        env._candidate_dataset_post_mitigation_audit_steps = (
                            pending_audit_steps - 1
                        )
                        continue
                    if (
                        post_mitigation_audit_steps > 0
                        and env.attack_mitigated
                        and not env.attack_success
                        and env.t < env.max_steps
                    ):
                        env._candidate_dataset_post_mitigation_audit_steps = (
                            post_mitigation_audit_steps
                        )
                        next_live.append(env)
                        continue
                    if not (
                        env.t >= env.max_steps
                        or env.attack_mitigated
                        or env.attack_success
                    ):
                        next_live.append(env)
                if len(records) >= args.max_records:
                    break
                if (
                    effective_group_limit > 0
                    and len(records) - group_record_start >= effective_group_limit
                ):
                    break
            live = next_live

    accepted = bool(records) and all(
        (
            int(row["audit"]["world_count"]) >= 2
            and int(row["audit"]["hidden_world_signature_count"]) >= 2
        )
        or (
            args.allow_post_probe_singletons
            and row["audit"].get("label_scope") == "post_probe_singleton"
        )
        for row in records
    )
    manifest = {
        "schema_version": 2,
        "kind": "candidate_listwise_teacher_dataset",
        "created_at": utc_now(),
        "accepted": accepted,
        "scenario_count": sum(len(group) for group in groups),
        "data_source": args.data_source,
        "protocol_role": args.protocol_role,
        "trajectory_policy": args.trajectory_policy,
        "public_action_flip": args.public_action_flip,
        "scenario_source": str(args.scenario_source.resolve()) if args.scenario_source else None,
        "scenario_source_sha256": sha256_file(args.scenario_source)
        if args.scenario_source
        else None,
        "task_schedule": (
            [item.strip().upper() for item in args.task_schedule.split(",") if item.strip()]
            if args.task_schedule
            else None
        ),
        "task_group_indices": (
            task_group_indices
            if args.task_schedule or injected_groups is not None
            else None
        ),
        "record_count": len(records),
        "candidate_count_min": min((len(row["candidates"]) for row in records), default=0),
        "candidate_count_max": max((len(row["candidates"]) for row in records), default=0),
        "target_family_counts": dict(sorted(family_counts.items())),
        "task_record_counts": dict(sorted(task_counts.items())),
        "causal_phase_counts": dict(
            sorted(Counter(str(row.get("causal_phase", "unknown")) for row in records).items())
        ),
        "causal_target_missing_count": sum(
            row.get("causal_target_family") is None for row in records
        ),
        "public_chain_priority_unscored_count": sum(
            not bool((row.get("audit") or {}).get("public_chain_priority_scored"))
            for row in records
        ),
        "skipped": dict(sorted(skipped.items())),
        "teacher_temperature": args.teacher_temperature,
        "candidate_keys_randomized": True,
        "probe_chain_record_count": sum(
            bool(row["probe_chain_target"]["is_probe_followup_state"])
            for row in records
        ),
        "post_probe_singleton_count": sum(
            row["audit"].get("label_scope") == "post_probe_singleton"
            for row in records
        ),
        "post_mitigation_audit_steps": max(
            0, int(getattr(args, "post_mitigation_audit_steps", 0) or 0)
        ),
        "post_mitigation_stop_record_count": sum(
            str(row.get("causal_phase", "")) == "post_mitigation_stop"
            for row in records
        ),
        "policy_target_argmax_rate": (
            sum(
                row["teacher_probabilities"].index(max(row["teacher_probabilities"]))
                == next(
                    index
                    for index, candidate in enumerate(row["candidates"])
                    if candidate["candidate_key"] == row["target_candidate_id"]
                )
                for row in records
            )
            / len(records)
            if records
            else 0.0
        ),
        "policy_ineligible_probability_mass": sum(
            sum(
                probability
                for probability, eligible in zip(
                    row["teacher_probabilities"],
                    row["teacher_eligible_mask"],
                    strict=True,
                )
                if not eligible
            )
            for row in records
        ),
        "teacher_target_recall": (
            sum(
                any(
                    candidate["candidate_key"] == row["target_candidate_id"]
                    for candidate in row["candidates"]
                )
                for row in records
            )
            / len(records)
            if records
            else 0.0
        ),
        "teacher_final_target_alignment_rate": (
            sum(
                str((row.get("audit") or {}).get("teacher_original_selected_candidate_id"))
                == str(row.get("target_semantic_id"))
                for row in records
            )
            / len(records)
            if records
            else 0.0
        ),
        "semantic_public_state_duplicate_count": len(records)
        - len({row["semantic_public_state_digest"] for row in records}),
        "semantic_scenario_count": len(
            {row["semantic_scenario_fingerprint"] for row in records}
        ),
        "hidden_state_in_model_input": False,
        "teacher_q_in_model_input": False,
        "formal_world_minimum": 2,
        "robust_world_multiplicity": args.robust_world_multiplicity,
        "minimum_hidden_world_signature_count": min(
            (
                int(row["audit"]["hidden_world_signature_count"])
                for row in records
            ),
            default=0,
        ),
        "model_input_fields": ["public_observation", "candidate public fields"],
        "offline_label_fields": [
            "teacher_probabilities",
            "teacher_policy_scores",
            "teacher_q_values",
            "teacher_core_q_values",
            "outcome_targets",
            "target_candidate_id",
        ],
    }
    return records, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-source", type=Path)
    parser.add_argument("--scenario-count", type=int, default=128)
    parser.add_argument("--group-offset", type=int, default=10000)
    parser.add_argument(
        "--task-schedule",
        help="Comma-separated canonical task schedule, for example T1,T2,T3,T4,T1.",
    )
    parser.add_argument("--max-records", type=int, default=512)
    parser.add_argument(
        "--max-records-per-group",
        type=int,
        default=0,
        help="Optional cap that prevents long task trajectories from crowding out T1-T4 coverage.",
    )
    parser.add_argument("--min-candidates", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-beam-width", type=int, default=20)
    parser.add_argument("--teacher-max-candidates", type=int, default=96)
    parser.add_argument(
        "--robust-world-multiplicity",
        type=int,
        default=1,
        help=(
            "Create distinct hidden-pressure replicas with identical public inputs; "
            "formal candidate labels use 2."
        ),
    )
    parser.add_argument("--permutation-seed", type=int, default=20260719)
    parser.add_argument(
        "--trajectory-policy",
        choices=["teacher", "random", "scripted", "balanced_chain", "noop"],
        default="teacher",
    )
    parser.add_argument("--data-source", default="teacher")
    parser.add_argument(
        "--protocol-role",
        choices=(
            "diagnostic",
            "learnability_256",
            "supervised_train",
            "supervised_dev",
            "frontier_train",
            "xplay_heldout",
            "ecrg_cal",
        ),
        default="diagnostic",
    )
    parser.add_argument("--public-action-flip", action="store_true")
    parser.add_argument(
        "--allow-post-probe-singletons",
        action="store_true",
        help=(
            "Diagnostic-only: retain post-probe states after public observations split "
            "the hidden-world equivalence class."
        ),
    )
    parser.add_argument("--rollout-seed", type=int, default=20260719)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    records, manifest = build(args)
    args.output_dir.mkdir(parents=True)
    data_path = args.output_dir / "candidate_sets.jsonl"
    _atomic_jsonl(data_path, records)
    manifest["candidate_sets_sha256"] = sha256_file(data_path)
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
