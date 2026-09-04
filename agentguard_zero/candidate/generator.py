"""Dynamic public-state candidate generation for candidate-level VDA policies."""

from __future__ import annotations

import copy
import hashlib
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from agentguard_zero.candidate.metrics import action_flags
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.defender_state.evidence_signals import evidence_signal
from agentguard_zero.recovery.public_teacher import (
    ActionCandidate,
    enumerate_public_candidates,
    probe_evidence_trust_operation,
    public_belief_variants,
    public_inventory,
    public_state_digest,
)
from agentguard_zero.runtime_policy import HIGH_IMPACT_ACTIONS, TOOL_COSTS
from agentguard_zero.schemas.action_schema import OBJECTIVE_KEYS
from agentguard_zero.schemas.action_schema_v4 import validate_action_packet_v4
from agentguard_zero.tools.business_impact import estimate_business_impact
from agentguard_zero.world.public_projector import assert_public


ACTION_FAMILIES = (
    "observe",
    "passive_verification",
    "active_probe",
    "trust",
    "memory",
    "mitigation",
)
DEFAULT_QUOTAS = {
    "observe": 2,
    "passive_verification": 2,
    "active_probe": 3,
    "trust": 2,
    "memory": 2,
    "mitigation": 4,
}

MITIGATING_ACTIONS = {
    "DeployDecoy",
    "LimitSession",
    "ShadowBlock",
    "Isolate",
    "Restore",
    "Remove",
}
MEMORY_VERIFICATION_TOOLS = {
    "CrossCheck",
    "ProvenanceCheck",
    "SourceChallenge",
    "CanaryProbe",
}


def candidate_response_cost(
    observation: Mapping[str, Any], packet: Mapping[str, Any]
) -> float:
    """Return the deterministic public business-cost estimate for a packet."""

    context = observation.get("defense_context") or {}
    response = packet.get("response") or {}
    target = str(response.get("target", ""))
    criticality = next(
        (
            float(row.get("criticality", 0.5))
            for row in context.get("public_assets", []) or []
            if isinstance(row, Mapping) and str(row.get("id", "")) == target
        ),
        0.5,
    )
    return float(
        estimate_business_impact(dict(response), criticality).get(
            "estimated_cost", 0.0
        )
    )


def _public_memory_records(observation: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    memory = ((observation.get("defender_state") or {}).get("memory") or {})
    return {
        str(row.get("memory_id", "")): row
        for bucket in (
            "retrieved_confirmed",
            "retrieved_quarantined",
            "rejected_warnings",
            "retrieved_profiles",
        )
        for row in memory.get(bucket, []) or []
        if isinstance(row, Mapping) and str(row.get("memory_id", ""))
    }


def _public_evidence_records(
    observation: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("evidence_id", "")): row
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, Mapping) and str(row.get("evidence_id", ""))
    }


def _independent_public_evidence_count(
    evidence_ids: Sequence[str], evidence: Mapping[str, Mapping[str, Any]]
) -> int:
    root_sets = []
    for evidence_id in set(map(str, evidence_ids)):
        row = evidence.get(evidence_id)
        if row is None:
            continue
        roots = frozenset(
            map(
                str,
                row.get("root_source_ids")
                or [row.get("source_id", "")],
            )
        ) - {""}
        if roots:
            root_sets.append(roots)
    best = 0

    def search(index: int, used: frozenset[str], count: int) -> None:
        nonlocal best
        if count + len(root_sets) - index <= best:
            return
        if index >= len(root_sets):
            best = max(best, count)
            return
        roots = root_sets[index]
        if roots.isdisjoint(used):
            search(index + 1, used | roots, count + 1)
        search(index + 1, used, count)

    search(0, frozenset(), 0)
    return best


def candidate_is_state_legal(
    observation: Mapping[str, Any], packet: Mapping[str, Any]
) -> bool:
    """Reject packets that the public pre-state proves cannot commit.

    This is deliberately narrower than outcome scoring: it removes impossible
    budget, trust, and memory operations without consulting hidden truth.
    """

    context = observation.get("defense_context") or {}
    defender = observation.get("defender_state") or {}
    evidence = _public_evidence_records(observation)
    memory_records = _public_memory_records(observation)
    retrieved_ids = set(memory_records)
    tool = packet.get("tool_call") or {}
    tool_name = str(tool.get("name", "None"))
    if tool_name != "None" and float(TOOL_COSTS.get(tool_name, 1.0)) > float(
        context.get("remaining_verification_budget", float("inf"))
    ) + 1.0e-9:
        return False

    response = packet.get("response") or {}
    action = str(response.get("action", "Observe"))
    response_cost = candidate_response_cost(observation, packet)
    if response_cost > float(
        context.get("remaining_business_budget", float("inf"))
    ) + 1.0e-9:
        return False
    if action in HIGH_IMPACT_ACTIONS and int(
        context.get("remaining_high_impact_actions", 1)
    ) <= 0:
        return False
    if action in MITIGATING_ACTIONS and any(
        isinstance(row, Mapping)
        and bool(row.get("authorized", False))
        and str(row.get("action", "")) in MITIGATING_ACTIONS
        for row in defender.get("response_history", []) or []
    ):
        return False

    for operation in packet.get("trust_operations", []) or []:
        if not isinstance(operation, Mapping):
            return False
        op = str(operation.get("op", ""))
        refs = list(map(str, operation.get("evidence_refs", []) or []))
        rows = [evidence[ref] for ref in refs if ref in evidence]
        if len(rows) != len(refs):
            return False
        positive = sum(evidence_signal(dict(row))[0] for row in rows)
        negative = sum(evidence_signal(dict(row))[1] for row in rows)
        if op == "support" and not (rows and positive > negative):
            return False
        if op == "contradict" and not (rows and negative > positive):
            return False
        if op == "recover" and not (
            rows
            and positive > negative
            and _independent_public_evidence_count(refs, evidence) >= 2
        ):
            return False
        if op in {"support", "contradict", "recover"}:
            event_id = str(operation.get("event_id", ""))
            claim = (
                ((defender.get("trust") or {}).get("current_claim_trust") or {}).get(
                    event_id, {}
                )
                or {}
            )
            if refs and set(refs).issubset(
                set(map(str, claim.get("evidence_refs", []) or []))
            ):
                return False

    for operation in packet.get("memory_operations", []) or []:
        if not isinstance(operation, Mapping):
            return False
        op = str(operation.get("op", ""))
        refs = list(map(str, operation.get("evidence_refs", []) or []))
        if any(ref not in evidence for ref in refs):
            return False
        if op == "ingest":
            if str(operation.get("target_status", "quarantined")) != "quarantined":
                return False
            matching = next(
                (
                    row
                    for row in memory_records.values()
                    if (row.get("claim") or {}) == (operation.get("claim") or {})
                    and not str(row.get("memory_id", "")).startswith("profile:")
                ),
                None,
            )
            if matching is not None and not (
                set(refs) - set(map(str, matching.get("evidence_refs", []) or []))
            ):
                return False
            continue
        memory_id = str(operation.get("memory_id", ""))
        record = memory_records.get(memory_id)
        if record is None:
            return False
        current = str(record.get("status", "quarantined"))
        allowed_from = {
            "promote": {"quarantined"},
            "demote": {"confirmed"},
            "reject": {"quarantined", "confirmed"},
            "reopen": {"rejected"},
        }
        if current not in allowed_from.get(op, set()):
            return False
        claim_status = str(
            (
                ((defender.get("trust") or {}).get("current_claim_trust") or {}).get(
                    str(operation.get("event_id", "")), {}
                )
                or {}
            ).get("status", "unassessed")
        )
        if op == "promote" and not (
            claim_status == "supported"
            and _independent_public_evidence_count(refs, evidence) >= 2
        ):
            return False
        if op == "demote" and claim_status not in {"challenged", "contradicted"}:
            return False
        if op == "reject" and claim_status != "contradicted":
            return False
        if op == "reopen" and not (
            claim_status in {"challenged", "supported"}
            and _independent_public_evidence_count(refs, evidence) >= 2
        ):
            return False

    for usage in packet.get("memory_usage", []) or []:
        if not isinstance(usage, Mapping):
            return False
        memory_id = str(usage.get("memory_id", ""))
        record = memory_records.get(memory_id)
        if memory_id not in retrieved_ids or record is None:
            return False
        # Quarantined/rejected memory may be read by a verification tool, but
        # it can never authorize a response.  This public-state rule closes the
        # T3 retrieve -> accepted-use -> mitigation contract without consulting
        # hidden attack truth.
        response_requires_confirmed_evidence_memory = (
            str(usage.get("used_for", "belief")) == "response"
            or action in MITIGATING_ACTIONS
        ) and str(record.get("memory_kind", "evidence")) != "source_profile"
        if (
            response_requires_confirmed_evidence_memory
            and str(record.get("status", "quarantined")) != "confirmed"
        ):
            return False
    return True


# Existing imports used the private spelling before the public execution
# contract became part of inference-time candidate masking.
_candidate_is_state_legal = candidate_is_state_legal


def public_response_anchors(
    observation: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    """Extract response targets and objectives explicitly present in public state."""

    targets: set[str] = set()
    objectives: set[str] = set()
    rows = list(observation.get("observed_events", []) or [])
    rows.extend(observation.get("available_evidence", []) or [])
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        content = row.get("content") or row
        if not isinstance(content, Mapping):
            continue
        semantics = content.get("claim_semantics") or {}
        if not isinstance(semantics, Mapping):
            semantics = {}
        for target in (content.get("entity_id"), semantics.get("entity_id")):
            if str(target or "").strip():
                targets.add(str(target))
        for objective in (content.get("objective_hint"), semantics.get("object")):
            if str(objective) in OBJECTIVE_KEYS:
                objectives.add(str(objective))
    return targets, objectives


def public_mitigation_count(observation: Mapping[str, Any]) -> int:
    history = (
        (observation.get("defender_state") or {}).get("response_history") or []
    )
    return sum(
        1
        for row in history
        if isinstance(row, Mapping)
        and bool(row.get("authorized", False))
        and str(row.get("action", ""))
        in {"DeployDecoy", "LimitSession", "ShadowBlock", "Isolate"}
    )


def _references(packet: Mapping[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()
    response = packet.get("response") or {}
    if isinstance(response, Mapping) and str(response.get("target", "none")) != "none":
        values.add(str(response["target"]))
    tool = packet.get("tool_call") or {}
    args = tool.get("args") or {} if isinstance(tool, Mapping) else {}
    if isinstance(args, Mapping):
        for key in ("event_id", "source", "node", "zone"):
            if str(args.get(key, "")).strip():
                values.add(str(args[key]))
        values.update(map(str, args.get("evidence_ids", []) or []))
        nested = args.get("action") or {}
        if isinstance(nested, Mapping) and str(nested.get("target", "")).strip():
            values.add(str(nested["target"]))
    for field in ("trust_operations", "memory_operations"):
        for operation in packet.get(field, []) or []:
            if not isinstance(operation, Mapping):
                continue
            for key in ("source_id", "event_id", "memory_id"):
                if str(operation.get(key, "")).strip():
                    values.add(str(operation[key]))
            values.update(map(str, operation.get("evidence_refs", []) or []))
    for usage in packet.get("memory_usage", []) or []:
        if isinstance(usage, Mapping) and str(usage.get("memory_id", "")).strip():
            values.add(str(usage["memory_id"]))
    return tuple(sorted(values))


def _summary(candidate: ActionCandidate, packet: Mapping[str, Any]) -> str:
    flags = action_flags(packet)
    parts = [candidate.label]
    if flags.memory_use:
        usage = (packet.get("memory_usage") or [{}])[0]
        parts.append(f"memory_use:{usage.get('usage')}:{usage.get('used_for')}")
    response = packet.get("response") or {}
    if flags.mitigation:
        parts.append(f"response:{response.get('action')}:{response.get('target')}")
    return " | ".join(map(str, parts))


class CandidateGenerator:
    def __init__(
        self,
        *,
        min_candidates: int = 8,
        max_candidates: int = 24,
        quotas: Mapping[str, int] | None = None,
        experiment_variant: str | None = None,
    ) -> None:
        if min_candidates < 1 or max_candidates < min_candidates:
            raise ValueError("invalid candidate count bounds")
        self.min_candidates = int(min_candidates)
        self.max_candidates = int(max_candidates)
        self.quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
        self.experiment_variant = experiment_variant

    def _variants(
        self, observation: Mapping[str, Any], base: Sequence[ActionCandidate]
    ) -> list[tuple[ActionCandidate, dict[str, Any], dict[str, Any]]]:
        beliefs = public_belief_variants(observation)
        rows: list[tuple[ActionCandidate, dict[str, Any], dict[str, Any]]] = []
        for candidate in base:
            packet = copy.deepcopy(candidate.packet)
            rows.append((candidate, packet, {"belief_variant": "teacher_public"}))
            if candidate.category == "mitigation":
                for belief_name, belief in beliefs.items():
                    variant = copy.deepcopy(packet)
                    variant["belief"] = copy.deepcopy(belief)
                    variant["uncertainty"] = 1.0 - max(belief.values())
                    rows.append((candidate, variant, {"belief_variant": belief_name}))

        inventory = public_inventory(observation)
        actionable_by_family: dict[str, list[tuple[ActionCandidate, dict[str, Any]]]] = defaultdict(list)
        for candidate, packet, _ in rows:
            if candidate.category in {
                "active_probe",
                "passive_verification",
                "mitigation",
            }:
                actionable_by_family[candidate.category].append((candidate, packet))
        public_targets, public_objectives = public_response_anchors(observation)

        def mitigation_priority(
            row: tuple[ActionCandidate, dict[str, Any]],
        ) -> tuple[int, int, int, str]:
            _, packet = row
            response = packet.get("response") or {}
            action_order = {"ShadowBlock": 0, "LimitSession": 1, "DeployDecoy": 2}
            belief = packet.get("belief") or {}
            top_belief = max(belief, key=belief.get) if belief else ""
            return (
                0 if str(response.get("target", "")) in public_targets else 1,
                action_order.get(str(response.get("action", "")), 3),
                0 if not public_objectives or top_belief in public_objectives else 1,
                CandidateOption.packet_digest(packet),
            )

        mitigation_rows = sorted(
            actionable_by_family["mitigation"], key=mitigation_priority
        )
        actionable = (
            actionable_by_family["passive_verification"][:2]
            + actionable_by_family["active_probe"][:2]
            + mitigation_rows[:6]
        )
        for memory_index, memory_id in enumerate(inventory.memory_ids):
            status = "unknown"
            memory_state = (
                (observation.get("defender_state") or {}).get("memory") or {}
            )
            for bucket, candidate_status in (
                ("retrieved_confirmed", "confirmed"),
                ("retrieved_quarantined", "quarantined"),
                ("rejected_warnings", "rejected"),
                ("retrieved_profiles", "profile"),
            ):
                matching = next(
                    (
                        row
                        for row in memory_state.get(bucket, []) or []
                        if isinstance(row, Mapping)
                        and str(row.get("memory_id", "")) == memory_id
                    ),
                    None,
                )
                if matching is not None:
                    status = (
                        str(matching.get("status", "quarantined"))
                        if candidate_status == "profile"
                        else candidate_status
                    )
                    break
            usage_role = "support" if status == "confirmed" else "contradict"
            for candidate, source_packet in actionable:
                packet = copy.deepcopy(source_packet)
                used_for = (
                    "response"
                    if action_flags(packet).mitigation
                    else "tool"
                    if (packet.get("tool_call") or {}).get("name", "None") != "None"
                    else "belief"
                )
                packet["memory_usage"] = [
                    {
                        "memory_id": memory_id,
                        "usage": usage_role,
                        "used_for": used_for,
                    }
                ]
                rows.append(
                    (
                        candidate,
                        packet,
                        {
                            "belief_variant": "public_posterior",
                            "memory_status": status,
                            "combined_memory_use": True,
                            "memory_index": memory_index,
                        },
                    )
                )
        trust_rows = [
            packet
            for candidate, packet, _ in rows
            if candidate.category == "trust" and packet.get("trust_operations")
        ]
        active_rows = [
            (candidate, packet)
            for candidate, packet, _ in rows
            if candidate.category == "active_probe"
        ]
        for trust_packet in trust_rows[:2]:
            trust_operation = copy.deepcopy(trust_packet["trust_operations"][0])
            for candidate, probe_packet in active_rows[:3]:
                packet = copy.deepcopy(probe_packet)
                packet["trust_operations"] = [trust_operation]
                if trust_packet.get("evidence_assessment"):
                    packet["evidence_assessment"] = copy.deepcopy(
                        trust_packet["evidence_assessment"]
                    )
                rows.append(
                    (
                        candidate,
                        packet,
                        {"combined_trust_probe": True},
                    )
                )

        probe_evidence = [
            evidence
            for evidence in observation.get("available_evidence", []) or []
            if isinstance(evidence, Mapping)
            and str((evidence.get("content") or {}).get("tool", ""))
            in MEMORY_VERIFICATION_TOOLS
            and str(evidence.get("evidence_id", ""))
            and str(evidence.get("event_id", ""))
            and list(evidence.get("root_source_ids", []) or [])
        ]
        if probe_evidence:
            public_evidence = _public_evidence_records(observation)
            rejected_memory_present = bool(
                (
                    ((observation.get("defender_state") or {}).get("memory") or {}).get(
                        "rejected_warnings", []
                    )
                    or []
                )
            )
            for candidate, packet, metadata in list(rows):
                if candidate.category != "trust" or not packet.get("trust_operations"):
                    continue
                for evidence in probe_evidence:
                    evidence_id = str(evidence["evidence_id"])
                    event_id = str(evidence["event_id"])
                    source_id = str(
                        (
                            (
                                (
                                    (observation.get("defender_state") or {}).get(
                                        "trust"
                                    )
                                    or {}
                                ).get("current_claim_trust", {})
                                or {}
                            ).get(event_id, {})
                            or {}
                        ).get("source_id", "")
                    )
                    if not source_id:
                        source_id = next(
                            (
                                str(row.get("source_id") or row.get("source") or "")
                                for row in observation.get("observed_events", []) or []
                                if isinstance(row, Mapping)
                                and str(row.get("event_id", "")) == event_id
                            ),
                            "",
                        )
                    operation = probe_evidence_trust_operation(evidence)
                    evidence_refs = [evidence_id]
                    if operation == "support" and rejected_memory_present:
                        operation = "recover"
                        for parent_id in map(
                            str, evidence.get("parent_evidence_ids", []) or []
                        ):
                            parent = public_evidence.get(parent_id) or {}
                            if str((parent.get("content") or {}).get("tool", "")) in MEMORY_VERIFICATION_TOOLS and probe_evidence_trust_operation(parent) != "support":
                                continue
                            evidence_refs.append(parent_id)
                        evidence_refs = list(dict.fromkeys(evidence_refs))[:6]
                    assessment_status = {
                        "support": "supported",
                        "contradict": "contradicted",
                    }.get(operation, "challenged")
                    grounded = copy.deepcopy(packet)
                    grounded["trust_operations"] = [
                        {
                            "op": operation,
                            "source_id": source_id,
                            "event_id": event_id,
                            "evidence_refs": evidence_refs,
                        }
                    ]
                    grounded["evidence_assessment"] = [
                        {
                            "event_id": event_id,
                            "status": assessment_status,
                            "suspected_poisoning": assessment_status
                            == "contradicted",
                        }
                    ]
                    rows.append(
                        (
                            candidate,
                            grounded,
                            metadata
                            | {
                                "probe_evidence_grounded": True,
                                "probe_evidence_ids": [evidence_id],
                            },
                        )
                    )
        return rows

    def generate_all(self, observation: Mapping[str, Any]) -> list[CandidateOption]:
        assert_public(dict(observation))
        remaining_probe_budget = (observation.get("defense_context") or {}).get(
            "remaining_active_probe_budget"
        )
        active_probe_available = (
            remaining_probe_budget is None or float(remaining_probe_budget) > 0.0
        )
        inventory = public_inventory(observation)
        allowed_references = set(
            inventory.event_ids
            + tuple(inventory.evidence_by_event)
            + inventory.evidence_ids
            + inventory.source_ids
            + inventory.asset_ids
            + inventory.zones
            + inventory.memory_ids
        )
        base = enumerate_public_candidates(observation, max_candidates=96)
        admitted: dict[str, CandidateOption] = {}
        for candidate, packet, metadata in self._variants(observation, base):
            flags = action_flags(packet)
            if self.experiment_variant is not None:
                from agentguard_zero.progressive_ablation import candidate_allowed

                provisional = CandidateOption.from_packet(
                    action_family=candidate.category,
                    public_summary=_summary(candidate, packet),
                    referenced_ids=_references(packet),
                    compiled_packet=packet,
                    action_flags=flags,
                    audit_metadata=metadata,
                )
                if not candidate_allowed(provisional, self.experiment_variant):
                    continue
            if flags.active_probe and not active_probe_available:
                continue
            valid, _ = validate_action_packet_v4(packet)
            references = _references(packet)
            if (
                not valid
                or not set(references).issubset(allowed_references)
                or not candidate_is_state_legal(observation, packet)
            ):
                continue
            option = CandidateOption.from_packet(
                action_family=candidate.category,
                public_summary=_summary(candidate, packet),
                referenced_ids=references,
                compiled_packet=packet,
                action_flags=flags,
                audit_metadata=metadata | {"teacher_label": candidate.label},
            )
            admitted.setdefault(option.semantic_id, option)

        return sorted(
            admitted.values(),
            key=lambda row: (row.action_family, row.public_summary, row.semantic_id),
        )

    @staticmethod
    def _remap_keys(
        observation: Mapping[str, Any],
        candidates: Sequence[CandidateOption],
        *,
        permutation_seed: int,
    ) -> list[CandidateOption]:
        digest = public_state_digest(observation)
        seed_wire = f"{permutation_seed}:{digest}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_wire).digest()[:8], "big")
        rng = random.Random(seed)
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        remapped: list[CandidateOption] = []
        for index, option in enumerate(shuffled):
            key_wire = f"{seed}:{index}:{option.semantic_id}".encode("utf-8")
            key = "k_" + hashlib.sha256(key_wire).hexdigest()[:12]
            remapped.append(option.with_candidate_key(key))
        return remapped

    def generate(
        self,
        observation: Mapping[str, Any],
        *,
        permutation_seed: int = 0,
    ) -> list[CandidateOption]:
        admitted = self.generate_all(observation)
        requirements = (
            (observation.get("defense_context") or {}).get("response_requirements")
            or {}
        )
        public_evidence = _public_evidence_records(observation)
        verification_evidence_ids = {
            evidence_id
            for evidence_id, row in public_evidence.items()
            if str((row.get("content") or {}).get("tool", ""))
            in MEMORY_VERIFICATION_TOOLS
        }

        def selection_priority(row: CandidateOption) -> tuple[Any, ...]:
            packet = row.compiled_packet
            tool = packet.get("tool_call") or {}
            tool_name = str(tool.get("name", ""))
            operations = packet.get("memory_operations") or []
            memory_op = str((operations[0] if operations else {}).get("op", ""))
            evidence_ids = list(
                map(str, ((tool.get("args") or {}).get("evidence_ids", []) or []))
            )
            prerequisite = 1
            if row.action_family == "memory" and memory_op in {
                "promote",
                "demote",
                "reject",
                "reopen",
            }:
                prerequisite = 0
            elif (
                row.action_family == "passive_verification"
                and requirements.get("memory_use_required_for_mitigation")
                and tool_name == "CrossCheck"
                and _independent_public_evidence_count(evidence_ids, public_evidence)
                >= 2
            ):
                prerequisite = 0
            elif (
                row.action_family == "passive_verification"
                and requirements.get("impact_probe_required_for_mitigation")
                and tool_name == "BusinessImpactEstimator"
            ):
                prerequisite = 0
            elif (
                row.action_family == "active_probe"
                and requirements.get("profile_memory_use_required_for_mitigation")
                and tool_name in {"SourceChallenge", "CanaryProbe"}
            ):
                prerequisite = 0
            elif (
                row.action_family == "trust"
                and requirements.get("memory_use_required_for_mitigation")
                and verification_evidence_ids.intersection(row.referenced_ids)
            ):
                prerequisite = 0
            return (prerequisite, row.public_summary, row.semantic_id)

        by_family: dict[str, list[CandidateOption]] = defaultdict(list)
        for option in sorted(
            admitted, key=lambda row: (row.action_family, selection_priority(row))
        ):
            by_family[option.action_family].append(option)
        selected: list[CandidateOption] = []
        for family in ACTION_FAMILIES:
            selected.extend(by_family[family][: max(0, int(self.quotas.get(family, 0)))])
        selected_ids = {row.semantic_id for row in selected}
        protected_ids: set[str] = set()

        def ensure(predicate: Any) -> None:
            option = next(
                (
                    row
                    for row in admitted
                    if predicate(row)
                ),
                None,
            )
            if option is not None:
                protected_ids.add(option.semantic_id)
                if option.semantic_id not in selected_ids:
                    selected.append(option)
                    selected_ids.add(option.semantic_id)

        public_targets, public_objectives = public_response_anchors(observation)
        if requirements.get("impact_probe_required_for_mitigation"):
            ensure(
                lambda row: (
                    str((row.compiled_packet.get("tool_call") or {}).get("name", ""))
                    == "ShadowActionProbe"
                    and str(
                        (
                            (row.compiled_packet.get("tool_call") or {}).get("args", {})
                            or {}
                        ).get("action", {}).get("action", "")
                    )
                    == "ShadowBlock"
                    and (
                        not public_targets
                        or str(
                            (
                                (row.compiled_packet.get("tool_call") or {}).get(
                                    "args", {}
                                )
                                or {}
                            ).get("action", {}).get("target", "")
                        )
                        in public_targets
                    )
                )
            )
        if requirements.get("active_probe_required_for_mitigation"):
            ensure(
                lambda row: str(
                    (row.compiled_packet.get("tool_call") or {}).get("name", "")
                )
                == "SourceChallenge"
            )
        if requirements.get("trust_update_required_for_mitigation"):
            ensure(
                lambda row: row.action_flags.active_probe
                and row.action_flags.trust
                and any(
                    str(operation.get("op", "")) == "challenge"
                    for operation in row.compiled_packet.get("trust_operations", [])
                    if isinstance(operation, Mapping)
                )
            )
        if requirements.get("memory_use_required_for_mitigation"):
            ensure(
                lambda row: (
                    str((row.compiled_packet.get("tool_call") or {}).get("name", ""))
                    == "CrossCheck"
                    and _independent_public_evidence_count(
                        list(
                            map(
                                str,
                                (
                                    (
                                        (row.compiled_packet.get("tool_call") or {}).get(
                                            "args", {}
                                        )
                                        or {}
                                    ).get("evidence_ids", [])
                                    or []
                                ),
                            )
                        ),
                        public_evidence,
                    )
                    >= 2
                )
            )
            ensure(
                lambda row: row.action_flags.mitigation
                and row.action_flags.memory_use
            )
        preferred_response = (
            "DeployDecoy"
            if requirements.get("memory_use_required_for_mitigation")
            and public_mitigation_count(observation) >= 2
            else "ShadowBlock"
        )
        ensure(
            lambda row: (
                row.action_flags.mitigation
                and str((row.compiled_packet.get("response") or {}).get("action", ""))
                == preferred_response
                and (
                    not public_targets
                    or str((row.compiled_packet.get("response") or {}).get("target", ""))
                    in public_targets
                )
                and (
                    not public_objectives
                    or max(
                        row.compiled_packet.get("belief") or {"": 0.0},
                        key=(row.compiled_packet.get("belief") or {"": 0.0}).get,
                    )
                    in public_objectives
                )
                and (
                    not requirements.get("memory_use_required_for_mitigation")
                    or row.action_flags.memory_use
                )
            )
        )
        rejected_memory_present = bool(
            (
                ((observation.get("defender_state") or {}).get("memory") or {}).get(
                    "rejected_warnings", []
                )
                or []
            )
        )
        probe_evidence_operations = {
            str(row.get("evidence_id", "")): (
                "recover"
                if rejected_memory_present
                and probe_evidence_trust_operation(row) == "support"
                else probe_evidence_trust_operation(row)
            )
            for row in observation.get("available_evidence", []) or []
            if isinstance(row, Mapping)
            and str((row.get("content") or {}).get("tool", ""))
            in MEMORY_VERIFICATION_TOOLS | {"DecoyProbe", "ShadowActionProbe"}
            and str(row.get("evidence_id", ""))
        }
        if probe_evidence_operations:
            ensure(
                lambda row: row.action_family == "trust"
                and row.action_flags.trust
                and any(
                    str(operation.get("op", ""))
                    == probe_evidence_operations.get(str(reference), "")
                    for operation in row.compiled_packet.get("trust_operations", [])
                    if isinstance(operation, Mapping)
                    for reference in operation.get("evidence_refs", []) or []
                )
            )
        if rejected_memory_present and "recover" in probe_evidence_operations.values():
            ensure(
                lambda row: row.action_family == "trust"
                and any(
                    str(operation.get("op", "")) == "recover"
                    and any(
                        probe_evidence_operations.get(str(reference)) == "recover"
                        for reference in operation.get("evidence_refs", []) or []
                    )
                    for operation in row.compiled_packet.get("trust_operations", [])
                    if isinstance(operation, Mapping)
                )
            )
        for option in sorted(admitted, key=lambda row: row.semantic_id):
            if len(selected) >= self.max_candidates:
                break
            if option.semantic_id not in selected_ids:
                selected.append(option)
                selected_ids.add(option.semantic_id)
        selected = (
            [row for row in selected if row.semantic_id in protected_ids]
            + [row for row in selected if row.semantic_id not in protected_ids]
        )[: self.max_candidates]
        if len(selected) < self.min_candidates:
            raise RuntimeError(
                f"public state produced {len(selected)} candidates; minimum is {self.min_candidates}"
            )
        return self._remap_keys(
            observation,
            selected,
            permutation_seed=int(permutation_seed),
        )
