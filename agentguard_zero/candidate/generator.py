"""Dynamic public-state candidate generation for candidate-level VDA policies."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from agentguard_zero.candidate.metrics import action_flags
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.recovery.public_teacher import (
    ActionCandidate,
    enumerate_public_candidates,
    public_inventory,
    public_state_digest,
)
from agentguard_zero.schemas.action_schema import OBJECTIVE_KEYS
from agentguard_zero.schemas.action_schema_v4 import validate_action_packet_v4
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
    "observe": 1,
    "passive_verification": 2,
    "active_probe": 3,
    "trust": 2,
    "memory": 2,
    "mitigation": 4,
}


def _normalized(values: Mapping[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(values.get(key, 0.0))) for key in OBJECTIVE_KEYS}
    total = sum(cleaned.values())
    if total <= 0.0:
        return {key: 1.0 / len(OBJECTIVE_KEYS) for key in OBJECTIVE_KEYS}
    return {key: value / total for key, value in cleaned.items()}


def public_belief_variants(observation: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    support = {key: 1.0 for key in OBJECTIVE_KEYS}
    for event in observation.get("observed_events", []) or []:
        if not isinstance(event, Mapping):
            continue
        objective = str(event.get("objective_hint", ""))
        semantics = event.get("claim_semantics") or {}
        semantic_object = str(semantics.get("object", "")) if isinstance(semantics, Mapping) else ""
        for value in (objective, semantic_object):
            if value in support:
                support[value] += 1.0
    for evidence in observation.get("available_evidence", []) or []:
        if not isinstance(evidence, Mapping):
            continue
        content = evidence.get("content") or {}
        if not isinstance(content, Mapping):
            continue
        semantics = content.get("claim_semantics") or {}
        objective = str(semantics.get("object", "")) if isinstance(semantics, Mapping) else ""
        if objective in support:
            signals = json.dumps(content, ensure_ascii=True).lower()
            support[objective] += -1.0 if any(
                word in signals for word in ("conflict", "failed", "inconsistent")
            ) else 0.5
    posterior = _normalized(support)
    ranked = sorted(OBJECTIVE_KEYS, key=lambda key: (-posterior[key], key))
    top1 = {key: (0.70 if key == ranked[0] else 0.10) for key in OBJECTIVE_KEYS}
    top2 = {
        key: (0.40 if key in ranked[:2] else 0.10) for key in OBJECTIVE_KEYS
    }
    return {
        "uniform": {key: 0.25 for key in OBJECTIVE_KEYS},
        "public_posterior": posterior,
        "top1_moderate": _normalized(top1),
        "top2_ambiguous": _normalized(top2),
    }


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
    ) -> None:
        if min_candidates < 1 or max_candidates < min_candidates:
            raise ValueError("invalid candidate count bounds")
        self.min_candidates = int(min_candidates)
        self.max_candidates = int(max_candidates)
        self.quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)

    def _variants(
        self, observation: Mapping[str, Any], base: Sequence[ActionCandidate]
    ) -> list[tuple[ActionCandidate, dict[str, Any], dict[str, Any]]]:
        beliefs = public_belief_variants(observation)
        rows: list[tuple[ActionCandidate, dict[str, Any], dict[str, Any]]] = []
        for candidate in base:
            packet = copy.deepcopy(candidate.packet)
            rows.append((candidate, packet, {"belief_variant": "teacher_public"}))
            if candidate.category in {"observe", "mitigation"}:
                for belief_name, belief in beliefs.items():
                    variant = copy.deepcopy(packet)
                    variant["belief"] = copy.deepcopy(belief)
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
        actionable = (
            actionable_by_family["passive_verification"][:2]
            + actionable_by_family["active_probe"][:2]
            + actionable_by_family["mitigation"][:4]
        )
        for memory_index, memory_id in enumerate(inventory.memory_ids):
            status = "unknown"
            memory_blob = json.dumps(
                (observation.get("defender_state") or {}).get("memory", {}),
                ensure_ascii=True,
                sort_keys=True,
            )
            if memory_id in memory_blob:
                for candidate_status in ("confirmed", "quarantined", "rejected"):
                    if candidate_status in memory_blob:
                        status = candidate_status
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
        return rows

    def generate_all(self, observation: Mapping[str, Any]) -> list[CandidateOption]:
        assert_public(dict(observation))
        inventory = public_inventory(observation)
        allowed_references = set(
            inventory.event_ids
            + inventory.evidence_ids
            + inventory.source_ids
            + inventory.asset_ids
            + inventory.zones
            + inventory.memory_ids
        )
        base = enumerate_public_candidates(observation, max_candidates=96)
        admitted: dict[str, CandidateOption] = {}
        for candidate, packet, metadata in self._variants(observation, base):
            valid, _ = validate_action_packet_v4(packet)
            references = _references(packet)
            if not valid or not set(references).issubset(allowed_references):
                continue
            option = CandidateOption.from_packet(
                action_family=candidate.category,
                public_summary=_summary(candidate, packet),
                referenced_ids=references,
                compiled_packet=packet,
                action_flags=action_flags(packet),
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
        by_family: dict[str, list[CandidateOption]] = defaultdict(list)
        for option in sorted(
            admitted, key=lambda row: (row.action_family, row.public_summary, row.semantic_id)
        ):
            by_family[option.action_family].append(option)
        selected: list[CandidateOption] = []
        for family in ACTION_FAMILIES:
            selected.extend(by_family[family][: max(0, int(self.quotas.get(family, 0)))])
        selected_ids = {row.semantic_id for row in selected}
        for option in sorted(admitted, key=lambda row: row.semantic_id):
            if len(selected) >= self.max_candidates:
                break
            if option.semantic_id not in selected_ids:
                selected.append(option)
                selected_ids.add(option.semantic_id)
        selected = selected[: self.max_candidates]
        if len(selected) < self.min_candidates:
            raise RuntimeError(
                f"public state produced {len(selected)} candidates; minimum is {self.min_candidates}"
            )
        return self._remap_keys(
            observation,
            selected,
            permutation_seed=int(permutation_seed),
        )
