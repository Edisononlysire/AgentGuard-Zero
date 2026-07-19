"""Finite-Counterfactual Public-State Robust Teacher.

Candidate construction reads public observations only. Hidden simulator state is
consulted exclusively when scoring cloned trajectories, and a single action is
chosen by max-min utility across every hidden world sharing the same public
state. The returned training target contains no world utility or hidden field.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.recovery.utility import (
    RecoveryCoreUtilityConfig,
    recovery_core_utility,
)
from agentguard_zero.schemas.action_schema import OBJECTIVE_KEYS
from agentguard_zero.schemas.action_schema_v4 import (
    DEFAULT_ACTION_PACKET_V4,
    normalize_action_packet_v4,
    validate_action_packet_v4,
)
from agentguard_zero.world.public_projector import assert_public, project_public


TARGETED_ACTIONS = {
    "DeployDecoy",
    "LimitSession",
    "ShadowBlock",
    "Isolate",
    "Restore",
    "Remove",
}
ACTION_CATEGORIES = (
    "observe",
    "passive_verification",
    "active_probe",
    "trust",
    "memory",
    "mitigation",
)
ACTION_HORIZONS = {category: 3 for category in ACTION_CATEGORIES}
SHORTLIST_QUOTAS = {
    "observe": 1,
    "passive_verification": 3,
    "active_probe": 4,
    "trust": 3,
    "memory": 3,
    "mitigation": 6,
}


@dataclass(frozen=True)
class PublicInventory:
    event_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_by_event: dict[str, tuple[str, ...]]
    source_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    zones: tuple[str, ...]
    memory_ids: tuple[str, ...]
    event_by_id: dict[str, dict[str, Any]]
    asset_by_id: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ActionCandidate:
    category: str
    packet: dict[str, Any]
    label: str

    @property
    def candidate_id(self) -> str:
        raw = json.dumps(
            self.packet,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class TeacherDecision:
    public_state_digest: str
    selected_candidate_id: str
    selected_category: str
    selected_packet: dict[str, Any]
    robust_value: float
    observe_value: float
    advantage_over_observe: float
    public_candidate_count: int
    admitted_candidate_count: int
    world_count: int
    search_horizon: int
    q_audit: dict[str, float]
    core_q_audit: dict[str, float]
    hidden_state_in_target: bool = False

    def to_audit_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("selected_packet")
        return payload


@dataclass
class _SearchOption:
    candidate: ActionCandidate
    final_worlds: list[Any]
    robust_value: float
    core_value: float


@dataclass
class _SearchResult:
    selected: _SearchOption
    observe: _SearchOption
    public_candidate_count: int
    admitted_candidate_count: int
    q_audit: dict[str, float]
    core_q_audit: dict[str, float]


def canonical_public_json(value: Any) -> str:
    projected = project_public(value)
    assert_public(projected)
    return json.dumps(
        projected,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def public_state_digest(observation: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_public_json(dict(observation)).encode("utf-8")
    ).hexdigest()


def compact_wire_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_action_packet_v4(dict(packet))
    valid, reason = validate_action_packet_v4(normalized)
    if not valid:
        raise ValueError(f"cannot compact invalid action packet: {reason}")
    return {
        "schema_version": 4,
        "belief": copy.deepcopy(normalized["belief"]),
        "assessment": (
            copy.deepcopy(normalized["evidence_assessment"][0])
            if normalized["evidence_assessment"]
            else None
        ),
        "trust_operation": (
            copy.deepcopy(normalized["trust_operations"][0])
            if normalized["trust_operations"]
            else None
        ),
        "memory_operation": (
            copy.deepcopy(normalized["memory_operations"][0])
            if normalized["memory_operations"]
            else None
        ),
        "memory_use": (
            copy.deepcopy(normalized["memory_usage"][0])
            if normalized["memory_usage"]
            else None
        ),
        "uncertainty": float(normalized["uncertainty"]),
        "tool_call": copy.deepcopy(normalized["tool_call"]),
        "safety_check": copy.deepcopy(normalized["safety_check"]),
        "response": copy.deepcopy(normalized["response"]),
    }


def compact_wire_json(packet: Mapping[str, Any]) -> str:
    return json.dumps(
        compact_wire_packet(packet),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


def public_inventory(observation: Mapping[str, Any]) -> PublicInventory:
    obs = dict(observation)
    assert_public(obs)
    events = [
        item
        for item in obs.get("observed_events", []) or []
        if isinstance(item, dict) and str(item.get("event_id", "")).strip()
    ]
    evidence = [
        item
        for item in obs.get("available_evidence", []) or []
        if isinstance(item, dict) and str(item.get("evidence_id", "")).strip()
    ]
    assets = [
        item
        for item in (obs.get("defense_context", {}) or {}).get("public_assets", [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]
    evidence_by_event: dict[str, list[str]] = {}
    for row in evidence:
        event_id = str(row.get("event_id", "")).strip()
        evidence_id = str(row.get("evidence_id", "")).strip()
        if event_id and evidence_id:
            evidence_by_event.setdefault(event_id, []).append(evidence_id)

    sources = {
        str(item.get("source_id") or item.get("source") or "").strip()
        for item in events
        if str(item.get("source_id") or item.get("source") or "").strip()
    }
    memory_ids = {
        str(item.get("memory_id", "")).strip()
        for item in _walk_dicts((obs.get("defender_state", {}) or {}).get("memory", {}))
        if str(item.get("memory_id", "")).strip()
    }
    return PublicInventory(
        event_ids=tuple(sorted(str(item["event_id"]) for item in events)),
        evidence_ids=tuple(sorted(str(item["evidence_id"]) for item in evidence)),
        evidence_by_event={
            key: tuple(sorted(set(values)))
            for key, values in sorted(evidence_by_event.items())
        },
        source_ids=tuple(sorted(sources)),
        asset_ids=tuple(sorted(str(item["id"]) for item in assets)),
        zones=tuple(
            sorted(
                {
                    str(item.get("zone", "")).strip()
                    for item in assets
                    if str(item.get("zone", "")).strip()
                }
            )
        ),
        memory_ids=tuple(sorted(memory_ids)),
        event_by_id={str(item["event_id"]): copy.deepcopy(item) for item in events},
        asset_by_id={str(item["id"]): copy.deepcopy(item) for item in assets},
    )


def _public_belief(observation: Mapping[str, Any]) -> dict[str, float]:
    support = {key: 0.0 for key in OBJECTIVE_KEYS}
    objective_by_event: dict[str, str] = {}
    evidence = observation.get("available_evidence", []) or []
    for row in evidence:
        if not isinstance(row, dict):
            continue
        content = row.get("content", {}) or {}
        semantics = (
            content.get("claim_semantics", {}) if isinstance(content, dict) else {}
        )
        objective = str(semantics.get("object", "")).strip()
        event_id = str(row.get("event_id", "")).strip()
        if objective in support:
            objective_by_event[event_id] = objective
            if str(row.get("evidence_origin", "")) == "raw_event":
                support[objective] += 0.50
    positive = {
        "support",
        "supported",
        "plausible",
        "challenge_passed",
        "consistent",
    }
    negative = {
        "conflict",
        "suspicious",
        "anomalous",
        "challenge_failed",
        "inconsistent",
    }
    for row in evidence:
        if (
            not isinstance(row, dict)
            or str(row.get("evidence_origin", "")) != "tool_generated"
        ):
            continue
        objective = objective_by_event.get(str(row.get("event_id", "")), "")
        if objective not in support:
            continue
        content = row.get("content", {}) or {}
        signals = {
            str(content.get("verdict", "")).strip().lower(),
            str(content.get("consistency_signal", "")).strip().lower(),
            str(content.get("provenance_signal", "")).strip().lower(),
        }
        if signals & positive:
            support[objective] += 1.50
        if signals & negative:
            support[objective] -= 1.00
    if not any(abs(value) > 1.0e-12 for value in support.values()):
        # Initial raw-event payloads are also present in observed_events before
        # the evidence snapshot is populated by some test adapters.
        for row in observation.get("observed_events", []) or []:
            semantics = row.get("claim_semantics", {}) if isinstance(row, dict) else {}
            objective = str(semantics.get("object", "")).strip()
            if objective in support:
                support[objective] += 0.50
    minimum = min(support.values())
    shifted = {key: value - min(0.0, minimum) + 0.25 for key, value in support.items()}
    total = sum(shifted.values())
    if total <= 0.0:
        return {key: 0.25 for key in OBJECTIVE_KEYS}
    # Public claims and tool results are noisy evidence, not truth; the
    # symmetric prior prevents hidden labels from creating one-hot beliefs.
    return {key: shifted[key] / total for key in OBJECTIVE_KEYS}


def _base_packet(observation: Mapping[str, Any]) -> dict[str, Any]:
    packet = copy.deepcopy(DEFAULT_ACTION_PACKET_V4)
    packet["belief"] = _public_belief(observation)
    packet["uncertainty"] = max(packet["belief"].values())
    packet["uncertainty"] = 1.0 - float(packet["uncertainty"])
    packet["safety_check"]["justification"] = "public-state search"
    return packet


def _assessment(event: Mapping[str, Any], status: str = "unverified") -> dict[str, Any]:
    return {
        "event_id": str(event.get("event_id", "")),
        "status": status,
        "suspected_poisoning": status in {"challenged", "contradicted"},
    }


def _tool_candidate(
    observation: Mapping[str, Any],
    *,
    category: str,
    name: str,
    args: Mapping[str, Any],
    response_action: str,
    response_target: str,
) -> ActionCandidate:
    packet = _base_packet(observation)
    packet["tool_call"] = {"name": name, "args": copy.deepcopy(dict(args))}
    packet["response"] = {
        "tier": "L1",
        "action": response_action,
        "target": response_target,
    }
    return ActionCandidate(category, packet, f"{name}:{response_target}")


def _response_candidate(
    observation: Mapping[str, Any],
    *,
    action: str,
    target: str,
    criticality: float,
) -> ActionCandidate:
    packet = _base_packet(observation)
    tier = "L3" if action in {"Isolate", "Remove"} else "L2"
    packet["response"] = {"tier": tier, "action": action, "target": target}
    packet["safety_check"] = {
        "business_risk": min(1.0, max(0.0, float(criticality))),
        "overresponse_risk": min(1.0, max(0.0, float(criticality))),
        "justification": "public-state admission",
    }
    return ActionCandidate("mitigation", packet, f"{action}:{target}")


def _trust_candidate(
    observation: Mapping[str, Any],
    *,
    operation: str,
    event: Mapping[str, Any],
    evidence_refs: Sequence[str],
) -> ActionCandidate:
    packet = _base_packet(observation)
    source_id = str(event.get("source_id") or event.get("source") or "")
    packet["trust_operations"] = [
        {
            "op": operation,
            "source_id": source_id,
            "event_id": str(event.get("event_id", "")),
            "evidence_refs": list(evidence_refs),
        }
    ]
    packet["evidence_assessment"] = [
        _assessment(
            event,
            "challenged" if operation in {"challenge", "contradict"} else "supported",
        )
    ]
    return ActionCandidate("trust", packet, f"trust:{operation}:{source_id}")


def _memory_candidate(
    observation: Mapping[str, Any],
    *,
    operation: str,
    event: Mapping[str, Any],
    evidence_refs: Sequence[str],
    memory_id: str = "",
) -> ActionCandidate:
    packet = _base_packet(observation)
    if operation == "ingest":
        semantics = event.get("claim_semantics", {}) or {}
        op = {
            "op": "ingest",
            "memory_id": "",
            "event_id": str(event.get("event_id", "")),
            "claim": {
                key: str(semantics.get(key, ""))
                for key in ("entity_id", "predicate", "object", "scope")
            },
            "source_ids": [str(event.get("source_id") or event.get("source") or "")],
            "evidence_refs": list(evidence_refs),
            "target_status": "quarantined",
        }
    else:
        op = {
            "op": operation,
            "memory_id": memory_id,
            "event_id": str(event.get("event_id", "")),
            "evidence_refs": list(evidence_refs),
            "target_status": (
                "confirmed" if operation in {"promote", "reopen"} else "quarantined"
            ),
        }
    packet["memory_operations"] = [op]
    packet["evidence_assessment"] = [_assessment(event)]
    return ActionCandidate("memory", packet, f"memory:{operation}:{memory_id}")


def _candidate_is_public(
    candidate: ActionCandidate,
    inventory: PublicInventory,
) -> bool:
    packet = candidate.packet
    try:
        assert_public(packet)
    except ValueError:
        return False
    valid, _ = validate_action_packet_v4(packet)
    if not valid:
        return False
    event_ids = set(inventory.event_ids)
    evidence_ids = set(inventory.evidence_ids)
    sources = set(inventory.source_ids)
    assets = set(inventory.asset_ids)
    zones = set(inventory.zones)
    memories = set(inventory.memory_ids)

    response = packet.get("response", {}) or {}
    if (
        response.get("action") in TARGETED_ACTIONS
        and response.get("target") not in assets
    ):
        return False
    tool = packet.get("tool_call", {}) or {}
    args = tool.get("args", {}) or {}
    if "event_id" in args and str(args["event_id"]) not in event_ids:
        return False
    if "evidence_ids" in args and not set(map(str, args["evidence_ids"])).issubset(
        evidence_ids
    ):
        return False
    if "source" in args and str(args["source"]) not in sources:
        return False
    if "node" in args and str(args["node"]) not in assets:
        return False
    if "zone" in args and str(args["zone"]) not in zones:
        return False
    nested_action = args.get("action")
    if (
        isinstance(nested_action, dict)
        and str(nested_action.get("target", "")) not in assets
    ):
        return False

    for operation in packet.get("trust_operations", []) or []:
        if str(operation.get("source_id", "")) not in sources:
            return False
        if str(operation.get("event_id", "")) not in event_ids:
            return False
        if not set(map(str, operation.get("evidence_refs", []))).issubset(evidence_ids):
            return False
    for operation in packet.get("memory_operations", []) or []:
        if str(operation.get("event_id", "")) not in event_ids:
            return False
        if not set(map(str, operation.get("evidence_refs", []))).issubset(evidence_ids):
            return False
        if (
            operation.get("op") != "ingest"
            and str(operation.get("memory_id", "")) not in memories
        ):
            return False
    for usage in packet.get("memory_usage", []) or []:
        if str(usage.get("memory_id", "")) not in memories:
            return False
    return True


def enumerate_public_candidates(
    observation: Mapping[str, Any],
    *,
    max_candidates: int = 96,
) -> list[ActionCandidate]:
    inventory = public_inventory(observation)
    candidates: list[ActionCandidate] = [
        ActionCandidate(
            "observe",
            _base_packet(observation),
            "Observe",
        )
    ]

    for event_id in inventory.event_ids:
        event = inventory.event_by_id[event_id]
        source = str(event.get("source_id") or event.get("source") or "")
        evidence_refs = inventory.evidence_by_event.get(event_id, ())
        if source:
            candidates.append(
                _tool_candidate(
                    observation,
                    category="passive_verification",
                    name="LogQuery",
                    args={"source": source, "time": int(observation.get("time", 0))},
                    response_action="Observe",
                    response_target="none",
                )
            )
        if evidence_refs:
            candidates.append(
                _tool_candidate(
                    observation,
                    category="passive_verification",
                    name="CrossCheck",
                    args={
                        "event_id": event_id,
                        "evidence_ids": list(evidence_refs[:6]),
                    },
                    response_action="CrossCheck",
                    response_target=event_id,
                )
            )
        candidates.append(
            _tool_candidate(
                observation,
                category="passive_verification",
                name="ProvenanceCheck",
                args={"event_id": event_id},
                response_action="QueryProvenance",
                response_target=event_id,
            )
        )
        for tool in ("SourceChallenge", "CanaryProbe"):
            candidates.append(
                _tool_candidate(
                    observation,
                    category="active_probe",
                    name=tool,
                    args={"event_id": event_id},
                    response_action=tool,
                    response_target=event_id,
                )
            )
        if evidence_refs and source:
            candidates.append(
                _trust_candidate(
                    observation,
                    operation="hold",
                    event=event,
                    evidence_refs=(),
                )
            )
            for operation in ("support", "challenge", "contradict", "recover"):
                candidates.append(
                    _trust_candidate(
                        observation,
                        operation=operation,
                        event=event,
                        evidence_refs=evidence_refs[:6],
                    )
                )
            semantics = event.get("claim_semantics", {}) or {}
            if all(
                str(semantics.get(key, "")).strip()
                for key in ("entity_id", "predicate", "object", "scope")
            ):
                candidates.append(
                    _memory_candidate(
                        observation,
                        operation="ingest",
                        event=event,
                        evidence_refs=evidence_refs[:6],
                    )
                )
            for memory_id in inventory.memory_ids:
                for operation in ("promote", "demote", "reject", "reopen"):
                    candidates.append(
                        _memory_candidate(
                            observation,
                            operation=operation,
                            event=event,
                            evidence_refs=evidence_refs[:6],
                            memory_id=memory_id,
                        )
                    )

    for asset_id in inventory.asset_ids:
        criticality = float(inventory.asset_by_id[asset_id].get("criticality", 0.5))
        candidates.append(
            _tool_candidate(
                observation,
                category="passive_verification",
                name="GraphQuery",
                args={"node": asset_id},
                response_action="Observe",
                response_target="none",
            )
        )
        for action in ("LimitSession", "ShadowBlock"):
            candidates.append(
                _tool_candidate(
                    observation,
                    category="passive_verification",
                    name="BusinessImpactEstimator",
                    args={"action": {"action": action, "target": asset_id}},
                    response_action="Observe",
                    response_target="none",
                )
            )
            candidates.append(
                _tool_candidate(
                    observation,
                    category="active_probe",
                    name="ShadowActionProbe",
                    args={"action": {"action": action, "target": asset_id}},
                    response_action="ShadowActionProbe",
                    response_target=asset_id,
                )
            )
        for action in (
            "DeployDecoy",
            "LimitSession",
            "ShadowBlock",
            "Isolate",
            "Restore",
            "Remove",
        ):
            candidates.append(
                _response_candidate(
                    observation,
                    action=action,
                    target=asset_id,
                    criticality=criticality,
                )
            )

    for zone in inventory.zones:
        candidates.append(
            _tool_candidate(
                observation,
                category="active_probe",
                name="DecoyProbe",
                args={"zone": zone},
                response_action="DecoyProbe",
                response_target=zone,
            )
        )

    memory_status: dict[str, str] = {}
    memory_blob = (observation.get("defender_state", {}) or {}).get("memory", {}) or {}
    for row in _walk_dicts(memory_blob):
        memory_id = str(row.get("memory_id", ""))
        if memory_id:
            memory_status[memory_id] = str(row.get("status", "quarantined"))
    combinable = [
        row
        for row in candidates
        if row.category in {"passive_verification", "active_probe", "mitigation"}
    ]
    for memory_id in inventory.memory_ids:
        role = "support" if memory_status.get(memory_id) == "confirmed" else "contradict"
        per_category: dict[str, int] = {}
        for base in combinable:
            if per_category.get(base.category, 0) >= (4 if base.category == "mitigation" else 2):
                continue
            packet = copy.deepcopy(base.packet)
            packet["memory_usage"] = [
                {
                    "memory_id": memory_id,
                    "usage": role,
                    "used_for": "response" if base.category == "mitigation" else "tool",
                }
            ]
            candidates.append(
                ActionCandidate(
                    base.category,
                    packet,
                    f"memory_use:{memory_id}+{base.label}",
                )
            )
            per_category[base.category] = per_category.get(base.category, 0) + 1

    trust_packets = [
        row for row in candidates if row.category == "trust" and row.packet.get("trust_operations")
    ]
    probe_packets = [row for row in candidates if row.category == "active_probe"]
    for trust in trust_packets[:2]:
        for probe in probe_packets[:3]:
            packet = copy.deepcopy(probe.packet)
            packet["trust_operations"] = copy.deepcopy(trust.packet["trust_operations"])
            packet["evidence_assessment"] = copy.deepcopy(
                trust.packet.get("evidence_assessment", [])
            )
            candidates.append(
                ActionCandidate(
                    "active_probe",
                    packet,
                    f"trust_probe:{trust.label}+{probe.label}",
                )
            )

    deduped: dict[str, ActionCandidate] = {}
    for candidate in candidates:
        if _candidate_is_public(candidate, inventory):
            deduped.setdefault(candidate.candidate_id, candidate)

    # Preserve all action classes before filling the remaining deterministic
    # budget. This avoids a long event list crowding out mitigation or state
    # operations.
    ordered = sorted(
        deduped.values(),
        key=lambda item: (item.category, item.label, item.candidate_id),
    )
    selected: list[ActionCandidate] = []
    for category in (
        "observe",
        "passive_verification",
        "active_probe",
        "trust",
        "memory",
        "mitigation",
    ):
        first = next((item for item in ordered if item.category == category), None)
        if first is not None:
            selected.append(first)
    selected_ids = {item.candidate_id for item in selected}
    selected.extend(item for item in ordered if item.candidate_id not in selected_ids)
    return selected[: max(1, int(max_candidates))]


def admitted_public_candidates(
    worlds: Sequence[Any],
    *,
    max_candidates: int = 96,
) -> list[ActionCandidate]:
    """Return candidates admitted by every hidden world in a public-state group."""

    if not worlds:
        raise ValueError("candidate admission requires at least one world")
    observations = [world.observe() for world in worlds]
    if len({public_state_digest(item) for item in observations}) != 1:
        raise ValueError("candidate admission worlds have different public states")
    return [
        candidate
        for candidate in enumerate_public_candidates(
            observations[0],
            max_candidates=max_candidates,
        )
        if all(_simulate_one_step(world, candidate) is not None for world in worlds)
    ]


def _environment_fingerprint(env: Any) -> str:
    try:
        payload = pickle.dumps(env, protocol=5)
    except Exception:
        payload = repr(env.__dict__).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _simulate_one_step(env: Any, candidate: ActionCandidate) -> Any | None:
    before = _environment_fingerprint(env)
    cloned = copy.deepcopy(env)
    cloned.step(copy.deepcopy(candidate.packet))
    if _environment_fingerprint(env) != before:
        raise RuntimeError("teacher simulation mutated the source environment")
    step = cloned.history[-1]
    response = step.get("response_result", {}) or {}
    proposed = str(response.get("proposed_action", "Observe"))
    if proposed != "Observe" and bool(response.get("blocked", False)):
        return None
    tool = candidate.packet.get("tool_call", {}) or {}
    if str(tool.get("name", "None")) != "None":
        result = step.get("tool_result", {}) or {}
        if result.get("error") or result.get("status") in {
            "invalid_reference",
            "budget_exhausted",
        }:
            return None
    if candidate.category == "trust" and not any(
        bool(item.get("committed", False)) for item in step.get("trust_events", [])
    ):
        return None
    if candidate.category == "memory" and not any(
        bool(item.get("committed", False)) for item in step.get("memory_events", [])
    ):
        return None
    return cloned


def teacher_rollout_shaping(env: Any) -> float:
    score = score_trajectory_v2(env)
    pressure = float(getattr(env.hidden_world, "attack_pressure", 1.0))
    pressure_reduction = max(0.0, min(1.0, 1.0 - pressure))
    probe_information = float(
        bool((getattr(env, "last_tool_result", {}) or {}).get("evidence_id"))
    )
    state_commits = sum(
        int(bool(item.get("committed", False)))
        for step in env.history
        for key in ("trust_events", "memory_events")
        for item in step.get(key, []) or []
        if isinstance(item, dict)
    )
    state_quality = max(
        -1.0,
        min(
            1.0,
            float(score.get("state_reward_component", 0.0)) / 0.25,
        ),
    )
    # Shaping is intentionally bounded to 25% of one mitigation outcome.  It
    # may break ties among partial trajectories, but cannot replace U_core.
    return float(
        0.13 * pressure_reduction
        + 0.02 * float(bool(score.get("correct_intent", False)))
        + 0.03 * probe_information
        + 0.01 * min(1.0, float(score.get("probe_yield", 0.0)))
        # Reward only offline-validated state quality. A committed but harmful
        # trust/memory transition receives negative shaping; a correct one may
        # clear the 0.05 action threshold after public evidence disambiguates
        # the matched worlds.
        + 0.06 * min(1, state_commits) * state_quality
    )


def teacher_rollout_utility(env: Any) -> float:
    cfg = RecoveryCoreUtilityConfig()
    core = recovery_core_utility(env, score_trajectory_v2(env), cfg)
    shaping = teacher_rollout_shaping(env)
    if abs(shaping) > cfg.teacher_shaping_cap + 1.0e-12:
        raise RuntimeError("teacher shaping exceeds the frozen core-utility cap")
    return float(core + shaping)


class PublicStateRobustTeacher:
    def __init__(
        self,
        *,
        advantage_delta: float = 0.05,
        min_worlds_per_public_state: int = 2,
        beam_width: int = 20,
        max_candidates: int = 96,
        allowed_categories: Sequence[str] | None = None,
        disabled_skills: Sequence[str] = (),
        core_tolerance: float = 0.02,
    ) -> None:
        if advantage_delta < 0.0:
            raise ValueError("advantage_delta must be non-negative")
        if min_worlds_per_public_state < 2:
            raise ValueError("robust teacher requires at least two hidden worlds")
        if not 16 <= int(beam_width) <= 24:
            raise ValueError("formal teacher beam_width must be in [16,24]")
        if int(max_candidates) < int(beam_width):
            raise ValueError("max_candidates must be at least beam_width")
        self.advantage_delta = float(advantage_delta)
        self.min_worlds = int(min_worlds_per_public_state)
        self.beam_width = int(beam_width)
        self.continuation_beam_width = min(8, self.beam_width)
        self.max_candidates = max(1, int(max_candidates))
        self.allowed_categories = frozenset(allowed_categories or ACTION_CATEGORIES)
        self.disabled_skills = frozenset(map(str, disabled_skills))
        if "observe" not in self.allowed_categories:
            raise ValueError("core-first teacher always requires Observe")
        unknown = self.allowed_categories.difference(ACTION_CATEGORIES)
        if unknown:
            raise ValueError(f"unknown teacher action categories: {sorted(unknown)}")
        if core_tolerance < 0.0:
            raise ValueError("core_tolerance must be non-negative")
        self.core_tolerance = float(core_tolerance)

    @staticmethod
    def _candidate_family(candidate: ActionCandidate) -> str:
        parts = candidate.label.split(":")
        if candidate.category in {"trust", "memory"}:
            return ":".join(parts[:2])
        return parts[0]

    def _candidate_pool(
        self,
        candidates: Sequence[ActionCandidate],
        *,
        continuation_mode: bool,
        observation: Mapping[str, Any],
    ) -> list[ActionCandidate]:
        """Choose a public-only diverse pool before expensive world cloning.

        The former implementation scored all candidates for one step and then
        kept one per category.  This deterministic family-balanced pool avoids
        that myopic pruning while keeping H=3 planning computationally finite.
        """

        limit = (
            self.continuation_beam_width
            if continuation_mode
            else min(
                self.max_candidates,
                self.beam_width + 12,
            )
        )
        selected: list[ActionCandidate] = []

        def add(candidate: ActionCandidate) -> None:
            if all(row.candidate_id != candidate.candidate_id for row in selected):
                selected.append(candidate)

        observed_entities = {
            str(item.get("entity_id", "")).strip()
            for item in observation.get("observed_events", []) or []
            if isinstance(item, dict) and str(item.get("entity_id", "")).strip()
        }
        for category in ACTION_CATEGORIES:

            def public_priority(row: ActionCandidate) -> tuple[Any, ...]:
                response = row.packet.get("response", {}) or {}
                target = str(response.get("target", ""))
                requirements = (
                    (observation.get("defense_context", {}) or {}).get(
                        "response_requirements", {}
                    )
                    or {}
                )
                mitigation_order = {
                    "ShadowBlock": 0,
                    "LimitSession": 1,
                    "DeployDecoy": 2,
                    "Restore": 3,
                    "Remove": 4,
                    "Isolate": 5,
                }
                return (
                    (
                        0
                        if not requirements.get("memory_use_required_for_mitigation")
                        or row.category != "mitigation"
                        or bool(row.packet.get("memory_usage"))
                        else 1
                    ),
                    0 if target in observed_entities else 1,
                    (
                        mitigation_order.get(str(response.get("action", "")), 0)
                        if row.category == "mitigation"
                        else 0
                    ),
                    row.label,
                    row.candidate_id,
                )

            rows = sorted(
                (row for row in candidates if row.category == category),
                key=public_priority,
            )
            quota = 1 if continuation_mode else SHORTLIST_QUOTAS[category] + 1
            families: set[str] = set()
            for row in rows:
                family = self._candidate_family(row)
                if family in families:
                    continue
                add(row)
                families.add(family)
                if sum(item.category == category for item in selected) >= quota:
                    break
            for row in rows:
                if sum(item.category == category for item in selected) >= quota:
                    break
                add(row)
        for row in candidates:
            if len(selected) >= limit:
                break
            add(row)
        return selected[:limit]

    def _public_observation(self, worlds: Sequence[Any]) -> dict[str, Any]:
        if not worlds:
            raise ValueError("teacher world group is empty")
        observations = [world.observe() for world in worlds]
        digests = {public_state_digest(item) for item in observations}
        if len(digests) != 1:
            raise ValueError("hidden worlds do not share an identical public state")
        return observations[0]

    def _simulate_candidate(
        self,
        worlds: Sequence[Any],
        candidate: ActionCandidate,
    ) -> list[Any] | None:
        simulated: list[Any] = []
        for world in worlds:
            result = _simulate_one_step(world, candidate)
            if result is None:
                return None
            simulated.append(result)
        return simulated

    def _shortlist(
        self,
        options: Sequence[_SearchOption],
    ) -> list[_SearchOption]:
        ranked = sorted(
            options,
            key=lambda row: (-row.robust_value, row.candidate.candidate_id),
        )
        selected: list[_SearchOption] = []

        def add(item: _SearchOption) -> None:
            if any(
                seen.candidate.candidate_id == item.candidate.candidate_id
                for seen in selected
            ):
                return
            selected.append(item)

        # First preserve a diverse set inside every class.  Candidate labels
        # encode the tool/operation/mitigation family before the final target.
        # This prevents a one-step tie from deleting every alternative probe or
        # every alternative response before the H=3 comparison.
        for category in ACTION_CATEGORIES:
            category_rows = [
                item for item in ranked if item.candidate.category == category
            ]
            quota = SHORTLIST_QUOTAS[category]
            family_seen: set[str] = set()
            for item in category_rows:
                parts = item.candidate.label.split(":")
                family = (
                    ":".join(parts[:2]) if category in {"trust", "memory"} else parts[0]
                )
                if family in family_seen:
                    continue
                add(item)
                family_seen.add(family)
                if sum(row.candidate.category == category for row in selected) >= quota:
                    break
            for item in category_rows:
                if sum(row.candidate.category == category for row in selected) >= quota:
                    break
                add(item)

        for item in ranked:
            if len(selected) >= self.beam_width:
                break
            add(item)
        observe = next(
            (item for item in options if item.candidate.category == "observe"),
            None,
        )
        if observe is not None and all(
            item.candidate.candidate_id != observe.candidate.candidate_id
            for item in selected
        ):
            selected.append(observe)
        return selected

    def _continue_groups(
        self,
        worlds: Sequence[Any],
        horizon: int,
    ) -> list[Any]:
        # Evaluate every root candidate over the same H=3, then use a greedy
        # receding-horizon public policy for its continuation.  This is a
        # finite-counterfactual beam rollout, not an exponential exhaustive
        # minimax tree; the distinction is recorded in the method claim.
        current = list(worlds)
        for _ in range(max(0, horizon)):
            completed = [
                world
                for world in current
                if world.t >= world.max_steps
                or world.attack_mitigated
                or world.attack_success
            ]
            live = [world for world in current if world not in completed]
            if not live:
                return completed
            grouped: dict[str, list[Any]] = {}
            for world in live:
                observation = world.observe()
                grouped.setdefault(public_state_digest(observation), []).append(world)
            continued: list[Any] = list(completed)
            for group in grouped.values():
                continued.extend(
                    self._search(
                        group,
                        1,
                        enforce_min_worlds=False,
                        continuation_mode=True,
                    ).selected.final_worlds
                )
            current = continued
        return current

    def _search(
        self,
        worlds: Sequence[Any],
        horizon: int,
        *,
        enforce_min_worlds: bool,
        continuation_mode: bool = False,
    ) -> _SearchResult:
        if enforce_min_worlds and len(worlds) < self.min_worlds:
            raise ValueError(
                f"public state has {len(worlds)} hidden world(s); "
                f"minimum is {self.min_worlds}"
            )
        observation = self._public_observation(worlds)
        public_candidates = enumerate_public_candidates(
            observation,
            max_candidates=self.max_candidates,
        )
        public_candidates = [
            row for row in public_candidates if row.category in self.allowed_categories
        ]
        if "active_probe" in self.disabled_skills:
            public_candidates = [
                row for row in public_candidates if row.category != "active_probe"
            ]
        if "trust" in self.disabled_skills:
            public_candidates = [
                row
                for row in public_candidates
                if row.category != "trust" and not row.packet.get("trust_operations")
            ]
        if "memory" in self.disabled_skills:
            public_candidates = [
                row
                for row in public_candidates
                if row.category != "memory"
                and not row.packet.get("memory_operations")
                and not row.packet.get("memory_usage")
            ]
        if "business_response" in self.disabled_skills:
            public_candidates = [
                row
                for row in public_candidates
                if str((row.packet.get("tool_call") or {}).get("name", ""))
                not in {"BusinessImpactEstimator", "ShadowActionProbe"}
            ]
        immediate: list[_SearchOption] = []
        candidate_pool = self._candidate_pool(
            public_candidates,
            continuation_mode=continuation_mode,
            observation=observation,
        )
        for candidate in candidate_pool:
            next_worlds = self._simulate_candidate(worlds, candidate)
            if next_worlds is None:
                continue
            immediate.append(
                _SearchOption(
                    candidate=candidate,
                    final_worlds=next_worlds,
                    robust_value=min(
                        teacher_rollout_utility(item) for item in next_worlds
                    ),
                    core_value=min(
                        recovery_core_utility(item, score_trajectory_v2(item))
                        for item in next_worlds
                    ),
                )
            )
        if not immediate:
            raise RuntimeError("public teacher found no commonly admitted action")

        completed: list[_SearchOption] = []
        for option in self._shortlist(immediate):
            option_horizon = min(horizon, ACTION_HORIZONS[option.candidate.category])
            final_worlds = (
                self._continue_groups(option.final_worlds, option_horizon - 1)
                if option_horizon > 1
                else option.final_worlds
            )
            completed.append(
                _SearchOption(
                    candidate=option.candidate,
                    final_worlds=final_worlds,
                    robust_value=min(
                        teacher_rollout_utility(item) for item in final_worlds
                    ),
                    core_value=min(
                        recovery_core_utility(item, score_trajectory_v2(item))
                        for item in final_worlds
                    ),
                )
            )
        observe = next(
            item for item in completed if item.candidate.category == "observe"
        )
        core_eligible = [
            item
            for item in completed
            if item.core_value >= observe.core_value - self.core_tolerance - 1.0e-12
        ]
        best = max(
            core_eligible,
            key=lambda item: (item.robust_value, item.candidate.candidate_id),
        )
        # Equality belongs to Observe.  The tolerance prevents a nominal
        # 0.05 advantage from becoming active only because of float rounding.
        if best.robust_value <= observe.robust_value + self.advantage_delta + 1.0e-12:
            selected = observe
        else:
            selected = best
        return _SearchResult(
            selected=selected,
            observe=observe,
            public_candidate_count=len(public_candidates),
            admitted_candidate_count=len(immediate),
            q_audit={
                item.candidate.candidate_id: float(item.robust_value)
                for item in completed
            },
            core_q_audit={
                item.candidate.candidate_id: float(item.core_value)
                for item in completed
            },
        )

    def decide(
        self,
        worlds: Sequence[Any],
        *,
        horizon: int = 3,
        enforce_min_worlds: bool = True,
    ) -> TeacherDecision:
        if horizon != 3:
            raise ValueError("formal teacher root horizon must equal 3")
        observation = self._public_observation(worlds)
        result = self._search(
            worlds,
            horizon,
            enforce_min_worlds=enforce_min_worlds,
        )
        selected = result.selected
        observe = result.observe
        packet = copy.deepcopy(selected.candidate.packet)
        assert_public(packet)
        return TeacherDecision(
            public_state_digest=public_state_digest(observation),
            selected_candidate_id=selected.candidate.candidate_id,
            selected_category=selected.candidate.category,
            selected_packet=packet,
            robust_value=float(selected.robust_value),
            observe_value=float(observe.robust_value),
            advantage_over_observe=float(selected.robust_value - observe.robust_value),
            public_candidate_count=result.public_candidate_count,
            admitted_candidate_count=result.admitted_candidate_count,
            world_count=len(worlds),
            search_horizon=int(horizon),
            q_audit=result.q_audit,
            core_q_audit=result.core_q_audit,
        )
