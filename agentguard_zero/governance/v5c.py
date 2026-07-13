from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable

from agentguard_zero.schemas.action_schema_v4 import (
    DEFAULT_ACTION_PACKET_V4,
    parse_action_json_v4,
)
from agentguard_zero.world.public_projector import assert_public


ACTIVE_PROBES = {"SourceChallenge", "CanaryProbe", "DecoyProbe", "ShadowActionProbe"}
PASSIVE_VERIFICATION = {"CrossCheck", "ProvenanceCheck", "GraphQuery", "BusinessImpactEstimator"}
REVERSIBLE_RESPONSES = {"Observe", "CrossCheck", "QueryProvenance", "DeployDecoy", "ShadowBlock", "LimitSession"}
HIGH_IMPACT_RESPONSES = {"Isolate", "Restore", "Remove"}


@dataclass(frozen=True)
class V5CCandidate:
    index: int
    packet: dict[str, Any]
    parse_ok: bool
    parse_message: str
    score: float
    worst_case_utility: float
    diagnostics: dict[str, Any]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _public_observation(public_context: dict[str, Any]) -> dict[str, Any]:
    observation = public_context.get("observation", public_context)
    return observation if isinstance(observation, dict) else {}


def _available_evidence(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    current_time = int(observation.get("time", 0))
    return {
        str(row.get("evidence_id")): row
        for row in observation.get("available_evidence", []) or []
        if isinstance(row, dict) and int(row.get("available_at", 0)) <= current_time
    }


def _public_risk(observation: dict[str, Any]) -> tuple[float, float, float]:
    events = observation.get("observed_events", []) or []
    assurance = {"high": 0.25, "medium": 0.50, "low": 0.75, "unknown": 0.60}
    source_risk = max(
        (
            assurance.get(str(event.get("source_assurance_level", "unknown")), 0.60)
            for event in events
            if isinstance(event, dict)
        ),
        default=0.50,
    )
    trust = observation.get("defender_state", {}).get("trust", {}) or {}
    claims = trust.get("current_claim_trust", {}) or {}
    contradiction = max(
        (_float(item.get("contradiction_score", 0.0)) for item in claims.values() if isinstance(item, dict)),
        default=0.0,
    )
    memory = observation.get("defender_state", {}).get("memory", {}) or {}
    quarantine = len(memory.get("retrieved_quarantined", []) or [])
    poison_risk = min(1.0, 0.25 * quarantine + contradiction)
    business_remaining = _float(
        observation.get("defense_context", {}).get("remaining_business_budget", 0.0)
    )
    return source_risk, poison_risk, business_remaining


def _reference_diagnostics(packet: dict[str, Any], available: dict[str, dict[str, Any]]) -> tuple[bool, int]:
    refs = [
        str(ref)
        for field in ("trust_operations", "memory_operations")
        for operation in packet.get(field, []) or []
        if isinstance(operation, dict)
        for ref in operation.get("evidence_refs", []) or []
    ]
    return all(ref in available for ref in refs), len(set(refs))


def score_v5c_candidate(public_context: dict[str, Any], text: str, *, index: int = 0) -> V5CCandidate:
    assert_public(public_context)
    observation = _public_observation(public_context)
    available = _available_evidence(observation)
    packet, ok, message = parse_action_json_v4(text)
    if not ok:
        return V5CCandidate(index, packet, False, message, -100.0, -100.0, {"parse_ok": False})

    legal_refs, reference_count = _reference_diagnostics(packet, available)
    source_risk, poison_risk, business_remaining = _public_risk(observation)
    uncertainty = max(0.0, min(1.0, _float(packet.get("uncertainty", 1.0), 1.0)))
    tool = str(packet.get("tool_call", {}).get("name", "None"))
    response = str(packet.get("response", {}).get("action", "Observe"))
    safety = packet.get("safety_check", {}) or {}
    declared_business = max(0.0, _float(safety.get("business_risk", 0.0)))
    declared_overresponse = max(0.0, _float(safety.get("overresponse_risk", 0.0)))
    memory_ops = packet.get("memory_operations", []) or []
    direct_promotions = sum(int(item.get("op") == "promote") for item in memory_ops if isinstance(item, dict))
    quarantining = sum(
        int(item.get("op") in {"ingest", "demote", "reject"})
        for item in memory_ops
        if isinstance(item, dict)
    )
    verification = tool in ACTIVE_PROBES | PASSIVE_VERIFICATION
    active_probe = tool in ACTIVE_PROBES
    high_impact = response in HIGH_IMPACT_RESPONSES
    reversible = response in REVERSIBLE_RESPONSES

    legal = 8.0 if legal_refs else -30.0
    verify = 4.0 * float(verification) + 2.0 * float(active_probe and (source_risk + poison_risk) >= 0.8)
    poison_guard = 2.0 * quarantining - 3.0 * direct_promotions * max(poison_risk, uncertainty)
    mitigate = 1.5 * float(response in {"DeployDecoy", "ShadowBlock", "LimitSession", "Isolate"})
    business_penalty = declared_business + max(0.0, 1.0 - business_remaining) * float(high_impact)
    overresponse_penalty = declared_overresponse + float(high_impact) * (uncertainty + source_risk + poison_risk)
    delay_penalty = 0.4 * float(response == "Observe" and not verification)
    score = legal + verify + poison_guard + mitigate - 2.0 * business_penalty - 3.0 * overresponse_penalty - delay_penalty

    betrayal_case = score - 2.5 * float(high_impact) * source_risk + 1.5 * float(active_probe)
    poisoning_case = score - 2.5 * direct_promotions * poison_risk + 1.0 * quarantining
    benign_change_case = score - 1.5 * float(response in {"Remove", "Isolate"}) + 0.5 * float(reversible)
    worst_case = min(betrayal_case, poisoning_case, benign_change_case)
    diagnostics = {
        "parse_ok": True,
        "legal_refs": legal_refs,
        "reference_count": reference_count,
        "source_risk": source_risk,
        "poison_risk": poison_risk,
        "business_remaining": business_remaining,
        "active_probe": active_probe,
        "high_impact": high_impact,
        "frontier_utilities": {
            "betrayal": betrayal_case,
            "poisoning": poisoning_case,
            "benign_change": benign_change_case,
        },
    }
    return V5CCandidate(index, packet, True, message, float(score), float(worst_case), diagnostics)


def select_v5c(public_context: dict[str, Any], candidates: Iterable[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a candidate using only public observation and defender state."""

    assert_public(public_context)
    scored = [
        score_v5c_candidate(public_context, text, index=index)
        for index, text in enumerate(candidates)
    ]
    if not scored:
        return copy.deepcopy(DEFAULT_ACTION_PACKET_V4), {"selected_index": -1, "fallback": "empty_candidates"}
    valid = [item for item in scored if item.parse_ok]
    selected = max(valid or scored, key=lambda item: (item.worst_case_utility, item.score, -item.index))
    diagnostics = {
        "selector": "v5_c_frontier_minimax",
        "selected_index": selected.index,
        "selected_score": selected.score,
        "selected_worst_case_utility": selected.worst_case_utility,
        "candidates": [
            {
                "index": item.index,
                "parse_ok": item.parse_ok,
                "score": item.score,
                "worst_case_utility": item.worst_case_utility,
                "diagnostics": item.diagnostics,
            }
            for item in scored
        ],
    }
    assert_public(diagnostics)
    return copy.deepcopy(selected.packet), diagnostics


def select_v5c_json(public_context: dict[str, Any], candidates: Iterable[str]) -> tuple[str, dict[str, Any]]:
    packet, diagnostics = select_v5c(public_context, candidates)
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":")), diagnostics
