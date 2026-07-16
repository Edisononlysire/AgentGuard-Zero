from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable

from agentguard_zero.governance.authorization import authorize_public_response
from agentguard_zero.schemas.action_schema_v4 import (
    DEFAULT_ACTION_PACKET_V4,
    parse_action_json_v4,
)
from agentguard_zero.runtime_policy import (
    HIGH_IMPACT_ACTIONS,
    TARGETED_RESPONSE_ACTIONS,
    TOOL_COSTS,
)
from agentguard_zero.tools.business_impact import estimate_business_impact
from agentguard_zero.world.public_projector import assert_public


ACTIVE_PROBES = {"SourceChallenge", "CanaryProbe", "DecoyProbe", "ShadowActionProbe"}
PASSIVE_VERIFICATION = {"CrossCheck", "ProvenanceCheck", "GraphQuery", "BusinessImpactEstimator"}
REVERSIBLE_RESPONSES = {"Observe", "CrossCheck", "QueryProvenance", "DeployDecoy", "ShadowBlock", "LimitSession"}


@dataclass(frozen=True)
class V5CCandidate:
    index: int
    packet: dict[str, Any]
    parse_ok: bool
    parse_message: str
    admissible: bool
    hard_violations: tuple[str, ...]
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
    trust = observation.get("defender_state", {}).get("trust", {}) or {}
    sources = trust.get("source_reputation", {}) or {}
    source_risk = max(
        (
            min(
                1.0,
                1.0 - _float(item.get("mean", 0.5), 0.5)
                + 0.5 * _float(item.get("uncertainty", 1.0), 1.0),
            )
            for item in sources.values()
            if isinstance(item, dict)
        ),
        default=0.50,
    )
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


def _memory_records(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    memory = observation.get("defender_state", {}).get("memory", {}) or {}
    return {
        str(record.get("memory_id")): record
        for key in ("retrieved_confirmed", "retrieved_quarantined", "rejected_warnings")
        for record in memory.get(key, []) or []
        if isinstance(record, dict) and str(record.get("memory_id", ""))
    }


def _evidence_roots(record: dict[str, Any]) -> set[str]:
    roots = {str(item) for item in record.get("root_source_ids", []) or [] if str(item)}
    if not roots and str(record.get("source_id", "")):
        roots.add(str(record["source_id"]))
    return roots


def _hard_gate(
    packet: dict[str, Any],
    observation: dict[str, Any],
    available: dict[str, dict[str, Any]],
) -> tuple[bool, tuple[str, ...]]:
    violations: set[str] = set()
    legal_refs, _ = _reference_diagnostics(packet, available)
    if not legal_refs:
        violations.add("unknown_or_unavailable_evidence_ref")

    trust = observation.get("defender_state", {}).get("trust", {}) or {}
    claims = trust.get("current_claim_trust", {}) or {}
    public_event_ids = {
        str(event.get("event_id", ""))
        for event in observation.get("observed_events", []) or []
        if isinstance(event, dict)
    } | {str(event_id) for event_id in claims}
    for assessment in packet.get("evidence_assessment", []) or []:
        if str(assessment.get("event_id", "")) not in public_event_ids:
            violations.add("unknown_event_ref")

    for operation in packet.get("trust_operations", []) or []:
        event_id = str(operation.get("event_id", ""))
        if operation.get("op") != "hold" and event_id not in claims:
            violations.add("unknown_trust_claim")
            continue
        claim = claims.get(event_id, {}) if event_id else {}
        if claim and str(claim.get("source_id", "")) != str(operation.get("source_id", "")):
            violations.add("trust_claim_source_mismatch")

    memory_records = _memory_records(observation)
    for operation in packet.get("memory_operations", []) or []:
        op = str(operation.get("op", ""))
        refs = [str(item) for item in operation.get("evidence_refs", []) or []]
        if op == "ingest":
            derived_sources: set[str] = set()
            for ref in refs:
                derived_sources.update(_evidence_roots(available.get(ref, {})))
            declared_sources = {
                str(item) for item in operation.get("source_ids", []) or [] if str(item)
            }
            if declared_sources and declared_sources != derived_sources:
                violations.add("memory_source_lineage_mismatch")
            continue
        memory_id = str(operation.get("memory_id", ""))
        record = memory_records.get(memory_id)
        if record is None:
            violations.add("unknown_memory_ref")
        elif op == "promote" and str(record.get("status")) != "quarantined":
            violations.add("invalid_memory_promotion_state")

    tool = str(packet.get("tool_call", {}).get("name", "None"))
    tool_args = packet.get("tool_call", {}).get("args", {}) or {}
    if tool == "CrossCheck":
        crosscheck_refs = [
            str(item) for item in tool_args.get("evidence_ids", []) or [] if str(item)
        ]
        if not crosscheck_refs:
            violations.add("crosscheck_requires_evidence_ids")
        elif any(ref not in available for ref in crosscheck_refs):
            violations.add("crosscheck_unknown_or_unavailable_evidence")
        else:
            target_event_id = str(tool_args.get("event_id", ""))
            target_record = next(
                (
                    record
                    for record in available.values()
                    if str(record.get("event_id", "")) == target_event_id
                ),
                None,
            )
            target_claim = (
                (target_record.get("content", {}) or {}).get("claim_semantics", {})
                if isinstance(target_record, dict)
                else {}
            )
            if not target_claim:
                violations.add("crosscheck_unknown_event_claim")
            elif any(
                ((available[ref].get("content", {}) or {}).get("claim_semantics", {}) or {})
                != target_claim
                for ref in crosscheck_refs
            ):
                violations.add("crosscheck_incompatible_claim_evidence")
    remaining_verification = _float(
        observation.get("defense_context", {}).get("remaining_verification_budget", 0.0)
    )
    if _float(TOOL_COSTS.get(tool, 0.0)) > remaining_verification + 1e-9:
        violations.add("verification_budget_exhausted")

    defense_context = observation.get("defense_context", {}) or {}
    authorization = authorize_public_response(
        response=packet.get("response", {}) or {},
        evidence_assessment=packet.get("evidence_assessment", []) or [],
        trust_snapshot=trust,
        evidence_snapshot=available.values(),
        public_assets=defense_context.get("public_assets", []) or [],
        business_state={
            "remaining_business_budget": defense_context.get(
                "remaining_business_budget", float("inf")
            ),
            "remaining_high_impact_actions": defense_context.get(
                "remaining_high_impact_actions"
            ),
        },
    )
    if not authorization.allowed:
        violations.add(authorization.reason)
    return not violations, tuple(sorted(violations))


def score_v5c_candidate(public_context: dict[str, Any], text: str, *, index: int = 0) -> V5CCandidate:
    assert_public(public_context)
    observation = _public_observation(public_context)
    available = _available_evidence(observation)
    packet, ok, message = parse_action_json_v4(text)
    if not ok:
        return V5CCandidate(
            index,
            packet,
            False,
            message,
            False,
            ("schema_invalid",),
            -100.0,
            -100.0,
            {"parse_ok": False, "admissible": False, "hard_violations": ["schema_invalid"]},
        )

    legal_refs, reference_count = _reference_diagnostics(packet, available)
    admissible, hard_violations = _hard_gate(packet, observation, available)
    source_risk, poison_risk, business_remaining = _public_risk(observation)
    trust_sources = observation.get("defender_state", {}).get("trust", {}).get("source_reputation", {}) or {}
    uncertainty = max(
        (_float(item.get("uncertainty", 1.0), 1.0) for item in trust_sources.values() if isinstance(item, dict)),
        default=1.0,
    )
    tool = str(packet.get("tool_call", {}).get("name", "None"))
    response = str(packet.get("response", {}).get("action", "Observe"))
    target = str(packet.get("response", {}).get("target", ""))
    asset = next(
        (
            item
            for item in observation.get("defense_context", {}).get("public_assets", []) or []
            if isinstance(item, dict) and str(item.get("id", "")) == target
        ),
        {},
    )
    impact = estimate_business_impact(
        packet.get("response", {}) or {},
        _float(asset.get("criticality", 0.5), 0.5),
    )
    estimated_business = max(0.0, _float(impact.get("estimated_cost", 0.0)))
    memory_ops = packet.get("memory_operations", []) or []
    direct_promotions = sum(int(item.get("op") == "promote") for item in memory_ops if isinstance(item, dict))
    quarantining = sum(
        int(item.get("op") in {"ingest", "demote", "reject"})
        for item in memory_ops
        if isinstance(item, dict)
    )
    verification = tool in ACTIVE_PROBES | PASSIVE_VERIFICATION
    active_probe = tool in ACTIVE_PROBES
    high_impact = response in HIGH_IMPACT_ACTIONS
    reversible = response in REVERSIBLE_RESPONSES

    legal = 8.0
    verify = 4.0 * float(verification) + 2.0 * float(active_probe and (source_risk + poison_risk) >= 0.8)
    poison_guard = 2.0 * quarantining - 3.0 * direct_promotions * max(poison_risk, uncertainty)
    mitigate = 1.5 * float(response in {"DeployDecoy", "ShadowBlock", "LimitSession", "Isolate"})
    business_penalty = estimated_business + max(0.0, estimated_business - business_remaining)
    overresponse_penalty = float(high_impact) * (uncertainty + source_risk + poison_risk)
    delay_penalty = 0.4 * float(response == "Observe" and not verification)
    score = legal + verify + poison_guard + mitigate - 2.0 * business_penalty - 3.0 * overresponse_penalty - delay_penalty

    betrayal_case = score - 2.5 * float(high_impact) * source_risk + 1.5 * float(active_probe)
    poisoning_case = score - 2.5 * direct_promotions * poison_risk + 1.0 * quarantining
    benign_change_case = score - 1.5 * float(response in {"Remove", "Isolate"}) + 0.5 * float(reversible)
    worst_case = min(betrayal_case, poisoning_case, benign_change_case)
    diagnostics = {
        "parse_ok": True,
        "admissible": admissible,
        "hard_violations": list(hard_violations),
        "legal_refs": legal_refs,
        "reference_count": reference_count,
        "source_risk": source_risk,
        "poison_risk": poison_risk,
        "business_remaining": business_remaining,
        "active_probe": active_probe,
        "high_impact": high_impact,
        "robust_scenario_utilities": {
            "betrayal": betrayal_case,
            "poisoning": poisoning_case,
            "benign_change": benign_change_case,
        },
    }
    return V5CCandidate(
        index,
        packet,
        True,
        message,
        admissible,
        hard_violations,
        float(score),
        float(worst_case),
        diagnostics,
    )


def safe_probe_fallback(
    public_context: dict[str, Any],
    scored: Iterable[V5CCandidate] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a bounded public-state probe when every sampled action is inadmissible."""

    assert_public(public_context)
    observation = _public_observation(public_context)
    remaining = _float(
        observation.get("defense_context", {}).get("remaining_verification_budget", 0.0)
    )
    events = [
        event
        for event in observation.get("observed_events", []) or []
        if isinstance(event, dict) and str(event.get("event_id", ""))
    ]
    claims = (
        observation.get("defender_state", {})
        .get("trust", {})
        .get("current_claim_trust", {})
        or {}
    )

    def packet_for(tool: str, args: dict[str, Any], reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
        packet = copy.deepcopy(DEFAULT_ACTION_PACKET_V4)
        packet["tool_call"] = {"name": tool, "args": copy.deepcopy(args)}
        packet["safety_check"]["justification"] = reason
        return packet, {
            "fallback": "safe_active_probe",
            "fallback_type": tool,
            "fallback_reason": reason,
        }

    unresolved = [
        event
        for event in events
        if str((claims.get(str(event.get("event_id")), {}) or {}).get("status", "unassessed"))
        in {"unassessed", "challenged"}
    ]
    if unresolved and remaining + 1e-9 >= _float(TOOL_COSTS["SourceChallenge"]):
        return packet_for(
            "SourceChallenge",
            {"event_id": str(unresolved[0]["event_id"])},
            "unresolved public claim",
        )

    sources = (
        observation.get("defender_state", {})
        .get("trust", {})
        .get("source_reputation", {})
        or {}
    )
    uncertain = [
        event
        for event in events
        if (
            _float((sources.get(str(event.get("source_id", "")), {}) or {}).get("mean", 0.5), 0.5) < 0.55
            or _float(
                (sources.get(str(event.get("source_id", "")), {}) or {}).get("uncertainty", 1.0),
                1.0,
            )
            > 0.30
        )
    ]
    if uncertain and remaining + 1e-9 >= _float(TOOL_COSTS["CanaryProbe"]):
        return packet_for(
            "CanaryProbe",
            {"event_id": str(uncertain[0]["event_id"])},
            "uncertain telemetry source",
        )

    public_assets = {
        str(asset.get("id", "")): asset
        for asset in observation.get("defense_context", {}).get("public_assets", []) or []
        if isinstance(asset, dict) and str(asset.get("id", ""))
    }
    high_impact = next(
        (
            item.packet.get("response", {})
            for item in scored
            if item.parse_ok
            and str(item.packet.get("response", {}).get("action", "")) in HIGH_IMPACT_ACTIONS
            and str(item.packet.get("response", {}).get("target", "")) in public_assets
        ),
        None,
    )
    if high_impact and remaining + 1e-9 >= _float(TOOL_COSTS["ShadowActionProbe"]):
        return packet_for(
            "ShadowActionProbe",
            {"action": copy.deepcopy(high_impact)},
            "high-impact consequence uncertain",
        )

    if events and public_assets and remaining + 1e-9 >= _float(TOOL_COSTS["DecoyProbe"]):
        entity = str(events[0].get("entity_id", ""))
        asset = public_assets.get(entity) or next(iter(public_assets.values()))
        return packet_for(
            "DecoyProbe",
            {"zone": str(asset.get("zone", "unknown"))},
            "activity requires discrimination",
        )

    packet = copy.deepcopy(DEFAULT_ACTION_PACKET_V4)
    return packet, {
        "fallback": "safe_observe",
        "fallback_type": "Observe",
        "fallback_reason": "no legal probe or budget",
    }


def select_v5c(public_context: dict[str, Any], candidates: Iterable[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a candidate using only public observation and defender state."""

    assert_public(public_context)
    scored = [
        score_v5c_candidate(public_context, text, index=index)
        for index, text in enumerate(candidates)
    ]
    if not scored:
        packet, fallback = safe_probe_fallback(public_context)
        return packet, {"selected_index": -1, "candidates": [], **fallback}
    admissible = [item for item in scored if item.parse_ok and item.admissible]
    if not admissible:
        packet, fallback = safe_probe_fallback(public_context, scored)
        diagnostics = {
            "selector": "v5_c_evidence_constrained_runtime_governor",
            "selected_index": -1,
            "candidates": [
                {
                    "index": item.index,
                    "parse_ok": item.parse_ok,
                    "admissible": item.admissible,
                    "hard_violations": list(item.hard_violations),
                }
                for item in scored
            ],
            **fallback,
        }
        assert_public(diagnostics)
        return packet, diagnostics
    selected = max(admissible, key=lambda item: (item.worst_case_utility, item.score, -item.index))
    diagnostics = {
        "selector": "v5_c_evidence_constrained_runtime_governor",
        "selected_index": selected.index,
        "selected_score": selected.score,
        "selected_worst_case_utility": selected.worst_case_utility,
        "candidates": [
            {
                "index": item.index,
                "parse_ok": item.parse_ok,
                "admissible": item.admissible,
                "hard_violations": list(item.hard_violations),
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
