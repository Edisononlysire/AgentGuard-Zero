from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from agentguard_zero.defender_state.evidence_signals import evidence_is_positive
from agentguard_zero.runtime_policy import HIGH_IMPACT_ACTIONS, TARGETED_RESPONSE_ACTIONS
from agentguard_zero.tools.business_impact import estimate_business_impact


CONTAINMENT_ACTIONS = frozenset({"Isolate", "Remove"})
CONTAINMENT_PREDICATES = frozenset(
    {
        "activity_type",
        "anomaly_flag",
        "asset_risk_level",
        "attack_objective",
        "attack_phase",
        "attack_stage",
        "attack_status",
        "auth_status",
        "authentication_failure",
        "compromise_state",
        "credential_state",
        "exfiltration_state",
        "integrity_status",
        "intrusion_state",
        "lateral_movement",
        "malicious_activity",
        "objective",
        "persistence_state",
        "risk",
        "risk_level",
        "threat_level",
        "threat_state",
    }
)
THREAT_POSITIVE_OBJECTS = frozenset(
    {
        "active",
        "anomalous",
        "anomaly",
        "breach",
        "breached",
        "collection",
        "compromised",
        "confirmed",
        "credential_theft",
        "critical",
        "detected",
        "discovery",
        "elevated",
        "exfiltration",
        "extraction",
        "failed",
        "failure",
        "high",
        "impact",
        "initial_access",
        "lateral_movement",
        "malicious",
        "persistence",
        "sabotage",
        "suspicious",
        "unauthorized",
    }
)
BENIGN_OR_LOW_RISK_OBJECTS = frozenset(
    {
        "benign",
        "clean",
        "healthy",
        "inactive",
        "legitimate_change",
        "low",
        "maintenance",
        "no_attack",
        "normal",
        "recovered",
        "safe",
    }
)
RECOVERY_PREDICATES = frozenset(
    {
        "asset_state",
        "compromise_state",
        "integrity_status",
        "recovery_state",
        "service_state",
        "system_status",
    }
)
RECOVERY_POSITIVE_OBJECTS = frozenset(
    {
        "available",
        "benign",
        "clean",
        "healthy",
        "normal",
        "recovered",
        "remediated",
        "restored",
        "safe",
        "stable",
    }
)


@dataclass(frozen=True)
class ResponseAuthorization:
    allowed: bool
    reason: str
    evidence_ids: tuple[str, ...] = ()
    estimated_cost: float = 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _asset_map(public_assets: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(asset.get("id", "")): asset
        for asset in public_assets or []
        if isinstance(asset, dict) and str(asset.get("id", "")).strip()
    }


def _evidence_index(
    evidence_snapshot: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_event: dict[str, list[dict[str, Any]]] = {}
    for record in evidence_snapshot or []:
        if not isinstance(record, dict):
            continue
        evidence_id = str(record.get("evidence_id", ""))
        event_id = str(record.get("event_id", ""))
        if evidence_id:
            by_id[evidence_id] = record
        if event_id:
            by_event.setdefault(event_id, []).append(record)
    return by_id, by_event


def _claim_semantics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        content = record.get("content", {}) or {}
        semantics = content.get("claim_semantics", {}) if isinstance(content, dict) else {}
        if isinstance(semantics, dict) and semantics:
            return semantics
    return {}


def _action_compatible(action: str, semantics: dict[str, Any]) -> bool:
    predicate = str(semantics.get("predicate", "")).strip().lower()
    object_value = str(semantics.get("object", "")).strip().lower()
    if not predicate or not object_value:
        return False
    if action in CONTAINMENT_ACTIONS:
        return (
            predicate in CONTAINMENT_PREDICATES
            and object_value in THREAT_POSITIVE_OBJECTS
            and object_value not in BENIGN_OR_LOW_RISK_OBJECTS
        )
    if action == "Restore":
        return (
            predicate in RECOVERY_PREDICATES
            and object_value in RECOVERY_POSITIVE_OBJECTS
        )
    return False


def _graph_link_evidence(
    *,
    entity_id: str,
    target: str,
    evidence_by_id: dict[str, dict[str, Any]],
) -> str | None:
    for evidence_id, record in evidence_by_id.items():
        if str(record.get("evidence_type", "")).lower() != "graphquery_result":
            continue
        content = record.get("content", {}) or {}
        if not isinstance(content, dict):
            continue
        node = str(content.get("node", ""))
        edges = content.get("edges", []) or []
        if node == target and (not entity_id or entity_id == target):
            return evidence_id
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            endpoints = {str(edge[0]), str(edge[1])}
            if {entity_id, target}.issubset(endpoints):
                return evidence_id
    return None


def authorize_public_response(
    *,
    response: dict[str, Any],
    evidence_assessment: Iterable[dict[str, Any]],
    trust_snapshot: dict[str, Any],
    evidence_snapshot: Iterable[dict[str, Any]],
    public_assets: Iterable[dict[str, Any]],
    business_state: dict[str, Any],
    oracle_override: bool = False,
) -> ResponseAuthorization:
    """Authorize a response using public pre-state only.

    Environment execution and V5-C dry-runs call this same predicate so a
    candidate cannot pass one gate and fail the other under an identical
    public state.
    """

    response = response if isinstance(response, dict) else {}
    action = str(response.get("action", "Observe"))
    target = str(response.get("target", "")).strip()
    assets = _asset_map(public_assets)
    criticality = _float(assets.get(target, {}).get("criticality", 0.5), 0.5)
    estimated_cost = _float(
        estimate_business_impact(response, criticality).get("estimated_cost", 0.0)
    )

    if action in TARGETED_RESPONSE_ACTIONS and (not target or target not in assets):
        return ResponseAuthorization(False, "invalid_or_missing_public_target", estimated_cost=estimated_cost)

    remaining_business = _float(
        business_state.get("remaining_business_budget", float("inf")),
        float("inf"),
    )
    if estimated_cost > remaining_business + 1e-9:
        return ResponseAuthorization(False, "business_budget_exceeded", estimated_cost=estimated_cost)

    if action not in HIGH_IMPACT_ACTIONS:
        return ResponseAuthorization(True, "ok", estimated_cost=estimated_cost)

    remaining_high = business_state.get("remaining_high_impact_actions")
    if remaining_high is not None and int(remaining_high) <= 0:
        return ResponseAuthorization(False, "high_impact_budget_exhausted", estimated_cost=estimated_cost)
    if oracle_override:
        return ResponseAuthorization(True, "ok", estimated_cost=estimated_cost)

    assessed_ids = {
        str(item.get("event_id", ""))
        for item in evidence_assessment or []
        if isinstance(item, dict) and str(item.get("event_id", ""))
    }
    claims = trust_snapshot.get("current_claim_trust", {}) or {}
    evidence_by_id, evidence_by_event = _evidence_index(evidence_snapshot)

    for event_id in sorted(assessed_ids):
        claim = claims.get(event_id, {})
        if not isinstance(claim, dict):
            continue
        if str(claim.get("status", "")) != "supported" or _float(claim.get("score", 0.0)) < 0.80:
            continue
        records = evidence_by_event.get(event_id, [])
        semantics = _claim_semantics(records)
        if not _action_compatible(action, semantics):
            continue
        entity_id = str(semantics.get("entity_id", "")).strip()
        supporting_ids = {
            str(item)
            for item in claim.get("support_evidence_refs", []) or []
            if str(item) in evidence_by_id
            and evidence_is_positive(evidence_by_id[str(item)])
        }
        if not supporting_ids:
            continue
        if entity_id == target:
            return ResponseAuthorization(
                True,
                "ok",
                tuple(sorted(supporting_ids)),
                estimated_cost,
            )
        graph_evidence = _graph_link_evidence(
            entity_id=entity_id,
            target=target,
            evidence_by_id=evidence_by_id,
        )
        if graph_evidence:
            supporting_ids.add(graph_evidence)
            return ResponseAuthorization(
                True,
                "ok",
                tuple(sorted(supporting_ids)),
                estimated_cost,
            )

    return ResponseAuthorization(
        False,
        "high_impact_requires_target_relevant_supported_public_evidence",
        estimated_cost=estimated_cost,
    )
