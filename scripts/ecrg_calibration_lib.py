#!/usr/bin/env python3
"""Public-state feature and fixed-search helpers for ECRG calibration."""

from __future__ import annotations

import copy
import itertools
from typing import Any, Iterable

from agentguard_zero.governance.v5c import (
    ACTIVE_PROBES,
    PASSIVE_VERIFICATION,
    REVERSIBLE_RESPONSES,
    V5CCandidate,
    score_v5c_candidate,
)
from agentguard_zero.runtime_policy import HIGH_IMPACT_ACTIONS
from agentguard_zero.tools.business_impact import estimate_business_impact


ADMISSION_PROFILES: dict[str, dict[str, float]] = {
    "runtime_compatible": {
        "min_supported_score": 0.80,
        "max_high_impact_risk": 1.60,
        "max_business_fraction": 1.00,
        "max_high_impact_uncertainty": 1.00,
    },
    "balanced": {
        "min_supported_score": 0.85,
        "max_high_impact_risk": 1.20,
        "max_business_fraction": 0.75,
        "max_high_impact_uncertainty": 0.60,
    },
    "evidence_strict": {
        "min_supported_score": 0.90,
        "max_high_impact_risk": 0.90,
        "max_business_fraction": 0.50,
        "max_high_impact_uncertainty": 0.45,
    },
}


RANKING_PROFILES: dict[str, dict[str, float]] = {
    "reference": {
        "legal": 8.0,
        "evidence": 1.0,
        "verification": 4.0,
        "active_probe": 2.0,
        "quarantine": 2.0,
        "promotion_risk": 3.0,
        "mitigation": 1.5,
        "business": 2.0,
        "overresponse": 3.0,
        "delay": 0.4,
        "betrayal": 2.5,
        "probe_robust": 1.5,
        "poisoning": 2.5,
        "quarantine_robust": 1.0,
        "benign_high_impact": 1.5,
        "reversible": 0.5,
    },
    "evidence_first": {
        "legal": 8.0,
        "evidence": 2.0,
        "verification": 5.0,
        "active_probe": 2.5,
        "quarantine": 2.0,
        "promotion_risk": 3.5,
        "mitigation": 1.25,
        "business": 2.0,
        "overresponse": 3.5,
        "delay": 0.3,
        "betrayal": 3.0,
        "probe_robust": 2.0,
        "poisoning": 3.0,
        "quarantine_robust": 1.25,
        "benign_high_impact": 1.75,
        "reversible": 0.75,
    },
    "business_cautious": {
        "legal": 8.0,
        "evidence": 1.0,
        "verification": 3.5,
        "active_probe": 2.0,
        "quarantine": 2.0,
        "promotion_risk": 3.0,
        "mitigation": 1.0,
        "business": 3.5,
        "overresponse": 4.0,
        "delay": 0.35,
        "betrayal": 2.5,
        "probe_robust": 1.5,
        "poisoning": 2.5,
        "quarantine_robust": 1.0,
        "benign_high_impact": 2.5,
        "reversible": 1.0,
    },
    "poison_robust": {
        "legal": 8.0,
        "evidence": 1.5,
        "verification": 4.0,
        "active_probe": 2.0,
        "quarantine": 3.0,
        "promotion_risk": 5.0,
        "mitigation": 1.25,
        "business": 2.0,
        "overresponse": 3.0,
        "delay": 0.4,
        "betrayal": 2.5,
        "probe_robust": 1.5,
        "poisoning": 4.0,
        "quarantine_robust": 2.0,
        "benign_high_impact": 1.5,
        "reversible": 0.5,
    },
    "mitigation_balanced": {
        "legal": 8.0,
        "evidence": 1.0,
        "verification": 3.0,
        "active_probe": 1.5,
        "quarantine": 2.0,
        "promotion_risk": 3.0,
        "mitigation": 2.5,
        "business": 2.25,
        "overresponse": 3.25,
        "delay": 0.6,
        "betrayal": 2.5,
        "probe_robust": 1.25,
        "poisoning": 2.5,
        "quarantine_robust": 1.0,
        "benign_high_impact": 1.75,
        "reversible": 0.5,
    },
}


FALLBACK_TRIGGER_PROFILES: dict[str, dict[str, float]] = {
    "empty_only": {"score_floor": -1.0e9, "risk_floor": 1.0e9},
    "cautious": {"score_floor": 2.0, "risk_floor": 0.80},
    "safety_first": {"score_floor": 4.0, "risk_floor": 0.50},
}


def search_space() -> list[dict[str, str]]:
    return [
        {
            "admission_profile": admission,
            "ranking_profile": ranking,
            "fallback_trigger_profile": fallback,
        }
        for admission, ranking, fallback in itertools.product(
            sorted(ADMISSION_PROFILES),
            sorted(RANKING_PROFILES),
            sorted(FALLBACK_TRIGGER_PROFILES),
        )
    ]


def resolve_config(profile_names: dict[str, str]) -> dict[str, Any]:
    admission_name = profile_names["admission_profile"]
    ranking_name = profile_names["ranking_profile"]
    fallback_name = profile_names["fallback_trigger_profile"]
    return {
        "profiles": copy.deepcopy(profile_names),
        "hard_admission_thresholds": copy.deepcopy(ADMISSION_PROFILES[admission_name]),
        "ranking_weights": copy.deepcopy(RANKING_PROFILES[ranking_name]),
        "safe_probe_fallback_trigger": copy.deepcopy(
            FALLBACK_TRIGGER_PROFILES[fallback_name]
        ),
    }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _observation(public_context: dict[str, Any]) -> dict[str, Any]:
    value = public_context.get("observation", public_context)
    return value if isinstance(value, dict) else {}


def candidate_features(
    public_context: dict[str, Any],
    text: str,
    *,
    index: int,
) -> tuple[dict[str, Any], V5CCandidate]:
    """Extract public-only features. Hidden scenario state is never accepted."""

    scored = score_v5c_candidate(public_context, text, index=index)
    observation = _observation(public_context)
    packet = scored.packet
    diagnostics = scored.diagnostics
    trust = observation.get("defender_state", {}).get("trust", {}) or {}
    source_states = trust.get("source_reputation", {}) or {}
    uncertainty = max(
        (
            _float(value.get("uncertainty", 1.0), 1.0)
            for value in source_states.values()
            if isinstance(value, dict)
        ),
        default=1.0,
    )
    claims = trust.get("current_claim_trust", {}) or {}
    assessed_ids = {
        str(item.get("event_id", ""))
        for item in packet.get("evidence_assessment", []) or []
        if isinstance(item, dict) and str(item.get("event_id", ""))
    }
    supported_score = max(
        (
            _float(claims.get(event_id, {}).get("score", 0.0))
            for event_id in assessed_ids
            if str((claims.get(event_id, {}) or {}).get("status", "")) == "supported"
        ),
        default=0.0,
    )
    tool = str((packet.get("tool_call", {}) or {}).get("name", "None"))
    response = packet.get("response", {}) or {}
    action = str(response.get("action", "Observe"))
    target = str(response.get("target", ""))
    asset = next(
        (
            item
            for item in observation.get("defense_context", {}).get("public_assets", []) or []
            if isinstance(item, dict) and str(item.get("id", "")) == target
        ),
        {},
    )
    estimated_business = max(
        0.0,
        _float(
            estimate_business_impact(
                response,
                _float(asset.get("criticality", 0.5), 0.5),
            ).get("estimated_cost", 0.0)
        ),
    )
    remaining_business = max(
        0.0,
        _float(
            observation.get("defense_context", {}).get(
                "remaining_business_budget", 0.0
            )
        ),
    )
    business_fraction = (
        estimated_business / remaining_business
        if remaining_business > 1.0e-9
        else (0.0 if estimated_business <= 1.0e-9 else 10.0)
    )
    memory_ops = packet.get("memory_operations", []) or []
    direct_promotions = sum(
        int(item.get("op") == "promote")
        for item in memory_ops
        if isinstance(item, dict)
    )
    quarantining = sum(
        int(item.get("op") in {"ingest", "demote", "reject"})
        for item in memory_ops
        if isinstance(item, dict)
    )
    source_risk = _float(diagnostics.get("source_risk", 0.5), 0.5)
    poison_risk = _float(diagnostics.get("poison_risk", 0.0), 0.0)
    verification = tool in ACTIVE_PROBES | PASSIVE_VERIFICATION
    active_probe = tool in ACTIVE_PROBES
    high_impact = action in HIGH_IMPACT_ACTIONS
    features = {
        "public_only": True,
        "parse_ok": bool(scored.parse_ok),
        "base_admissible": bool(scored.admissible),
        "hard_violations": list(scored.hard_violations),
        "reference_count": int(diagnostics.get("reference_count", 0)),
        "supported_score": supported_score,
        "source_risk": source_risk,
        "poison_risk": poison_risk,
        "uncertainty": uncertainty,
        "risk_signal": source_risk + poison_risk,
        "verification": float(verification),
        "active_probe": float(active_probe),
        "direct_promotions": float(direct_promotions),
        "quarantining": float(quarantining),
        "mitigation": float(
            action in {"DeployDecoy", "ShadowBlock", "LimitSession", "Isolate"}
        ),
        "high_impact": float(high_impact),
        "reversible": float(action in REVERSIBLE_RESPONSES),
        "observe_without_verification": float(action == "Observe" and not verification),
        "estimated_business_cost": estimated_business,
        "remaining_business_budget": remaining_business,
        "business_fraction": business_fraction,
        "business_penalty": estimated_business
        + max(0.0, estimated_business - remaining_business),
        "action": action,
        "tool": tool,
    }
    return features, scored


def ranking_score(features: dict[str, Any], weights: dict[str, float]) -> tuple[float, float]:
    risk = _float(features.get("risk_signal"))
    source_risk = _float(features.get("source_risk"))
    poison_risk = _float(features.get("poison_risk"))
    uncertainty = _float(features.get("uncertainty"), 1.0)
    high_impact = _float(features.get("high_impact"))
    active_probe = _float(features.get("active_probe"))
    direct_promotions = _float(features.get("direct_promotions"))
    quarantining = _float(features.get("quarantining"))
    score = (
        weights["legal"] * float(bool(features.get("base_admissible")))
        + weights["evidence"] * min(3.0, _float(features.get("reference_count"))) / 3.0
        + weights["verification"] * _float(features.get("verification"))
        + weights["active_probe"] * active_probe * risk
        + weights["quarantine"] * quarantining
        - weights["promotion_risk"]
        * direct_promotions
        * max(poison_risk, uncertainty)
        + weights["mitigation"] * _float(features.get("mitigation"))
        - weights["business"] * _float(features.get("business_penalty"))
        - weights["overresponse"] * high_impact * risk
        - weights["delay"] * _float(features.get("observe_without_verification"))
    )
    betrayal = (
        score
        - weights["betrayal"] * high_impact * source_risk
        + weights["probe_robust"] * active_probe
    )
    poisoning = (
        score
        - weights["poisoning"] * direct_promotions * poison_risk
        + weights["quarantine_robust"] * quarantining
    )
    benign = (
        score
        - weights["benign_high_impact"] * high_impact
        + weights["reversible"] * _float(features.get("reversible"))
    )
    return float(score), float(min(betrayal, poisoning, benign))


def admission_allowed(features: dict[str, Any], thresholds: dict[str, float]) -> bool:
    if not features.get("parse_ok") or not features.get("base_admissible"):
        return False
    if not bool(features.get("high_impact")):
        return True
    return bool(
        _float(features.get("supported_score"))
        >= thresholds["min_supported_score"]
        and _float(features.get("risk_signal"))
        <= thresholds["max_high_impact_risk"]
        and _float(features.get("business_fraction"))
        <= thresholds["max_business_fraction"]
        and _float(features.get("uncertainty"), 1.0)
        <= thresholds["max_high_impact_uncertainty"]
    )


def select_trace_decision(decision: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["hard_admission_thresholds"]
    weights = config["ranking_weights"]
    fallback_trigger = config["safe_probe_fallback_trigger"]
    ranked = []
    for candidate in decision.get("candidates", []) or []:
        features = candidate["features"]
        if not admission_allowed(features, thresholds):
            continue
        score, worst = ranking_score(features, weights)
        ranked.append((worst, score, -int(candidate.get("index", 0)), candidate))
    fallback = decision["fallback"]
    if not ranked:
        return {**fallback, "selection_reason": "empty_admissible_set"}
    ranked.sort(reverse=True, key=lambda item: item[:3])
    worst, score, _, selected = ranked[0]
    public_risk = max(
        (_float(item["features"].get("risk_signal")) for item in decision.get("candidates", [])),
        default=0.0,
    )
    if (
        worst < fallback_trigger["score_floor"]
        and public_risk >= fallback_trigger["risk_floor"]
    ):
        return {**fallback, "selection_reason": "calibrated_safe_probe_trigger"}
    return {
        **selected,
        "calibrated_score": score,
        "calibrated_worst_case_utility": worst,
        "selection_reason": "ranked_admissible_candidate",
    }


def immutable_safety_invariants() -> list[str]:
    return [
        "strict_action_schema",
        "known_and_available_evidence_references",
        "known_public_event_and_memory_references",
        "tool_and_verification_budget",
        "public_target_authorization",
        "business_budget",
        "high_impact_action_budget",
        "target_relevant_supported_public_evidence_for_high_impact_actions",
    ]
