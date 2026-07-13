#!/usr/bin/env python3
"""Evaluate AgentGuard-Zero frozen-model policies on Level-1 scenarios.

This script is intentionally training-free. It runs the same Level-1 symbolic
environment and trajectory reward used by VDA training, but freezes the base
model and optionally applies AgentGuard-Zero-Select candidate selection.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from agentguard_zero.schemas.action_schema import ALLOWED_ACTIONS, ALLOWED_TOOLS, OBJECTIVE_KEYS, parse_action_json
from level1_rollout_server import Level1RolloutStore


PASSIVE_VERIFY_TOOLS = {"CrossCheck", "ProvenanceCheck", "GraphQuery", "BusinessImpactEstimator"}
ACTIVE_PROBE_TOOLS = {"SourceChallenge", "CanaryProbe", "DecoyProbe", "ShadowActionProbe", "DeployDecoy", "ShadowResponse"}
ACTIVE_PROBE_ACTIONS = {"SourceChallenge", "CanaryProbe", "DecoyProbe", "ShadowActionProbe"}
VERIFY_TOOLS = PASSIVE_VERIFY_TOOLS | ACTIVE_PROBE_TOOLS
HIGH_IMPACT_ACTIONS = {"Isolate", "Restore", "Remove"}
LOW_IMPACT_MITIGATION = {"DeployDecoy", "ShadowBlock", "LimitSession"}
V5_SELECTOR_MODES = {"v5_a_constrained", "v5_b_belief_q", "v5_c_frontier_minimax"}
ADVANCED_SELECTOR_MODES = {"mitigation_v2", "mitigation_v3", "mitigation_v4"} | V5_SELECTOR_MODES
HIDDEN_KEYS = {
    "oracle",
    "true_attack",
    "true_objective",
    "attack_path",
    "is_fake",
    "is_true",
    "ground_truth",
    "hidden",
    "hidden_state",
    "compromised_assets",
}


@dataclasses.dataclass
class Candidate:
    text: str
    packet: dict[str, Any]
    ok: bool
    parse_msg: str
    selector_score: float
    diagnostics: dict[str, Any]


def json_dumps(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, indent=indent, default=str)


def maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return json.loads(stripped)
        except Exception:
            return value
    return value


def as_dict(value: Any) -> dict[str, Any]:
    loaded = maybe_json(value)
    return loaded if isinstance(loaded, dict) else {}


def as_messages(value: Any) -> list[dict[str, str]]:
    loaded = maybe_json(value)
    if hasattr(loaded, "tolist"):
        loaded = loaded.tolist()
    if isinstance(loaded, list):
        messages: list[dict[str, str]] = []
        for item in loaded:
            item = as_dict(item)
            role = str(item.get("role", "user"))
            content = item.get("content", "")
            messages.append({"role": role, "content": str(content)})
        if messages:
            return messages
    if isinstance(loaded, dict) and "content" in loaded:
        return [{"role": str(loaded.get("role", "user")), "content": str(loaded.get("content", ""))}]
    return [{"role": "user", "content": str(value)}]


def sanitize_public_value(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, child in value.items():
            key_str = str(key)
            lowered = key_str.lower()
            if lowered in HIDDEN_KEYS or lowered.startswith("_hidden"):
                continue
            clean[key_str] = sanitize_public_value(child)
        return clean
    if isinstance(value, list):
        return [sanitize_public_value(item) for item in value]
    return value


def extract_event_ids(value: Any) -> list[str]:
    ids: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            event_id = node.get("event_id") or node.get("id")
            if event_id is not None:
                ids.append(str(event_id))
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return ids


def flatten_public_text(value: Any, max_chars: int = 16000) -> str:
    text = json_dumps(sanitize_public_value(value))
    return text[:max_chars]


def public_signals(public_context: Any) -> dict[str, Any]:
    clean = sanitize_public_value(public_context)
    text = flatten_public_text(clean).lower()
    signals = {
        "has_conflict": any(word in text for word in ("conflict", "contradict", "inconsistent", "mismatch")),
        "weak_source": any(word in text for word in ("low reliability", "weak source", "untrusted", "spoof", "suspicious")),
        "critical_asset": any(word in text for word in ("critical", "high impact", "business impact", "database", "payment")),
        "verified_evidence": any(word in text for word in ("cross_checked", "confirmed", "provenance_ok", "verified")),
        "high_public_risk": any(word in text for word in ("exfiltration", "sabotage", "credential", "persistence", "lateral")),
        "event_ids": extract_event_ids(clean),
        "turn": 0.0,
        "verification_cost": 0.0,
        "business_cost": 0.0,
        "remaining_budget": 0.0,
        "verification_budget": 0.0,
        "high_impact_count": 0.0,
    }

    weak_numeric = 0
    high_spoof = 0
    numeric_values: dict[str, list[float]] = {
        "time": [],
        "verification_cost": [],
        "business_cost": [],
        "remaining_budget": [],
        "remaining_business_budget": [],
        "business_budget": [],
        "verification_budget": [],
        "high_impact_count": [],
    }

    def visit(node: Any) -> None:
        nonlocal weak_numeric, high_spoof
        if isinstance(node, dict):
            for key, value in node.items():
                key_str = str(key)
                if key_str in numeric_values:
                    parsed = safe_float(value, math.nan)
                    if not math.isnan(parsed):
                        numeric_values[key_str].append(parsed)
            for key in ("source_reliability", "reliability", "confidence"):
                if key in node:
                    try:
                        if float(node[key]) < 0.6:
                            weak_numeric += 1
                    except Exception:
                        pass
            for key in ("spoofability", "poisoning_risk", "deception_risk"):
                if key in node:
                    try:
                        if float(node[key]) > 0.55:
                            high_spoof += 1
                    except Exception:
                        pass
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(clean)
    signals["weak_source"] = bool(signals["weak_source"] or weak_numeric > 0 or high_spoof > 0)
    signals["suspicious"] = bool(signals["has_conflict"] or signals["weak_source"])
    signals["weak_numeric_count"] = weak_numeric
    signals["high_spoof_count"] = high_spoof
    signals["turn"] = max(numeric_values["time"] or [0.0])
    signals["verification_cost"] = max(numeric_values["verification_cost"] or [0.0])
    signals["business_cost"] = max(numeric_values["business_cost"] or [0.0])
    signals["remaining_budget"] = max(
        numeric_values["remaining_budget"]
        or numeric_values["remaining_business_budget"]
        or numeric_values["business_budget"]
        or [0.0]
    )
    signals["verification_budget"] = max(numeric_values["verification_budget"] or [0.0])
    signals["high_impact_count"] = max(numeric_values["high_impact_count"] or [0.0])
    return signals


def top_belief(packet: dict[str, Any]) -> str:
    belief = packet.get("belief", {}) if isinstance(packet, dict) else {}
    if not isinstance(belief, dict) or not belief:
        return "unknown"
    values: dict[str, float] = {}
    for key in OBJECTIVE_KEYS:
        try:
            values[key] = float(belief.get(key, 0.0))
        except Exception:
            values[key] = 0.0
    return max(values, key=values.get)


def belief_confidence(packet: dict[str, Any]) -> float:
    belief = packet.get("belief", {}) if isinstance(packet, dict) else {}
    if not isinstance(belief, dict) or not belief:
        return 0.0
    return max(safe_float(belief.get(key, 0.0), 0.0) for key in OBJECTIVE_KEYS)


def belief_margin(packet: dict[str, Any]) -> float:
    belief = packet.get("belief", {}) if isinstance(packet, dict) else {}
    if not isinstance(belief, dict) or not belief:
        return 0.0
    values = sorted((safe_float(belief.get(key, 0.0), 0.0) for key in OBJECTIVE_KEYS), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return max(0.0, values[0] - values[1])


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def has_verified_evidence(packet: dict[str, Any]) -> bool:
    assessments = packet.get("evidence_assessment", [])
    if not isinstance(assessments, list):
        return False
    verified_status = {"cross_checked", "confirmed", "rejected"}
    for item in assessments:
        if isinstance(item, dict) and item.get("status") in verified_status:
            return True
    return False


def action_like_active_probe(tool: str, action: str) -> bool:
    return tool in ACTIVE_PROBE_TOOLS or action in ACTIVE_PROBE_ACTIONS


def action_like_verification(tool: str, action: str) -> bool:
    return tool in VERIFY_TOOLS or action in {"CrossCheck", "QueryProvenance"} or action_like_active_probe(tool, action)


def action_like_low_mitigation(action: str) -> bool:
    return action in LOW_IMPACT_MITIGATION


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def public_risk_level(signals: dict[str, Any]) -> float:
    risk = 0.0
    risk += 0.35 if signals.get("high_public_risk") else 0.0
    risk += 0.25 if signals.get("suspicious") else 0.0
    risk += 0.15 if signals.get("critical_asset") else 0.0
    risk += min(0.15, 0.04 * safe_float(signals.get("high_spoof_count", 0.0), 0.0))
    risk += min(0.10, 0.03 * safe_float(signals.get("weak_numeric_count", 0.0), 0.0))
    return clamp(risk)


def public_belief_alignment(public_context: Any, packet: dict[str, Any]) -> float:
    public_prior = public_belief_guess(public_context)
    public_top = max(public_prior, key=public_prior.get)
    candidate_top = top_belief(packet)
    if candidate_top == "unknown":
        return 0.0
    if candidate_top == public_top:
        return 1.0
    return 0.35


def apply_v5_adjustment(
    score: float,
    public_context: Any,
    packet: dict[str, Any],
    selector_mode: str,
    signals: dict[str, Any],
    diag: dict[str, Any],
) -> float:
    if selector_mode not in V5_SELECTOR_MODES:
        return score

    tool = str(diag.get("tool", "None"))
    action = str(diag.get("response_action", "Observe"))
    confidence = safe_float(diag.get("belief_confidence", 0.0), 0.0)
    margin = belief_margin(packet)
    uncertainty = safe_float(diag.get("uncertainty", 1.0), 1.0)
    turn = safe_float(diag.get("turn", 0.0), 0.0)
    verification_spent = safe_float(diag.get("verification_spent", 0.0), 0.0)
    business_spent = safe_float(signals.get("business_cost", 0.0), 0.0)
    remaining_budget = safe_float(signals.get("remaining_budget", 0.0), 0.0)
    risk = public_risk_level(signals)
    verified = bool(diag.get("verified_or_crosschecked"))
    verification_like = bool(diag.get("verification_like"))
    active_probe_like = bool(diag.get("active_probe_like"))
    mitigation_like = bool(diag.get("mitigation_like"))
    high_impact = action in HIGH_IMPACT_ACTIONS
    memory = packet.get("memory_update", {}) if isinstance(packet, dict) else {}
    quarantine_count = list_len(memory.get("profile_quarantine")) if isinstance(memory, dict) else 0
    rejected_count = list_len(memory.get("rejected_profile")) if isinstance(memory, dict) else 0
    confirmed_count = list_len(memory.get("confirmed_profile")) if isinstance(memory, dict) else 0
    safety = packet.get("safety_check", {}) if isinstance(packet, dict) else {}
    business_risk = safe_float(safety.get("business_risk", 0.5) if isinstance(safety, dict) else 0.5, 0.5)
    overresponse_risk = safe_float(safety.get("overresponse_risk", 0.5) if isinstance(safety, dict) else 0.5, 0.5)
    ready_soft = bool(
        risk >= 0.35
        and (
            verified
            or verification_spent >= 1.0
            or (confidence >= 0.62 and uncertainty <= 0.65)
            or turn >= 2.0
        )
    )
    budget_tight = bool(remaining_budget > 0 and (business_spent / max(remaining_budget + business_spent, 1.0)) >= 0.55)
    repeated_verification = bool(verification_like and verification_spent >= 1.0 and turn >= 1.0)

    poison_constraint = 0.0
    if confirmed_count and not verified:
        poison_constraint += 1.0
    if signals.get("suspicious") and not quarantine_count and not rejected_count and not mitigation_like:
        poison_constraint += 0.55

    cost_constraint = clamp(0.65 * business_risk + 0.75 * overresponse_risk)
    if high_impact and not verified:
        cost_constraint += 0.75
    if budget_tight and action not in {"Observe", "CrossCheck", "QueryProvenance", "LimitSession", "ShadowBlock", "DeployDecoy"}:
        cost_constraint += 0.45

    delay_constraint = 0.0
    if risk >= 0.45 and (action == "Observe" or repeated_verification) and ready_soft:
        delay_constraint += 1.0
    if risk >= 0.55 and turn >= 2.0 and not mitigation_like:
        delay_constraint += 0.75

    verification_value = 0.0
    if verification_like:
        if signals.get("suspicious") and verification_spent <= 0.0:
            verification_value += 5.0
        elif risk >= 0.5 and verification_spent < 1.5 and uncertainty > 0.5 and not verified:
            verification_value += 2.0
        else:
            verification_value -= 4.0
    if active_probe_like:
        if signals.get("suspicious") and uncertainty >= 0.45 and verification_spent <= 1.0:
            verification_value += 4.0
        if high_impact and not verified:
            verification_value += 3.0
        if turn >= 2.0 or verification_spent >= 2.0:
            verification_value -= 5.0

    mitigation_value = 0.0
    if mitigation_like:
        mitigation_value += 6.0 * risk
        if ready_soft:
            mitigation_value += 8.0
        if confidence >= 0.60:
            mitigation_value += 2.0
        if margin >= 0.30:
            mitigation_value += 1.5
        if action in {"LimitSession", "ShadowBlock"}:
            mitigation_value += 2.0
        if action == "DeployDecoy" and verified:
            mitigation_value += 1.0

    alignment = public_belief_alignment(public_context, packet)
    diag["v5_risk_level"] = risk
    diag["v5_belief_margin"] = margin
    diag["v5_ready_soft"] = ready_soft
    diag["v5_cost_constraint"] = cost_constraint
    diag["v5_poison_constraint"] = poison_constraint
    diag["v5_delay_constraint"] = delay_constraint
    diag["v5_public_alignment"] = alignment

    if selector_mode == "v5_a_constrained":
        # Lagrangian-style public constraints: cheap to evaluate, hard on unsafe shortcuts.
        lagrangian_penalty = 9.0 * cost_constraint + 8.0 * poison_constraint + 5.0 * delay_constraint
        adjustment = verification_value + mitigation_value + 2.0 * alignment - lagrangian_penalty
        if action == "Observe" and risk < 0.30 and business_risk < 0.45:
            adjustment += 1.0
        diag["v5_objective"] = "constrained_safe_utility"
        diag["v5_adjustment"] = adjustment
        return score + adjustment

    if selector_mode == "v5_b_belief_q":
        # Hand-initialized public belief-state value proxy. This is the API-search
        # version of the small Q selector that Qwen will later learn from rollout rewards.
        q_verify = verification_value - 1.2 * verification_spent
        q_mitigate = mitigation_value + 6.0 * confidence + 3.0 * margin - 4.0 * cost_constraint
        q_wait = 1.0 - 7.0 * risk - 4.0 * float(ready_soft)
        action_q = q_wait
        if verification_like:
            action_q = q_verify
        if mitigation_like:
            action_q = q_mitigate
        if high_impact:
            action_q = q_mitigate - 12.0 - 7.0 * float(not verified)
        adjustment = action_q + 2.5 * alignment - 5.0 * poison_constraint - 3.0 * delay_constraint
        if risk >= 0.45 and ready_soft and mitigation_like:
            adjustment += 4.0
        diag["v5_objective"] = "public_belief_q"
        diag["v5_action_q"] = action_q
        diag["v5_adjustment"] = adjustment
        return score + adjustment

    # Robust minimax selector: assume the most adversarial plausible observation
    # source unless public verification has reduced profile-poisoning risk.
    robust_penalty = 7.0 * cost_constraint + 10.0 * poison_constraint + 4.0 * delay_constraint
    robust_bonus = 0.0
    if quarantine_count or rejected_count:
        robust_bonus += 3.0
    if verified:
        robust_bonus += 2.5
    if mitigation_like and risk >= 0.40:
        robust_bonus += 7.0
    if mitigation_like and action in {"LimitSession", "ShadowBlock"}:
        robust_bonus += 3.0
    if verification_like and verification_spent <= 0.0 and signals.get("suspicious"):
        robust_bonus += 4.0
    if active_probe_like and signals.get("suspicious") and uncertainty >= 0.45:
        robust_bonus += 3.5
    if verification_like and verification_spent >= 1.0:
        robust_penalty += 5.0
    if active_probe_like and verification_spent <= 1.0:
        robust_penalty = max(0.0, robust_penalty - 1.0)
    if action == "Observe" and risk >= 0.45:
        robust_penalty += 5.0
    adjustment = robust_bonus - robust_penalty + 1.5 * alignment
    diag["v5_objective"] = "frontier_minimax"
    diag["v5_robust_bonus"] = robust_bonus
    diag["v5_adjustment"] = adjustment
    return score + adjustment


def score_candidate(public_context: Any, action_text: str, selector_mode: str = "mitigation_v2") -> Candidate:
    packet, ok, parse_msg = parse_action_json(action_text)
    signals = public_signals(public_context)
    score = 0.0
    diag: dict[str, Any] = {
        "parse_msg": parse_msg,
        "ok": bool(ok),
        "json_parse_failure": bool(str(parse_msg).startswith("json_parse_error")),
        "invalid_tool_call": bool(parse_msg == "invalid_tool"),
        "invalid_response_action": bool(parse_msg == "invalid_response_action"),
        "selector_signals": {k: v for k, v in signals.items() if k != "event_ids"},
    }

    if not ok:
        score -= 50.0
        if diag["json_parse_failure"]:
            score -= 30.0
        if diag["invalid_tool_call"] or diag["invalid_response_action"]:
            score -= 20.0
        return Candidate(action_text, packet, ok, parse_msg, score, diag)

    score += 12.0
    tool = packet.get("tool_call", {}).get("name", "None")
    response = packet.get("response", {})
    action = response.get("action", "Observe")
    diag["tool"] = tool
    diag["response_action"] = action
    diag["top_belief"] = top_belief(packet)
    diag["belief_confidence"] = belief_confidence(packet)

    for required_key in ("belief", "evidence_assessment", "trust_update", "uncertainty", "memory_update", "safety_check", "response"):
        score += 1.0 if required_key in packet else -2.0

    if tool not in ALLOWED_TOOLS:
        score -= 25.0
        diag["invalid_tool_call"] = True
    if action not in ALLOWED_ACTIONS:
        score -= 25.0
        diag["invalid_response_action"] = True

    uncertainty = safe_float(packet.get("uncertainty", 1.0), 1.0)
    diag["uncertainty"] = uncertainty
    if 0.15 <= uncertainty <= 0.85:
        score += 1.5
    else:
        score -= 2.0

    memory = packet.get("memory_update", {})
    quarantine = memory.get("profile_quarantine", []) if isinstance(memory, dict) else []
    confirmed = memory.get("confirmed_profile", []) if isinstance(memory, dict) else []
    rejected = memory.get("rejected_profile", []) if isinstance(memory, dict) else []
    turn = safe_float(signals.get("turn", 0.0), 0.0)
    verification_spent = safe_float(signals.get("verification_cost", 0.0), 0.0)
    verification_budget = safe_float(signals.get("verification_budget", 0.0), 0.0)
    verification_like = action_like_verification(str(tool), str(action))
    active_probe_like = action_like_active_probe(str(tool), str(action))
    mitigation_like = action_like_low_mitigation(str(action))
    late_in_rollout = turn >= 2.0 or verification_spent >= 2.0
    diag["turn"] = turn
    diag["verification_spent"] = verification_spent
    diag["verification_budget"] = verification_budget
    diag["verification_like"] = verification_like
    diag["active_probe_like"] = active_probe_like
    diag["mitigation_like"] = mitigation_like

    if selector_mode == "v1":
        if signals["suspicious"]:
            if verification_like:
                score += 7.0
            else:
                score -= 4.0
            if quarantine or rejected:
                score += 3.0
            if confirmed:
                score -= 4.0
        else:
            if tool in VERIFY_TOOLS:
                score += 1.0
    else:
        if signals["suspicious"]:
            if quarantine or rejected:
                score += 4.0
            if confirmed and not has_verified_evidence(packet):
                score -= 3.0
            if verification_like:
                if turn <= 0.0 and verification_spent <= 0.0:
                    score += 6.0
                elif verification_spent < 1.5 and uncertainty > 0.45 and not has_verified_evidence(packet):
                    score += 3.0
                else:
                    score -= 4.0
                if late_in_rollout:
                    score -= 7.0
            elif not mitigation_like:
                score -= 1.0
        elif verification_like and verification_spent <= 0.0:
            score += 1.0

    assessments = packet.get("evidence_assessment", [])
    if isinstance(assessments, list):
        suspected = 0
        verified = 0
        for item in assessments:
            if not isinstance(item, dict):
                continue
            if item.get("suspected_poisoning"):
                suspected += 1
            if item.get("status") in {"cross_checked", "rejected", "confirmed"}:
                verified += 1
        score += min(3.0, suspected * 1.0)
        score += min(3.0, verified * 0.75)
        diag["suspected_poisoning_mentions"] = suspected
        diag["verified_evidence_mentions"] = verified

    safety = packet.get("safety_check", {})
    business_risk = safe_float(safety.get("business_risk", 0.5) if isinstance(safety, dict) else 0.5, 0.5)
    overresponse_risk = safe_float(safety.get("overresponse_risk", 0.5) if isinstance(safety, dict) else 0.5, 0.5)
    diag["business_risk"] = business_risk
    diag["overresponse_risk"] = overresponse_risk

    verified = bool(signals["verified_evidence"] or has_verified_evidence(packet))
    confident = belief_confidence(packet) >= 0.55
    ready_to_mitigate = bool(
        signals["high_public_risk"]
        and (
            verified
            or verification_spent >= 1.0
            or uncertainty <= 0.45
            or confident
            or late_in_rollout
        )
    )
    diag["verified_or_crosschecked"] = verified
    diag["ready_to_mitigate"] = ready_to_mitigate
    if action in HIGH_IMPACT_ACTIONS:
        score -= 7.0
        if not verified or uncertainty > 0.35:
            score -= 6.0
        if business_risk > 0.35 or overresponse_risk > 0.35:
            score -= 4.0
    elif action in LOW_IMPACT_MITIGATION:
        score += 3.0 if signals["high_public_risk"] else 1.0
        if selector_mode in ADVANCED_SELECTOR_MODES:
            if ready_to_mitigate:
                score += 10.0
            if late_in_rollout:
                score += 5.0
            if action in {"LimitSession", "ShadowBlock"}:
                score += 3.0
            elif action == "DeployDecoy":
                score += 1.5
            if uncertainty > 0.70 and not verified and verification_spent <= 0.0:
                score -= 4.0
            if business_risk <= 0.55 and overresponse_risk <= 0.55:
                score += 1.5
    elif action == "Observe":
        score += 0.5
        if signals["high_public_risk"] and signals["suspicious"]:
            score -= 2.0
        if selector_mode in ADVANCED_SELECTOR_MODES:
            if ready_to_mitigate:
                score -= 8.0
            if late_in_rollout:
                score -= 4.0

    if tool == "BusinessImpactEstimator":
        score += 2.0 if signals["critical_asset"] else 0.5
        if selector_mode in ADVANCED_SELECTOR_MODES and late_in_rollout and not mitigation_like:
            score -= 7.0
    if active_probe_like:
        if selector_mode in V5_SELECTOR_MODES and signals["suspicious"] and verification_spent <= 1.0:
            score += 4.0
        elif selector_mode in ADVANCED_SELECTOR_MODES and signals["suspicious"]:
            score += 2.0
        if verification_spent >= 2.0 or turn >= 2.0:
            score -= 4.0
    if tool == "DeployDecoy" or action == "DeployDecoy":
        score += 1.5
    if business_risk < 0.5 and overresponse_risk < 0.5:
        score += 1.0

    if selector_mode in ADVANCED_SELECTOR_MODES and verification_like and not mitigation_like:
        if verification_spent >= 1.0 and turn >= 1.0:
            score -= 5.0
        if verification_spent >= 2.0 or turn >= 2.0:
            score -= 7.0

    score = apply_v5_adjustment(score, public_context, packet, selector_mode, signals, diag)
    diag["selector_score"] = score
    return Candidate(action_text, packet, ok, parse_msg, score, diag)


def partial_belief_from_text(text: str) -> dict[str, float]:
    belief: dict[str, float] = {}
    for key in OBJECTIVE_KEYS:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            belief[key] = min(1.0, max(0.0, safe_float(match.group(1), 0.0)))
    if sum(belief.values()) <= 0:
        return {}
    for key in OBJECTIVE_KEYS:
        belief.setdefault(key, 0.0)
    return belief


def public_belief_guess(public_context: Any) -> dict[str, float]:
    text = flatten_public_text(public_context).lower()
    counts = {key: text.count(key) for key in OBJECTIVE_KEYS}
    if sum(counts.values()) <= 0:
        return {key: 1.0 / len(OBJECTIVE_KEYS) for key in OBJECTIVE_KEYS}
    top = max(counts, key=counts.get)
    belief = {key: 0.10 for key in OBJECTIVE_KEYS}
    belief[top] = 0.70
    return belief


def compact_mitigation_packet_from_base(public_context: Any, base_packet: dict[str, Any], reason: str) -> dict[str, Any]:
    signals = public_signals(public_context)
    event_ids = signals.get("event_ids") or ["event-0"]
    event_id = str(event_ids[0])
    belief = base_packet.get("belief", {}) if isinstance(base_packet, dict) else {}
    if not isinstance(belief, dict) or sum(safe_float(belief.get(key, 0.0), 0.0) for key in OBJECTIVE_KEYS) <= 0:
        belief = public_belief_guess(public_context)
    top = top_belief({"belief": belief})
    packet = default_action(event_id=event_id, action="LimitSession", tool="None")
    packet["belief"] = {key: safe_float(belief.get(key, 0.0), 0.0) for key in OBJECTIVE_KEYS}
    packet["uncertainty"] = min(0.35, safe_float(base_packet.get("uncertainty", 0.35), 0.35)) if isinstance(base_packet, dict) else 0.35
    packet["evidence_assessment"] = base_packet.get("evidence_assessment", [])[:4] if isinstance(base_packet.get("evidence_assessment", []), list) else packet["evidence_assessment"]
    packet["trust_update"] = {
        "source_reliability": (base_packet.get("trust_update", {}) or {}).get("source_reliability", {}) if isinstance(base_packet.get("trust_update", {}), dict) else {},
        "rationale": reason,
    }
    packet["memory_update"] = {
        "profile_quarantine": (base_packet.get("memory_update", {}) or {}).get("profile_quarantine", []) if isinstance(base_packet.get("memory_update", {}), dict) else [],
        "confirmed_profile": [top],
        "rejected_profile": (base_packet.get("memory_update", {}) or {}).get("rejected_profile", []) if isinstance(base_packet.get("memory_update", {}), dict) else [],
    }
    packet["safety_check"] = {
        "business_risk": 0.30,
        "overresponse_risk": 0.25,
        "justification": "bounded verification budget; reversible mitigation",
    }
    packet["response"] = {"tier": "L1", "action": "LimitSession", "target": event_id}
    return packet


def rescue_candidate(public_context: Any, action_texts: list[str], selector_mode: str) -> Candidate:
    signals = public_signals(public_context)
    event_ids = signals.get("event_ids") or ["event-0"]
    event_id = str(event_ids[0])
    best_belief: dict[str, float] = {}
    best_confidence = -1.0
    for text in action_texts:
        belief = partial_belief_from_text(text)
        confidence = max(belief.values()) if belief else -1.0
        if confidence > best_confidence:
            best_belief = belief
            best_confidence = confidence
    if not best_belief:
        best_belief = public_belief_guess(public_context)

    turn = safe_float(signals.get("turn", 0.0), 0.0)
    verification_spent = safe_float(signals.get("verification_cost", 0.0), 0.0)
    high_risk = bool(signals.get("high_public_risk", False))
    suspicious = bool(signals.get("suspicious", False))
    ready = high_risk and (turn >= 1.0 or verification_spent >= 1.0 or max(best_belief.values()) >= 0.55)
    if ready:
        packet = compact_mitigation_packet_from_base(
            public_context,
            {"belief": best_belief, "uncertainty": 0.30},
            "selector fallback after invalid candidate JSON",
        )
    elif suspicious:
        if selector_mode in V5_SELECTOR_MODES:
            packet = default_action(event_id=event_id, action="SourceChallenge", tool="SourceChallenge")
            packet["trust_update"]["rationale"] = "selector fallback active source challenge"
        else:
            packet = default_action(event_id=event_id, action="CrossCheck", tool="CrossCheck")
        packet["belief"] = best_belief
    else:
        packet = default_action(event_id=event_id, action="Observe", tool="None")
        packet["belief"] = best_belief
    packet["uncertainty"] = 0.30 if ready else 0.55
    packet["trust_update"]["rationale"] = "selector fallback after invalid candidate JSON"
    packet["memory_update"]["profile_quarantine"] = [] if ready else [event_id]
    packet["memory_update"]["confirmed_profile"] = [top_belief(packet)] if ready else []
    packet["safety_check"] = {
        "business_risk": 0.25 if ready else 0.15,
        "overresponse_risk": 0.25 if ready else 0.20,
        "justification": "compact fallback using public evidence only",
    }
    text = json_dumps(packet)
    candidate = score_candidate(public_context, text, selector_mode=selector_mode)
    candidate.parse_msg = "selector_fallback_all_invalid"
    candidate.diagnostics["fallback"] = True
    candidate.diagnostics["original_invalid_count"] = len(action_texts)
    candidate.selector_score += 0.5
    return candidate


def late_mitigation_governor(public_context: Any, candidates: list[Candidate], selector_mode: str) -> Candidate | None:
    if selector_mode not in {"mitigation_v3", "mitigation_v4"} | V5_SELECTOR_MODES:
        return None
    valid = [candidate for candidate in candidates if candidate.ok]
    if not valid:
        return None
    signals = public_signals(public_context)
    turn = safe_float(signals.get("turn", 0.0), 0.0)
    verification_spent = safe_float(signals.get("verification_cost", 0.0), 0.0)
    if not signals.get("high_public_risk"):
        return None

    has_mitigation = any(action_like_low_mitigation(str(candidate.diagnostics.get("response_action"))) for candidate in valid)
    if has_mitigation:
        return None
    non_verification = [
        candidate
        for candidate in valid
        if not action_like_verification(str(candidate.diagnostics.get("tool")), str(candidate.diagnostics.get("response_action")))
        and candidate.diagnostics.get("response_action") != "Observe"
    ]
    if non_verification:
        return None

    best = max(valid, key=lambda candidate: (candidate.diagnostics.get("belief_confidence", 0.0), -safe_float(candidate.diagnostics.get("uncertainty", 1.0), 1.0)))
    confidence = safe_float(best.diagnostics.get("belief_confidence", 0.0), 0.0)
    uncertainty = safe_float(best.diagnostics.get("uncertainty", 1.0), 1.0)
    risk = public_risk_level(signals)
    business_risk = safe_float(best.diagnostics.get("business_risk", 0.5), 0.5)
    overresponse_risk = safe_float(best.diagnostics.get("overresponse_risk", 0.5), 0.5)
    late_ready = bool(turn >= 2.0 or verification_spent >= 2.0)
    early_ready = bool(selector_mode == "mitigation_v4" and turn >= 1.0 and verification_spent >= 1.0 and confidence >= 0.70 and uncertainty <= 0.60)
    if selector_mode == "v5_a_constrained":
        early_ready = bool(
            turn >= 1.0
            and verification_spent >= 1.0
            and confidence >= 0.74
            and uncertainty <= 0.55
            and business_risk <= 0.60
            and overresponse_risk <= 0.60
        )
    elif selector_mode == "v5_b_belief_q":
        early_ready = bool(turn >= 1.0 and risk >= 0.35 and confidence >= 0.60 and uncertainty <= 0.70)
    elif selector_mode == "v5_c_frontier_minimax":
        early_ready = bool(
            turn >= 1.0
            and verification_spent >= 1.0
            and risk >= 0.40
            and confidence >= 0.55
            and uncertainty <= 0.75
            and business_risk <= 0.65
            and overresponse_risk <= 0.65
        )
    if not late_ready and not early_ready:
        return None
    if confidence < 0.55 and uncertainty > 0.45:
        return None

    reason = "early high-confidence governor" if early_ready and not late_ready else "late governor after bounded verification"
    packet = compact_mitigation_packet_from_base(public_context, best.packet, reason)
    candidate = score_candidate(public_context, json_dumps(packet), selector_mode=selector_mode)
    candidate.parse_msg = "selector_governor_early_mitigation" if early_ready and not late_ready else "selector_governor_late_mitigation"
    candidate.diagnostics["governor_override"] = True
    candidate.diagnostics["governor_stage"] = "early" if early_ready and not late_ready else "late"
    candidate.diagnostics["governor_source_tool"] = best.diagnostics.get("tool")
    candidate.diagnostics["governor_source_action"] = best.diagnostics.get("response_action")
    candidate.diagnostics["governor_risk_level"] = risk
    candidate.diagnostics["original_candidate_count"] = len(candidates)
    candidate.selector_score += 1.0
    return candidate


def active_probe_candidate(public_context: Any, selector_mode: str) -> Candidate | None:
    if selector_mode not in V5_SELECTOR_MODES:
        return None
    signals = public_signals(public_context)
    turn = safe_float(signals.get("turn", 0.0), 0.0)
    verification_spent = safe_float(signals.get("verification_cost", 0.0), 0.0)
    if not signals.get("suspicious") or turn > 0.0 or verification_spent > 0.0:
        return None
    event_ids = signals.get("event_ids") or ["event-0"]
    event_id = str(event_ids[0])
    packet = default_action(event_id=event_id, action="SourceChallenge", tool="SourceChallenge")
    packet["trust_update"]["rationale"] = "v5-c active probe augmentation"
    packet["uncertainty"] = 0.60
    packet["safety_check"] = {
        "business_risk": 0.10,
        "overresponse_risk": 0.10,
        "justification": "low-impact source challenge before trust-dependent response",
    }
    candidate = score_candidate(public_context, json_dumps(packet), selector_mode=selector_mode)
    candidate.diagnostics["active_probe_augmentation"] = True
    candidate.selector_score += 0.75
    return candidate


def select_candidate(public_context: Any, action_texts: list[str], policy: str, selector_mode: str = "mitigation_v2") -> Candidate:
    candidates = [score_candidate(public_context, text, selector_mode=selector_mode) for text in action_texts]
    if policy != "agentguard_zero_select":
        return candidates[0]
    if not any(bool(candidate.diagnostics.get("active_probe_like")) for candidate in candidates):
        augmented = active_probe_candidate(public_context, selector_mode=selector_mode)
        if augmented is not None:
            candidates.append(augmented)
    valid_candidates = [candidate for candidate in candidates if candidate.ok]
    if not valid_candidates and selector_mode != "v1":
        return rescue_candidate(public_context, action_texts, selector_mode=selector_mode)
    override = late_mitigation_governor(public_context, candidates, selector_mode=selector_mode)
    if override is not None:
        return override
    return max(candidates, key=lambda cand: cand.selector_score)


def default_action(event_id: str = "event-0", action: str = "Observe", tool: str = "None") -> dict[str, Any]:
    belief = {key: 0.10 for key in OBJECTIVE_KEYS}
    belief["exfiltration"] = 0.70
    return {
        "belief": belief,
        "evidence_assessment": [
            {"event_id": event_id, "status": "unverified", "suspected_poisoning": True},
        ],
        "trust_update": {"source_reliability": {}, "rationale": "public-context conservative assessment"},
        "uncertainty": 0.45,
        "tool_call": {"name": tool, "args": {"event_id": event_id} if tool != "None" else {}},
        "memory_update": {"profile_quarantine": [event_id], "confirmed_profile": [], "rejected_profile": []},
        "safety_check": {"business_risk": 0.2, "overresponse_risk": 0.2, "justification": "avoid high-impact response before verification"},
        "response": {"tier": "L1", "action": action, "target": event_id},
    }


class MockBackend:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def generate(self, messages: list[dict[str, str]], public_context: Any, n: int) -> list[str]:
        event_ids = public_signals(public_context).get("event_ids") or ["event-0"]
        event_id = str(event_ids[0])
        packets = [
            default_action(event_id=event_id, action="Observe", tool="None"),
            default_action(event_id=event_id, action="CrossCheck", tool="CrossCheck"),
            default_action(event_id=event_id, action="SourceChallenge", tool="SourceChallenge"),
            default_action(event_id=event_id, action="DecoyProbe", tool="DecoyProbe"),
            default_action(event_id=event_id, action="ShadowBlock", tool="ProvenanceCheck"),
            default_action(event_id=event_id, action="Isolate", tool="None"),
        ]
        packets[-1]["safety_check"]["business_risk"] = 0.85
        packets[-1]["safety_check"]["overresponse_risk"] = 0.90
        texts = [json_dumps(packet) for packet in packets]
        while len(texts) < n:
            texts.append(json_dumps(default_action(event_id=event_id)))
        return texts[:n]


class HFBackend:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

        self.torch = torch
        self.args = args
        self.prompt_style = (
            "lily" if "lily-cybersecurity" in str(args.model_path).lower() else "chat"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if args.attn_implementation != "auto":
            kwargs["attn_implementation"] = args.attn_implementation
        if args.dtype == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif args.dtype == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif args.dtype == "fp32":
            kwargs["torch_dtype"] = torch.float32
        else:
            kwargs["torch_dtype"] = "auto"
        if args.device_map:
            kwargs["device_map"] = args.device_map

        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        model_class = (
            AutoModelForVision2Seq
            if type(model_config) in AutoModelForVision2Seq._model_mapping.keys()
            else AutoModelForCausalLM
        )
        self.model = model_class.from_pretrained(args.model_path, config=model_config, **kwargs)
        if args.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, args.adapter_path)
        if not args.device_map:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device
        self.model.eval()

    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        if self.prompt_style == "lily":
            system_parts = [
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "system"
            ]
            history_parts = []
            for message in messages:
                role = str(message.get("role", "user"))
                if role == "system":
                    continue
                label = "Defender action" if role == "assistant" else "Defense observation"
                history_parts.append(f"{label}:\n{message.get('content', '')}")
            instruction = " ".join(
                [
                    "You are Lily, a cybersecurity defense analyst.",
                    *system_parts,
                    "Follow the supplied VDA schema and return exactly one compact JSON object only.",
                ]
            )
            history_text = "\n\n".join(history_parts)
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{history_text}\n\n"
                "### Response:\n"
            )
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                # Some chat templates, notably Mistral-style templates, reject system
                # messages or non-alternating roles. Fall back to a plain prompt so
                # VDA smoke/eval can still run with those models.
                pass
        chunks = []
        for msg in messages:
            chunks.append(f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}")
        chunks.append("ASSISTANT:")
        return "\n\n".join(chunks)

    def generate_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        public_contexts: list[Any],
        n: int,
    ) -> list[list[str]]:
        del public_contexts
        prompt_texts = [self.format_prompt(messages) for messages in message_batches]
        encoded = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_input_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        do_sample = bool(n > 1 or self.args.do_sample)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.max_new_tokens,
            "num_return_sequences": n,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p
            generate_kwargs["top_k"] = max(0, int(getattr(self.args, "top_k", 0)))
        if bool(getattr(self.args, "stop_on_complete_json", True)):
            from transformers import StoppingCriteriaList
            from agentguard_zero.json_stopping import CompleteJSONObjectCriteria

            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [CompleteJSONObjectCriteria(self.tokenizer, batch_size=len(message_batches) * n)]
            )
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **generate_kwargs)
        input_len = encoded["input_ids"].shape[-1]
        decoded: list[str] = []
        for row in output:
            decoded.append(self.tokenizer.decode(row[input_len:], skip_special_tokens=True).strip())
        return [decoded[index * n : (index + 1) * n] for index in range(len(message_batches))]

    def generate(self, messages: list[dict[str, str]], public_context: Any, n: int) -> list[str]:
        return self.generate_batch([messages], [public_context], n)[0]


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class APIBackend:
    """OpenAI-compatible chat-completions backend for black-box transfer eval."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = args.api_model or first_env("AGZ_API_MODEL", "LLM_MODEL")
        self.base_url = (args.api_base_url or first_env("AGZ_API_BASE_URL", "LLM_BASE_URL")).rstrip("/")
        key_envs = [args.api_key_env] if args.api_key_env else []
        key_envs.extend(["AGZ_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY", "ZHIPUAI_API_KEY", "GLM_API_KEY", "BIGMODEL_API_KEY", "OPENAI_API_KEY"])
        self.api_key_env = next((name for name in key_envs if name and os.environ.get(name)), "")
        self.api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        self.timeout = float(args.api_timeout)
        self.retries = int(args.api_retries)
        self.response_format_json = bool(args.api_response_format_json)
        self.multi_choice = bool(args.api_multi_choice)
        self.usage: dict[str, int] = {
            "api_request_count": 0,
            "api_prompt_tokens": 0,
            "api_completion_tokens": 0,
            "api_total_tokens": 0,
        }

        if not self.model:
            raise SystemExit("API backend requires --api_model or AGZ_API_MODEL/LLM_MODEL.")
        if not self.base_url:
            raise SystemExit("API backend requires --api_base_url or AGZ_API_BASE_URL/LLM_BASE_URL.")
        if not self.api_key:
            raise SystemExit(
                "API backend requires an API key in AGZ_API_KEY, LLM_API_KEY, DEEPSEEK_API_KEY, "
                "ZHIPUAI_API_KEY, GLM_API_KEY, BIGMODEL_API_KEY, or OPENAI_API_KEY."
            )

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def _request_many(self, messages: list[dict[str, str]], n: int = 1) -> list[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_tokens": self.args.max_new_tokens,
            "stream": False,
        }
        if n > 1:
            payload["n"] = n
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if self.args.api_disable_thinking:
            payload["thinking"] = {"type": "disabled"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(self.chat_url, data=body, headers=headers, method="POST")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                self.usage["api_request_count"] += 1
                usage = data.get("usage", {}) if isinstance(data, dict) else {}
                if isinstance(usage, dict):
                    self.usage["api_prompt_tokens"] += int(safe_float(usage.get("prompt_tokens", 0), 0))
                    self.usage["api_completion_tokens"] += int(safe_float(usage.get("completion_tokens", 0), 0))
                    self.usage["api_total_tokens"] += int(safe_float(usage.get("total_tokens", 0), 0))
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("API response has no choices.")
                outputs: list[str] = []
                for choice in choices:
                    message = choice.get("message", {}) if isinstance(choice, dict) else {}
                    content = message.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
                    if str(content).strip():
                        outputs.append(str(content).strip())
                if not outputs:
                    raise RuntimeError("API response content is empty.")
                return outputs
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:800]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            except Exception as exc:  # pragma: no cover - network/provider path
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
        raise RuntimeError(f"API request failed after {self.retries + 1} attempts: {last_error}")

    def _request_one(self, messages: list[dict[str, str]]) -> str:
        return self._request_many(messages, n=1)[0]

    def generate(self, messages: list[dict[str, str]], public_context: Any, n: int) -> list[str]:
        request_messages = list(messages)
        if self.args.api_system_prompt:
            if request_messages and request_messages[0].get("role") == "system":
                request_messages[0] = {
                    "role": "system",
                    "content": self.args.api_system_prompt + "\n" + request_messages[0].get("content", ""),
                }
            else:
                request_messages = [{"role": "system", "content": self.args.api_system_prompt}] + request_messages
        n = max(1, n)
        if n > 1 and self.multi_choice:
            try:
                outputs = self._request_many(request_messages, n=n)
                if len(outputs) >= n:
                    return outputs[:n]
                outputs.extend(self._request_one(request_messages) for _ in range(n - len(outputs)))
                return outputs
            except Exception:
                pass
        return [self._request_one(request_messages) for _ in range(n)]


def scenario_extra_from_row(row: dict[str, Any]) -> dict[str, Any]:
    extra = as_dict(row.get("extra_info", {}))
    scenario = (
        as_dict(extra.get("scenario"))
        or as_dict(row.get("scenario"))
        or as_dict(row.get("answer"))
        or as_dict(as_dict(row.get("reward_model", {})).get("ground_truth"))
    )
    if scenario:
        extra["scenario"] = scenario
    scenario_id = row.get("scenario_id") or extra.get("scenario_id") or scenario.get("scenario_id") or row.get("id")
    if scenario_id is not None:
        extra["scenario_id"] = str(scenario_id)
    if "max_env_steps" not in extra and scenario:
        phases = scenario.get("true_attack", {}).get("phase_schedule", [])
        extra["max_env_steps"] = max(5, len(phases) + 2)
    return extra


def row_public_context(messages: list[dict[str, str]]) -> Any:
    joined = "\n".join(msg.get("content", "") for msg in messages)
    start = joined.find("Current decision instance:")
    if start >= 0:
        maybe = joined[start + len("Current decision instance:") :].strip()
        loaded = maybe_json(maybe)
        if isinstance(loaded, dict):
            return sanitize_public_value(loaded)
    return sanitize_public_value({"messages": messages})


def sanitize_initial_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], Any]:
    marker = "Current decision instance:"
    sanitized: list[dict[str, str]] = []
    public_context: Any | None = None
    for msg in messages:
        content = msg.get("content", "")
        if marker in content:
            prefix, suffix = content.split(marker, 1)
            loaded = maybe_json(suffix.strip())
            if isinstance(loaded, dict):
                public_context = sanitize_public_value(loaded)
                content = prefix + marker + json_dumps(public_context)
        sanitized.append({"role": msg.get("role", "user"), "content": content})
    if public_context is None:
        public_context = row_public_context(sanitized)
    return sanitized, public_context


def next_user_message(observation: dict[str, Any]) -> tuple[dict[str, str], Any]:
    raw = observation.get("obs", "")
    payload = maybe_json(raw)
    if not isinstance(payload, dict):
        payload = {"observation": raw}
    public_payload = sanitize_public_value(payload)
    content = (
        "Environment observation after your previous VDA action:"
        + json_dumps(public_payload)
        + "\nReturn the next VDA action as compact strict JSON only. "
        "Use at most four evidence_assessment items; keep rationale and justification under 20 words."
    )
    return {"role": "user", "content": content}, public_payload


def compute_safe_utility(score: dict[str, Any], steps: int, max_steps: int, budget: float) -> float:
    attack_mitigation = float(bool(score.get("attack_mitigated", False)))
    intent_f1 = float(bool(score.get("correct_intent", False)))
    business_cost = min(1.0, safe_float(score.get("business_cost", 0.0)) / max(1.0, budget))
    overresponse = float(bool(score.get("overresponse", False)))
    verification_cost = min(1.0, safe_float(score.get("verification_cost", 0.0)) / max(1.0, float(max_steps)))
    delay = min(1.0, float(steps) / max(1.0, float(max_steps)))
    return (
        0.35 * attack_mitigation
        + 0.20 * intent_f1
        - 0.20 * business_cost
        - 0.15 * overresponse
        - 0.05 * verification_cost
        - 0.05 * delay
    )


def run_one(row: dict[str, Any], row_index: int, backend: Any, args: argparse.Namespace) -> dict[str, Any]:
    policy = args.policy
    candidate_count = max(1, args.candidate_count) if policy == "agentguard_zero_select" else 1
    messages, public_context = sanitize_initial_messages(as_messages(row.get("problem", "")))
    extra = scenario_extra_from_row(row)
    scenario = as_dict(extra.get("scenario"))
    scenario_id = str(extra.get("scenario_id", row.get("scenario_id", f"row-{row_index}")))
    max_env_steps = int(extra.get("max_env_steps", args.max_turns))
    max_turns = min(args.max_turns, max_env_steps)
    budget = safe_float(scenario.get("defense_constraints", {}).get("business_budget", 5.0), 5.0)
    trajectory_id = f"{args.run_name}-{policy}-{row_index}-{scenario_id}"
    store = Level1RolloutStore(invalid_penalty=args.invalid_penalty)

    selected_actions: list[dict[str, Any]] = []
    final_observation: dict[str, Any] | None = None
    done = False

    for turn in range(max_turns):
        raw_candidates = backend.generate(messages, public_context, candidate_count)
        selected = select_candidate(public_context, raw_candidates, policy, selector_mode=args.selector_mode)
        selected_actions.append(
            {
                "turn": turn,
                "selected_text": selected.text,
                "selected_packet": selected.packet,
                "selected_ok": selected.ok,
                "parse_msg": selected.parse_msg,
                "selector_score": selected.selector_score,
                "diagnostics": selected.diagnostics,
                "candidate_count": len(raw_candidates),
                "candidate_diagnostics": [
                    score_candidate(public_context, cand, selector_mode=args.selector_mode).diagnostics
                    for cand in raw_candidates
                ],
            }
        )
        response = store.handle(
            {
                "trajectory_ids": [trajectory_id],
                "actions": [selected.text],
                "finish": [False],
                "is_last_step": [turn == max_turns - 1],
                "extra_fields": [extra],
            }
        )
        final_observation = response["observations"][0]
        done = bool(response["dones"][0])
        messages.append({"role": "assistant", "content": selected.text})
        if done:
            break
        user_msg, public_context = next_user_message(final_observation)
        messages.append(user_msg)

    score = dict((final_observation or {}).get("score", {}))
    steps = int(score.get("steps", len(selected_actions)))
    score["safe_utility"] = compute_safe_utility(score, steps=steps, max_steps=max_env_steps, budget=budget)
    selected_parse_failures = sum(1 for item in selected_actions if str(item["parse_msg"]).startswith("json_parse_error"))
    selected_invalid_tools = sum(1 for item in selected_actions if item["parse_msg"] == "invalid_tool")
    selected_invalid_actions = sum(1 for item in selected_actions if item["parse_msg"] == "invalid_response_action")
    return {
        "row_index": row_index,
        "scenario_id": scenario_id,
        "split": row.get("split", "unknown"),
        "policy": policy,
        "trajectory_id": trajectory_id,
        "done": done,
        "steps": steps,
        "score": score,
        "selected_json_parse_failures": selected_parse_failures,
        "selected_invalid_tool_calls": selected_invalid_tools,
        "selected_invalid_response_actions": selected_invalid_actions,
        "selected_actions": selected_actions,
    }


def load_rows(path: str, split: str, limit: int | None, seed: int, offset: int = 0) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    if split != "all" and "split" in df.columns:
        df = df[df["split"] == split]
    rows = df.to_dict(orient="records")
    random.Random(seed).shuffle(rows)
    if offset > 0:
        rows = rows[offset:]
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = len(results)
    action_count = sum(max(1, len(item.get("selected_actions", []))) for item in results)
    raw_candidate_count = 0
    raw_candidate_json_failures = 0
    selector_fallbacks = 0
    selector_governor_overrides = 0
    for item in results:
        for action in item.get("selected_actions", []):
            diagnostics = action.get("diagnostics", {}) if isinstance(action, dict) else {}
            if diagnostics.get("fallback"):
                selector_fallbacks += 1
            if diagnostics.get("governor_override"):
                selector_governor_overrides += 1
            candidate_diags = action.get("candidate_diagnostics", []) if isinstance(action, dict) else []
            for diag in candidate_diags:
                if isinstance(diag, dict):
                    raw_candidate_count += 1
                    raw_candidate_json_failures += int(bool(diag.get("json_parse_failure")))
    summary = {
        "run_name": args.run_name,
        "policy": args.policy,
        "model_backend": args.model_backend,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "api_model": args.api_model if args.model_backend == "api" else "",
        "api_base_url": args.api_base_url if args.model_backend == "api" else "",
        "num_scenarios": n,
        "candidate_count": args.candidate_count if args.policy == "agentguard_zero_select" else 1,
        "selector_mode": args.selector_mode if args.policy == "agentguard_zero_select" else "",
        "offset": args.offset,
        "safe_utility": mean(item["score"].get("safe_utility", 0.0) for item in results),
        "trajectory_reward": mean(item["score"].get("reward", 0.0) for item in results),
        "safe_success_rate": mean(float(bool(item["score"].get("safe_success", False))) for item in results),
        "attack_mitigation": mean(float(bool(item["score"].get("attack_mitigated", False))) for item in results),
        "attack_success": mean(float(bool(item["score"].get("attack_success", False))) for item in results),
        "intent_accuracy": mean(float(bool(item["score"].get("correct_intent", False))) for item in results),
        "business_cost": mean(safe_float(item["score"].get("business_cost", 0.0)) for item in results),
        "verification_cost": mean(safe_float(item["score"].get("verification_cost", 0.0)) for item in results),
        "overresponse_rate": mean(float(bool(item["score"].get("overresponse", False))) for item in results),
        "json_parse_failure_rate": sum(item["selected_json_parse_failures"] for item in results) / max(1, action_count),
        "raw_candidate_json_parse_failure_rate": raw_candidate_json_failures / max(1, raw_candidate_count),
        "selector_fallback_rate": selector_fallbacks / max(1, action_count),
        "selector_governor_override_rate": selector_governor_overrides / max(1, action_count),
        "invalid_tool_call_rate": sum(item["selected_invalid_tool_calls"] for item in results) / max(1, action_count),
        "invalid_response_action_rate": sum(item["selected_invalid_response_actions"] for item in results) / max(1, action_count),
        "avg_steps": mean(float(item.get("steps", 0)) for item in results),
    }
    return summary


def write_outputs(results: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json_dumps(item) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        f.write(json_dumps(summary, indent=2) + "\n")
    with (output_dir / "pilot_table.md").open("w", encoding="utf-8") as f:
        f.write("| Metric | Value |\n|---|---:|\n")
        for key in (
            "safe_utility",
            "trajectory_reward",
            "safe_success_rate",
            "attack_mitigation",
            "attack_success",
            "intent_accuracy",
            "business_cost",
            "verification_cost",
            "overresponse_rate",
            "json_parse_failure_rate",
            "raw_candidate_json_parse_failure_rate",
            "selector_fallback_rate",
            "selector_governor_override_rate",
            "invalid_tool_call_rate",
            "invalid_response_action_rate",
            "avg_steps",
        ):
            f.write(f"| {key} | {summary.get(key, 0.0):.6f} |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Level-1 frontier VDA parquet.")
    parser.add_argument("--model_path", default=os.environ.get("AGZ_MODEL_PATH", ""))
    parser.add_argument("--adapter_path", default=os.environ.get("AGZ_ADAPTER_PATH", ""))
    parser.add_argument("--policy", choices=["base_tools", "zero_shot_vda", "agentguard_zero_select"], default="agentguard_zero_select")
    parser.add_argument("--model_backend", choices=["hf", "mock", "api"], default="hf")
    parser.add_argument("--candidate_count", type=int, default=4)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", default="all")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument(
        "--selector_mode",
        choices=[
            "v1",
            "mitigation_v2",
            "mitigation_v3",
            "mitigation_v4",
            "v5_a_constrained",
            "v5_b_belief_q",
            "v5_c_frontier_minimax",
        ],
        default=os.environ.get("AGZ_SELECTOR_MODE", "mitigation_v3"),
    )
    parser.add_argument("--run_name", default=f"select_eval_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "eval_select"))
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--invalid_penalty", type=float, default=0.5)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device_map", default=os.environ.get("AGZ_EVAL_DEVICE_MAP", ""))
    parser.add_argument("--attn_implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default=os.environ.get("AGZ_ATTN_IMPLEMENTATION", "auto"))
    parser.add_argument("--api_model", default=os.environ.get("AGZ_API_MODEL", os.environ.get("LLM_MODEL", "")))
    parser.add_argument("--api_base_url", default=os.environ.get("AGZ_API_BASE_URL", os.environ.get("LLM_BASE_URL", "")))
    parser.add_argument("--api_key_env", default=os.environ.get("AGZ_API_KEY_ENV", ""))
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("AGZ_API_TIMEOUT", "90")))
    parser.add_argument("--api_retries", type=int, default=int(os.environ.get("AGZ_API_RETRIES", "2")))
    parser.add_argument("--api_response_format_json", action="store_true", default=env_flag("AGZ_API_RESPONSE_FORMAT_JSON", False))
    parser.add_argument("--api_disable_thinking", action="store_true", default=env_flag("AGZ_API_DISABLE_THINKING", False))
    parser.add_argument("--api_multi_choice", action="store_true", default=env_flag("AGZ_API_MULTI_CHOICE", False))
    parser.add_argument(
        "--api_system_prompt",
        default=os.environ.get(
            "AGZ_API_SYSTEM_PROMPT",
            "Return compact strict JSON only. Use at most four evidence_assessment items and keep rationale/justification under 20 words. Do not include markdown, comments, prose, executable code, payloads, exploit steps, malware logic, real IPs, or real organizations.",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rows = load_rows(args.data, split=args.split, limit=args.limit, seed=args.seed, offset=args.offset)
    if not rows:
        raise SystemExit(f"No rows loaded from {args.data} with split={args.split}")
    if args.model_backend == "hf" and not args.model_path:
        raise SystemExit("--model_path is required for --model_backend hf")

    if args.model_backend == "mock":
        backend = MockBackend(args.seed)
    elif args.model_backend == "api":
        backend = APIBackend(args)
    else:
        backend = HFBackend(args)
    output_dir = Path(args.output_dir) / args.run_name
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        result = run_one(row, idx, backend, args)
        results.append(result)
        score = result["score"]
        print(
            json_dumps(
                {
                    "progress": f"{idx + 1}/{len(rows)}",
                    "scenario_id": result["scenario_id"],
                    "safe_utility": score.get("safe_utility"),
                    "reward": score.get("reward"),
                    "attack_mitigated": score.get("attack_mitigated"),
                    "overresponse": score.get("overresponse"),
                }
            ),
            flush=True,
        )

    summary = summarize(results, args)
    if hasattr(backend, "usage"):
        summary.update(getattr(backend, "usage"))
    write_outputs(results, summary, output_dir)
    print(json_dumps({"summary": summary, "output_dir": str(output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
