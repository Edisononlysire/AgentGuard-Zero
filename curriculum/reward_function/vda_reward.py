# Copyright 2026
# AgentGuard-Zero VDA reward adapter for VeRL/EasyR1-style training.

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for path in [ROOT, os.path.abspath(os.path.join(ROOT, ".."))]:
    if path not in sys.path:
        sys.path.insert(0, path)

from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.schemas.action_schema import parse_action_json


HIGH_IMPACT_ACTIONS = {"Isolate", "Restore", "Remove"}
MITIGATING_ACTIONS = {"LimitSession", "ShadowBlock", "Isolate", "DeployDecoy"}
ACTIVE_PROBE_TOOLS = {"SourceChallenge", "CanaryProbe", "DecoyProbe", "ShadowActionProbe"}
VERIFY_TOOLS = {
    "CrossCheck",
    "ProvenanceCheck",
    "GraphQuery",
    "BusinessImpactEstimator",
    "ShadowResponse",
    "DeployDecoy",
} | ACTIVE_PROBE_TOOLS


def _load_scenario(raw: str) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def _top_belief(action_packet: Dict[str, Any]) -> str:
    belief = action_packet.get("belief", {}) or {}
    if not belief:
        return "unknown"
    return max(belief, key=lambda key: float(belief.get(key, 0.0)))


def _initial_fake_ids(env) -> set[str]:
    fake_ids = set()
    for event in env.observe().get("observed_events", []):
        if event.get("is_fake"):
            fake_ids.add(str(event.get("event_id")))
    return fake_ids


def _initial_high_spoof_ids(env, threshold: float = 0.6) -> set[str]:
    ids = set()
    for event in env.observe().get("observed_events", []):
        if float(event.get("spoofability", 0.0)) >= threshold:
            ids.add(str(event.get("event_id")))
    return ids


def _verified_suspicious_event(action_packet: Dict[str, Any], suspicious_ids: set[str]) -> bool:
    tool = action_packet.get("tool_call", {}) or {}
    if tool.get("name") not in VERIFY_TOOLS:
        return False
    args = tool.get("args", {}) or {}
    event_id = str(args.get("event_id", ""))
    return not suspicious_ids or event_id in suspicious_ids or tool.get("name") in {"GraphQuery", "BusinessImpactEstimator", "ShadowResponse"}


def score_vda_prediction(predict: str, scenario: Dict[str, Any]) -> Dict[str, float]:
    env = instantiate_scenario(scenario)
    fake_ids = _initial_fake_ids(env)
    suspicious_ids = _initial_high_spoof_ids(env)
    action_packet, ok, msg = parse_action_json(predict)

    if not ok:
        return {
            "overall": -1.0,
            "format": 0.0,
            "intent": 0.0,
            "mitigation": 0.0,
            "verification": 0.0,
            "probe_yield": 0.0,
            "quarantine": 0.0,
            "business_cost": 0.0,
            "overresponse": 0.0,
            "profile_poisoning": 0.0,
            "verification_cost": 0.0,
            "invalid": -1.0,
        }

    format_score = 1.0
    invalid_penalty = 0.0

    try:
        _, _, _ = env.step(action_packet)
        step_failed = 0.0
    except Exception:
        step_failed = 1.0

    true_obj = scenario.get("oracle", {}).get("true_objective", "unknown")
    intent_score = 1.0 if _top_belief(action_packet) == true_obj else 0.0

    memory_update = action_packet.get("memory_update", {}) or {}
    confirmed = {str(x) for x in memory_update.get("confirmed_profile", [])}
    quarantined = {str(x) for x in memory_update.get("profile_quarantine", [])}
    fake_confirmed = 1.0 if fake_ids & confirmed else 0.0
    fake_quarantined = 1.0 if fake_ids and fake_ids & quarantined else 0.0
    suspicious_verified = 1.0 if _verified_suspicious_event(action_packet, suspicious_ids) else 0.0
    tool_name = str((action_packet.get("tool_call", {}) or {}).get("name", "None"))
    probe_yield = 1.0 if tool_name in ACTIVE_PROBE_TOOLS and (fake_ids or suspicious_ids) else 0.0

    response = action_packet.get("response", {}) or {}
    action = response.get("action", "Observe")
    mitigation_score = 1.0 if env.attack_mitigated or (intent_score and action in MITIGATING_ACTIONS) else 0.0

    budget = float(scenario.get("defense_constraints", {}).get("business_budget", 5.0))
    business_cost = min(1.0, float(getattr(env, "business_cost", 0.0)) / max(budget, 1e-6))
    high_impact_without_verification = action in HIGH_IMPACT_ACTIONS and not suspicious_verified
    overresponse = 1.0 if high_impact_without_verification or getattr(env, "high_impact_count", 0) > int(scenario.get("defense_constraints", {}).get("max_high_impact_actions", 1)) else 0.0
    verification_cost = min(1.0, float(getattr(env, "verification_cost", 0.0)) / max(float(scenario.get("defense_constraints", {}).get("verification_budget", 4)), 1.0))

    overall = (
        0.10 * format_score
        + 0.20 * intent_score
        + 0.20 * mitigation_score
        + 0.15 * suspicious_verified
        + 0.15 * probe_yield
        + 0.10 * fake_quarantined
        - 0.20 * business_cost
        - 0.15 * overresponse
        - 0.20 * fake_confirmed
        - 0.05 * verification_cost
        - 0.25 * invalid_penalty
        - 0.25 * step_failed
    )

    return {
        "overall": float(overall),
        "format": format_score,
        "intent": intent_score,
        "mitigation": mitigation_score,
        "verification": suspicious_verified,
        "probe_yield": probe_yield,
        "quarantine": fake_quarantined,
        "business_cost": -business_cost,
        "overresponse": -overresponse,
        "profile_poisoning": -fake_confirmed,
        "verification_cost": -verification_cost,
        "invalid": -invalid_penalty,
    }


def _score_one(predict: str, ground_truth: str) -> Dict[str, float]:
    try:
        scenario = _load_scenario(ground_truth)
        score = score_vda_prediction(predict, scenario)
        score["score"] = score["overall"]
        return score
    except Exception:
        return {
            "score": -1.0,
            "overall": -1.0,
            "format": 0.0,
            "intent": 0.0,
            "mitigation": 0.0,
            "verification": 0.0,
            "quarantine": 0.0,
            "invalid": -1.0,
        }


def compute_score(
    predicts: List[str] | None = None,
    ground_truths: List[str] | None = None,
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> List[Dict[str, float]] | Dict[str, float]:
    if solution_str is not None:
        return _score_one(solution_str, ground_truth or "{}")

    scores = []
    for predict, gt in zip(predicts or [], ground_truths or []):
        try:
            scores.append(_score_one(predict, gt))
        except Exception:
            scores.append({
                "score": -1.0,
                "overall": -1.0,
                "format": 0.0,
                "intent": 0.0,
                "mitigation": 0.0,
                "verification": 0.0,
                "probe_yield": 0.0,
                "quarantine": 0.0,
                "invalid": -1.0,
            })
    return scores
