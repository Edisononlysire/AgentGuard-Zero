from __future__ import annotations

import json
from collections import deque
from typing import Any, Dict, Tuple

from agentguard_zero.schemas.scenario_schema import OBJECTIVES, PHASES, EVIDENCE_TYPES


def parse_scenario_json(text: str) -> Tuple[Dict[str, Any], bool, str]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
            except Exception as exc:
                return {}, False, f"json_parse_error: {exc}"
        else:
            return {}, False, "json_parse_error"
    return obj, True, "ok"


def format_checker(scenario: Dict[str, Any]) -> Tuple[bool, str]:
    required = ["scenario_id", "network_context", "true_attack", "poisoning_plan", "defense_constraints", "oracle"]
    for k in required:
        if k not in scenario:
            return False, f"missing_{k}"
    return True, "ok"


def validity_checker(scenario: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = format_checker(scenario)
    if not ok:
        return ok, msg
    true_attack = scenario["true_attack"]
    poisoning = scenario["poisoning_plan"]
    if not isinstance(true_attack, dict) or not isinstance(poisoning, dict):
        return False, "invalid_attack_or_poisoning_type"
    if true_attack.get("objective") not in OBJECTIVES:
        return False, "invalid_true_objective"
    if poisoning.get("false_objective") not in OBJECTIVES:
        return False, "invalid_false_objective"
    if true_attack.get("objective") == poisoning.get("false_objective"):
        return False, "false_objective_equals_true_objective"
    for phase in true_attack.get("phase_schedule", []):
        if phase not in PHASES:
            return False, f"invalid_phase:{phase}"
    fake_evidence = poisoning.get("fake_evidence", [])
    if not isinstance(fake_evidence, list):
        return False, "invalid_fake_evidence_list"
    for ev in fake_evidence:
        if not isinstance(ev, dict):
            return False, "invalid_fake_evidence_item"
        if ev.get("type") not in EVIDENCE_TYPES:
            return False, f"invalid_evidence_type:{ev.get('type')}"
        if ev.get("intended_misbelief") not in OBJECTIVES:
            return False, "invalid_intended_misbelief"
        if isinstance(ev.get("time"), bool) or not isinstance(ev.get("time"), int) or ev["time"] < 0:
            return False, "invalid_fake_evidence_time"
    decoy_behavior = poisoning.get("decoy_behavior", [])
    if not isinstance(decoy_behavior, list):
        return False, "invalid_decoy_behavior_list"
    for ev in decoy_behavior:
        if not isinstance(ev, dict):
            return False, "invalid_decoy_behavior_item"
        if isinstance(ev.get("time"), bool) or not isinstance(ev.get("time"), int) or ev["time"] < 0:
            return False, "invalid_decoy_behavior_time"
    strategy_switch = poisoning.get("strategy_switch")
    if strategy_switch is not None:
        if not isinstance(strategy_switch, dict):
            return False, "invalid_strategy_switch"
        if (
            isinstance(strategy_switch.get("time"), bool)
            or not isinstance(strategy_switch.get("time"), int)
            or strategy_switch["time"] < 0
        ):
            return False, "invalid_strategy_switch_time"
    return True, "ok"


def _reachable(edges, src: str, dst: str) -> bool:
    graph = {}
    for a, b in edges:
        graph.setdefault(a, []).append(b)
    q, seen = deque([src]), {src}
    while q:
        node = q.popleft()
        if node == dst:
            return True
        for nxt in graph.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def solvability_checker(scenario: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = validity_checker(scenario)
    if not ok:
        return ok, msg
    assets = {a["id"] for a in scenario["network_context"].get("assets", [])}
    entry = scenario["true_attack"].get("entry")
    target = scenario["true_attack"].get("target")
    if entry not in assets or target not in assets:
        return False, "entry_or_target_not_in_assets"
    if not _reachable(scenario["network_context"].get("reachable_edges", []), entry, target):
        return False, "attack_path_not_reachable"
    cons = scenario.get("defense_constraints", {})
    if cons.get("business_budget", 0) <= 0 or cons.get("verification_budget", 0) < 1:
        return False, "invalid_budget"
    return True, "ok"


def safety_checker(scenario: Dict[str, Any]) -> Tuple[bool, str]:
    text = json.dumps(scenario).lower()
    banned = ["payload", "exploit code", "shellcode", "malware", "ransomware binary", "real ip"]
    for b in banned:
        if b in text:
            return False, f"unsafe_content:{b}"
    return True, "ok"


def full_check(scenario: Dict[str, Any]) -> Dict[str, Any]:
    if scenario.get("protocol_version") == "tmcd-v2":
        from agentguard_zero.env.checker_v2 import full_check_v2

        return full_check_v2(scenario)
    checks = {}
    for name, fn in [
        ("format", format_checker),
        ("valid", validity_checker),
        ("solvable", solvability_checker),
        ("safe", safety_checker),
    ]:
        try:
            ok, msg = fn(scenario)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            ok, msg = False, f"checker_exception:{type(exc).__name__}"
        checks[name] = {"ok": ok, "message": msg}
    checks["all_ok"] = all(v["ok"] for v in checks.values() if isinstance(v, dict))
    return checks
