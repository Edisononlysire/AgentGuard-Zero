from __future__ import annotations

import copy
import json
from typing import Any, Dict, Tuple

OBJECTIVE_KEYS = ["exfiltration", "sabotage", "persistence", "credential_theft"]
ALLOWED_TOOLS = {
    "None",
    "LogQuery",
    "CrossCheck",
    "ProvenanceCheck",
    "GraphQuery",
    "BusinessImpactEstimator",
    "SourceChallenge",
    "CanaryProbe",
    "DecoyProbe",
    "ShadowActionProbe",
    "DeployDecoy",
    "ShadowResponse",
}
ALLOWED_ACTIONS = {
    "Observe",
    "CrossCheck",
    "QueryProvenance",
    "SourceChallenge",
    "CanaryProbe",
    "DecoyProbe",
    "ShadowActionProbe",
    "DeployDecoy",
    "ShadowBlock",
    "LimitSession",
    "Isolate",
    "Restore",
    "Remove",
}

DEFAULT_ACTION_PACKET: Dict[str, Any] = {
    "belief": {k: 0.25 for k in OBJECTIVE_KEYS},
    "evidence_assessment": [],
    "tool_call": {"name": "None", "args": {}},
    "memory_update": {"profile_quarantine": [], "confirmed_profile": [], "rejected_profile": []},
    "response": {"tier": "L0", "action": "Observe", "target": "none"},
}


def _default_action_packet() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_ACTION_PACKET)


def parse_action_json(text: str) -> Tuple[Dict[str, Any], bool, str]:
    raw = text.strip() if isinstance(text, str) else str(text).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
    except Exception as exc:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
            except Exception as exc2:
                return _default_action_packet(), False, f"json_parse_error: {exc2}"
        else:
            return _default_action_packet(), False, f"json_parse_error: {exc}"
    try:
        ok, msg = validate_action_packet(obj)
    except Exception as exc:
        return _default_action_packet(), False, f"schema_validation_error: {exc}"
    return obj if ok else _default_action_packet(), ok, msg


def validate_action_packet(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_a_dict"
    for key in ["belief", "tool_call", "memory_update", "response"]:
        if key not in obj:
            return False, f"missing_{key}"
    belief = obj.get("belief", {})
    if not isinstance(belief, dict):
        return False, "belief_not_dict"
    total = 0.0
    for k in OBJECTIVE_KEYS:
        try:
            v = float(belief.get(k, 0.0))
        except Exception:
            return False, f"belief_{k}_not_float"
        if v < 0 or v > 1:
            return False, f"belief_{k}_out_of_range"
        total += v
    if total <= 0:
        return False, "belief_sum_zero"

    tool = obj.get("tool_call", {})
    if not isinstance(tool, dict):
        return False, "tool_call_not_dict"
    if tool.get("name", "None") not in ALLOWED_TOOLS:
        return False, "invalid_tool"
    args = tool.get("args", {})
    if not isinstance(args, dict):
        return False, "tool_args_not_dict"
    if tool.get("name") in {"BusinessImpactEstimator", "ShadowResponse", "ShadowActionProbe"} and "action" in args and not isinstance(args.get("action"), dict):
        return False, "tool_action_arg_not_dict"

    memory = obj.get("memory_update", {})
    if not isinstance(memory, dict):
        return False, "memory_update_not_dict"
    for key in ["profile_quarantine", "confirmed_profile", "rejected_profile"]:
        if key in memory and not isinstance(memory.get(key), list):
            return False, f"memory_{key}_not_list"
        if any(not isinstance(item, str) or not item.strip() for item in memory.get(key, [])):
            return False, f"memory_{key}_item_not_string"

    response = obj.get("response", {})
    if not isinstance(response, dict):
        return False, "response_not_dict"
    if response.get("action", "Observe") not in ALLOWED_ACTIONS:
        return False, "invalid_response_action"
    return True, "ok"
