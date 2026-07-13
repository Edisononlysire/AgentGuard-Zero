from __future__ import annotations

import copy
import json
from typing import Any

from agentguard_zero.schemas.action_schema import ALLOWED_ACTIONS, ALLOWED_TOOLS, OBJECTIVE_KEYS


TRUST_OPERATIONS = {"hold", "support", "challenge", "contradict", "recover"}
MEMORY_OPERATIONS = {"ingest", "promote", "demote", "reject", "reopen"}
MEMORY_USAGE_ROLES = {"support", "contradict", "background"}
MEMORY_USAGE_TARGETS = {"belief", "tool", "response"}

DEFAULT_ACTION_PACKET_V4: dict[str, Any] = {
    "schema_version": 4,
    "belief": {key: 0.25 for key in OBJECTIVE_KEYS},
    "evidence_assessment": [],
    "trust_operations": [],
    "memory_operations": [],
    "memory_usage": [],
    "uncertainty": 1.0,
    "tool_call": {"name": "None", "args": {}},
    "safety_check": {"business_risk": 0.0, "overresponse_risk": 0.0, "justification": ""},
    "response": {"tier": "L0", "action": "Observe", "target": "none"},
}


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def validate_action_packet_v4(packet: Any) -> tuple[bool, str]:
    if not isinstance(packet, dict):
        return False, "not_a_dict"
    for key in DEFAULT_ACTION_PACKET_V4:
        if key not in packet:
            return False, f"missing_{key}"
    if int(packet.get("schema_version", 0)) != 4:
        return False, "invalid_schema_version"

    belief = packet.get("belief")
    if not isinstance(belief, dict):
        return False, "belief_not_dict"
    try:
        values = [float(belief.get(key, 0.0)) for key in OBJECTIVE_KEYS]
    except (TypeError, ValueError):
        return False, "belief_not_numeric"
    if any(value < 0.0 or value > 1.0 for value in values) or sum(values) <= 0.0:
        return False, "invalid_belief"

    assessment = packet.get("evidence_assessment")
    if not isinstance(assessment, list) or len(assessment) > 4:
        return False, "invalid_evidence_assessment"
    for item in assessment:
        if not isinstance(item, dict) or not str(item.get("event_id", "")).strip():
            return False, "invalid_evidence_assessment_item"
        if item.get("status") not in {"unverified", "challenged", "supported", "contradicted"}:
            return False, "invalid_evidence_status"

    trust_ops = packet.get("trust_operations")
    if not isinstance(trust_ops, list):
        return False, "trust_operations_not_list"
    for operation in trust_ops:
        if not isinstance(operation, dict) or operation.get("op") not in TRUST_OPERATIONS:
            return False, "invalid_trust_operation"
        if not isinstance(operation.get("source_id", ""), str) or not operation.get("source_id", "").strip():
            return False, "trust_operation_missing_source"
        if operation.get("op") != "hold" and not str(operation.get("event_id", "")).strip():
            return False, "trust_operation_missing_event"
        if not _string_list(operation.get("evidence_refs", [])):
            return False, "trust_evidence_refs_invalid"
        if operation.get("op") != "hold" and not operation.get("evidence_refs"):
            return False, "trust_operation_missing_evidence"

    memory_ops = packet.get("memory_operations")
    if not isinstance(memory_ops, list):
        return False, "memory_operations_not_list"
    for operation in memory_ops:
        if not isinstance(operation, dict) or operation.get("op") not in MEMORY_OPERATIONS:
            return False, "invalid_memory_operation"
        if not _string_list(operation.get("evidence_refs", [])):
            return False, "memory_evidence_refs_invalid"
        if not operation.get("evidence_refs"):
            return False, "memory_operation_missing_evidence"
        if operation.get("op") == "ingest":
            claim = operation.get("claim")
            required = ("entity_id", "predicate", "object", "scope")
            if not isinstance(claim, dict) or any(not str(claim.get(key, "")).strip() for key in required):
                return False, "invalid_canonical_claim"
            if not operation.get("source_ids") or not _string_list(operation.get("source_ids", [])):
                return False, "invalid_memory_source_ids"
        elif not str(operation.get("memory_id", "")).strip():
            return False, "memory_operation_missing_id"

    usage = packet.get("memory_usage")
    if not isinstance(usage, list):
        return False, "memory_usage_not_list"
    for item in usage:
        if not isinstance(item, dict) or not str(item.get("memory_id", "")).strip():
            return False, "invalid_memory_usage"
        if item.get("usage") not in MEMORY_USAGE_ROLES or item.get("used_for") not in MEMORY_USAGE_TARGETS:
            return False, "invalid_memory_usage_role"

    tool = packet.get("tool_call")
    if not isinstance(tool, dict) or tool.get("name", "None") not in ALLOWED_TOOLS:
        return False, "invalid_tool"
    if not isinstance(tool.get("args", {}), dict):
        return False, "tool_args_not_dict"
    try:
        uncertainty = float(packet.get("uncertainty"))
    except (TypeError, ValueError):
        return False, "uncertainty_not_numeric"
    if not 0.0 <= uncertainty <= 1.0:
        return False, "invalid_uncertainty"
    safety = packet.get("safety_check")
    if not isinstance(safety, dict):
        return False, "safety_check_not_dict"
    try:
        business_risk = float(safety.get("business_risk"))
        overresponse_risk = float(safety.get("overresponse_risk"))
    except (TypeError, ValueError):
        return False, "safety_risk_not_numeric"
    if not 0.0 <= business_risk <= 1.0 or not 0.0 <= overresponse_risk <= 1.0:
        return False, "invalid_safety_risk"
    response = packet.get("response")
    if not isinstance(response, dict) or response.get("action", "Observe") not in ALLOWED_ACTIONS:
        return False, "invalid_response_action"
    if response.get("tier") not in {"L0", "L1", "L2", "L3"}:
        return False, "invalid_response_tier"
    if not str(response.get("target", "")).strip():
        return False, "response_missing_target"
    return True, "ok"


def parse_action_json_v4(text: str) -> tuple[dict[str, Any], bool, str]:
    raw = text.strip() if isinstance(text, str) else str(text).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        packet = json.loads(raw)
    except Exception as exc:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return copy.deepcopy(DEFAULT_ACTION_PACKET_V4), False, f"json_parse_error:{exc}"
        try:
            packet = json.loads(raw[start : end + 1])
        except Exception as nested:
            return copy.deepcopy(DEFAULT_ACTION_PACKET_V4), False, f"json_parse_error:{nested}"
    ok, message = validate_action_packet_v4(packet)
    return (packet if ok else copy.deepcopy(DEFAULT_ACTION_PACKET_V4)), ok, message
