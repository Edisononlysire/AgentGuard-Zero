"""Schema-preserving action-first serialization for structured VDA targets."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Mapping

from agentguard_zero.schemas.action_schema_v4 import parse_action_json_v4


TOP_LEVEL_ORDER = (
    "response",
    "tool_call",
    "trust_operation",
    "memory_operation",
    "memory_use",
    "assessment",
    "belief",
    "uncertainty",
    "safety_check",
    "schema_version",
)


def _ordered_mapping(
    value: Mapping[str, Any] | None, preferred: tuple[str, ...]
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    result: OrderedDict[str, Any] = OrderedDict()
    for key in preferred:
        if key in value:
            result[key] = value[key]
    for key in sorted(set(value).difference(result)):
        result[key] = value[key]
    return result


def action_first_wire_json(source: str | Mapping[str, Any]) -> str:
    """Return the same valid v4 packet with action-bearing keys first."""

    if isinstance(source, Mapping):
        packet = dict(source)
        raw = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    else:
        raw = str(source).strip()
        packet = json.loads(raw)
    _, valid, reason = parse_action_json_v4(raw)
    if not valid:
        raise ValueError(f"cannot action-first serialize invalid packet: {reason}")
    tool_name = str((packet.get("tool_call") or {}).get("name", "None"))
    response_action = str((packet.get("response") or {}).get("action", "Observe"))
    if tool_name != "None":
        leading_key = "tool_call"
    elif packet.get("trust_operation") is not None:
        leading_key = "trust_operation"
    elif packet.get("memory_operation") is not None:
        leading_key = "memory_operation"
    elif packet.get("memory_use") is not None:
        leading_key = "memory_use"
    elif response_action != "Observe":
        leading_key = "response"
    else:
        leading_key = "response"
    reordered: OrderedDict[str, Any] = OrderedDict()
    action_order = (leading_key,) + tuple(
        key for key in TOP_LEVEL_ORDER if key != leading_key
    )
    for key in action_order:
        if key not in packet:
            continue
        value = packet[key]
        if key == "response":
            value = _ordered_mapping(value, ("action", "target", "tier"))
        elif key == "tool_call":
            value = _ordered_mapping(value, ("name", "args"))
        elif key == "trust_operation":
            value = _ordered_mapping(
                value, ("op", "source_id", "event_id", "evidence_refs")
            )
        elif key == "memory_operation":
            value = _ordered_mapping(
                value, ("op", "memory_id", "event_id", "claim")
            )
        reordered[key] = value
    for key in sorted(set(packet).difference(reordered)):
        reordered[key] = packet[key]
    output = json.dumps(reordered, ensure_ascii=False, separators=(",", ":"))
    _, roundtrip_valid, roundtrip_reason = parse_action_json_v4(output)
    if not roundtrip_valid:
        raise RuntimeError(
            f"action-first serialization broke schema: {roundtrip_reason}"
        )
    return output
