"""Minimal, public-only VDA intent wire format.

The language model chooses one semantic operation.  A deterministic compiler
then supplies protocol boilerplate and validates the resulting Action Schema
v4 packet.  This keeps the learned target focused on the actual decision while
preserving the exact environment contract.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from agentguard_zero.schemas.action_schema import ALLOWED_ACTIONS, ALLOWED_TOOLS
from agentguard_zero.schemas.action_schema_v4 import (
    DEFAULT_ACTION_PACKET_V4,
    parse_action_json_v4,
)


INTENT_FORMAT = "compact_public_action_intent_v1"
DECISION_MARKER = "Current decision instance:"


def compact_intent_prompt(full_prompt: str) -> str:
    """Remove schema-repetition instructions while retaining public context."""

    prompt = str(full_prompt)
    if DECISION_MARKER not in prompt:
        raise ValueError("VDA prompt is missing its public decision instance")
    instance = prompt.split(DECISION_MARKER, 1)[1].strip()
    # Parsing here is both a format gate and a hidden-field safety check already
    # enforced by the upstream public projector.
    parsed = json.loads(instance)
    instance = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    instruction = (
        "You are AgentGuard-Zero's VDA. Use only the public decision instance; "
        "never infer hidden truth. Choose exactly one legal intent. Prefer a "
        "useful verification/probe or reversible defense when public evidence "
        "supports it; use observe only when no other legal action is justified. "
        "Return one compact JSON object and no prose. Allowed forms: "
        '{"kind":"tool","name":string,"args":object}; '
        '{"kind":"trust","operation":object}; '
        '{"kind":"memory","operation":object}; '
        '{"kind":"memory_use","operation":object}; '
        '{"kind":"response","tier":"L0|L1|L2|L3",'
        '"action":string,"target":string}; or {"kind":"observe"}. '
        "Tool arguments and trust/memory operations may reference only IDs "
        "already present in the public instance.\n"
    )
    return f"{instruction}{DECISION_MARKER}{instance}"


def _validated_packet(source: str | Mapping[str, Any]) -> dict[str, Any]:
    raw = (
        json.dumps(dict(source), ensure_ascii=False, separators=(",", ":"))
        if isinstance(source, Mapping)
        else str(source)
    )
    packet, valid, reason = parse_action_json_v4(raw)
    if not valid:
        raise ValueError(f"invalid Action Schema v4 packet: {reason}")
    return packet


def action_intent(packet_source: str | Mapping[str, Any]) -> dict[str, Any]:
    """Project a valid full packet to its single primary semantic operation."""

    packet = _validated_packet(packet_source)
    tool = dict(packet.get("tool_call") or {})
    if str(tool.get("name", "None")) != "None":
        return {
            "kind": "tool",
            "name": str(tool["name"]),
            "args": dict(tool.get("args") or {}),
        }
    trust = list(packet.get("trust_operations") or [])
    if trust:
        return {"kind": "trust", "operation": dict(trust[0])}
    memory = list(packet.get("memory_operations") or [])
    if memory:
        return {"kind": "memory", "operation": dict(memory[0])}
    memory_use = list(packet.get("memory_usage") or [])
    if memory_use:
        return {"kind": "memory_use", "operation": dict(memory_use[0])}
    response = dict(packet.get("response") or {})
    if str(response.get("action", "Observe")) != "Observe":
        return {
            "kind": "response",
            "tier": str(response.get("tier", "L1")),
            "action": str(response["action"]),
            "target": str(response.get("target", "none")),
        }
    return {"kind": "observe"}


def action_intent_wire_json(packet_source: str | Mapping[str, Any]) -> str:
    return json.dumps(
        action_intent(packet_source),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def compile_action_intent(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one minimal intent into a validated full Action Schema v4 packet."""

    if not isinstance(intent, Mapping):
        raise ValueError("intent must be an object")
    kind = str(intent.get("kind", ""))
    packet = copy.deepcopy(DEFAULT_ACTION_PACKET_V4)
    packet["safety_check"]["justification"] = kind[:7]
    if kind == "observe":
        if set(intent) != {"kind"}:
            raise ValueError("observe intent has unexpected fields")
    elif kind == "tool":
        if set(intent) != {"kind", "name", "args"}:
            raise ValueError("tool intent has unexpected or missing fields")
        name = str(intent.get("name", ""))
        if name not in ALLOWED_TOOLS or name == "None":
            raise ValueError("invalid intent tool")
        args = intent.get("args")
        if not isinstance(args, Mapping):
            raise ValueError("intent tool args must be an object")
        packet["tool_call"] = {"name": name, "args": dict(args)}
        packet["response"]["tier"] = "L1"
    elif kind == "trust":
        if set(intent) != {"kind", "operation"}:
            raise ValueError("trust intent has unexpected or missing fields")
        operation = intent.get("operation")
        if not isinstance(operation, Mapping):
            raise ValueError("trust operation must be an object")
        packet["trust_operations"] = [dict(operation)]
    elif kind == "memory":
        if set(intent) != {"kind", "operation"}:
            raise ValueError("memory intent has unexpected or missing fields")
        operation = intent.get("operation")
        if not isinstance(operation, Mapping):
            raise ValueError("memory operation must be an object")
        packet["memory_operations"] = [dict(operation)]
    elif kind == "memory_use":
        if set(intent) != {"kind", "operation"}:
            raise ValueError("memory-use intent has unexpected or missing fields")
        operation = intent.get("operation")
        if not isinstance(operation, Mapping):
            raise ValueError("memory-use operation must be an object")
        packet["memory_usage"] = [dict(operation)]
    elif kind == "response":
        if set(intent) != {"kind", "tier", "action", "target"}:
            raise ValueError("response intent has unexpected or missing fields")
        action = str(intent.get("action", ""))
        tier = str(intent.get("tier", ""))
        target = str(intent.get("target", "")).strip()
        if action not in ALLOWED_ACTIONS or action == "Observe":
            raise ValueError("invalid intent response action")
        if tier not in {"L0", "L1", "L2", "L3"} or not target:
            raise ValueError("invalid intent response tier/target")
        packet["response"] = {"tier": tier, "action": action, "target": target}
    else:
        raise ValueError("unknown intent kind")
    raw = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    normalized, valid, reason = parse_action_json_v4(raw)
    if not valid:
        raise ValueError(f"compiled intent violates Action Schema v4: {reason}")
    return normalized


def parse_action_intent(text: str) -> tuple[dict[str, Any], bool, str]:
    """Parse generated compact intent and return a full environment packet."""

    raw = str(text).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        intent = json.loads(raw)
    except Exception as exc:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return copy.deepcopy(DEFAULT_ACTION_PACKET_V4), False, f"intent_json:{exc}"
        try:
            intent = json.loads(raw[start : end + 1])
        except Exception as nested:
            return (
                copy.deepcopy(DEFAULT_ACTION_PACKET_V4),
                False,
                f"intent_json:{nested}",
            )
    try:
        return compile_action_intent(intent), True, "ok"
    except Exception as exc:
        return copy.deepcopy(DEFAULT_ACTION_PACKET_V4), False, f"intent_invalid:{exc}"
