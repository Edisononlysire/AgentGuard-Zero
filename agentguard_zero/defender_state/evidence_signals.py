from __future__ import annotations

from typing import Any

from agentguard_zero.protocol import EVIDENCE_ORIGIN_TOOL_GENERATED


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(low, min(high, number))


def evidence_signal(record: dict[str, Any]) -> tuple[float, float]:
    """Return public positive/negative support carried by one evidence record."""

    evidence_origin = str(record.get("evidence_origin", "")).lower()
    evidence_type = str(record.get("evidence_type", "")).lower()
    if (
        evidence_origin != EVIDENCE_ORIGIN_TOOL_GENERATED
        or not evidence_type.startswith("tool:")
    ):
        return 0.0, 0.0

    payload = record.get("public_payload")
    if not isinstance(payload, dict):
        payload = record.get("content", {})
    if not isinstance(payload, dict):
        payload = {}
    verdict = str(payload.get("verdict", "")).lower()
    positive = 0.0
    negative = 0.0
    if verdict in {
        "supported",
        "strongly_supported",
        "plausible",
        "challenge_passed",
        "consistent",
    }:
        positive += 1.0
    if verdict in {
        "suspicious",
        "challenge_failed",
        "contradicted",
        "inconsistent",
    }:
        negative += 1.0
    positive += _clamp(payload.get("consistency", 0.0))
    positive += _clamp(payload.get("challenge_consistency", 0.0))
    positive += _clamp(payload.get("source_reliability", 0.0))
    negative += _clamp(payload.get("contradiction_risk", 0.0))
    if payload.get("canary_triggered"):
        negative += _clamp(payload.get("leakage_risk", 0.0))
    return min(2.0, positive), min(2.0, negative)


def evidence_is_positive(record: dict[str, Any]) -> bool:
    positive, negative = evidence_signal(record)
    return positive > negative


def evidence_is_negative(record: dict[str, Any]) -> bool:
    positive, negative = evidence_signal(record)
    return negative > positive
