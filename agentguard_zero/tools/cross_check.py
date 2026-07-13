from __future__ import annotations

from typing import Any, Dict, List


def cross_check(event: Dict[str, Any], sources: List[str]) -> Dict[str, Any]:
    spoof = float(event.get("spoofability", 0.5))
    support = max(0.0, min(1.0, 1.0 - spoof))
    return {
        "tool": "CrossCheck",
        "event_id": event.get("event_id", event.get("id", "unknown")),
        "requested_sources": sources,
        "consistency": support,
        "verdict": "weakly_supported" if support < 0.5 else "supported",
    }
