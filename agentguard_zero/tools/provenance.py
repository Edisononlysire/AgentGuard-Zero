from __future__ import annotations

from typing import Any, Dict


def provenance_check(event: Dict[str, Any]) -> Dict[str, Any]:
    spoof = float(event.get("spoofability", 0.5))
    return {
        "tool": "ProvenanceCheck",
        "event_id": event.get("event_id", event.get("id", "unknown")),
        "source": event.get("source", "unknown"),
        "source_reliability": max(0.0, min(1.0, 1.0 - spoof)),
        "verdict": "suspicious" if spoof >= 0.6 else "plausible",
    }
