from __future__ import annotations

from typing import Any, Dict, List


def make_observation(t: int, events: List[Dict[str, Any]], business_budget: float, memory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "time": t,
        "observed_events": events,
        "defense_context": {"remaining_business_budget": business_budget},
        "profile_memory": memory,
    }
