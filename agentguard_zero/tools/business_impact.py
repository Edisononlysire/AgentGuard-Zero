from __future__ import annotations

from typing import Any, Dict

ACTION_COST = {
    "Observe": 0.0,
    "CrossCheck": 0.1,
    "QueryProvenance": 0.1,
    "SourceChallenge": 0.1,
    "CanaryProbe": 0.15,
    "DecoyProbe": 0.25,
    "ShadowActionProbe": 0.1,
    "DeployDecoy": 0.3,
    "ShadowBlock": 0.4,
    "LimitSession": 0.7,
    "Isolate": 2.0,
    "Restore": 1.0,
    "Remove": 1.5,
}


def estimate_business_impact(action: Dict[str, Any], target_criticality: float = 0.5) -> Dict[str, Any]:
    name = action.get("action", "Observe")
    base = ACTION_COST.get(name, 0.2)
    return {"tool": "BusinessImpactEstimator", "action": name, "estimated_cost": base * (0.5 + target_criticality)}
