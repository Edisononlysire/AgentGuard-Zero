from __future__ import annotations

from typing import Any, Dict


def compute_vda_reward(outcome: Dict[str, Any]) -> Dict[str, float]:
    reward = float(outcome.get("reward", 0.0))
    return {
        "overall": reward,
        "intent": float(outcome.get("correct_intent", False)),
        "mitigation": float(outcome.get("attack_mitigated", False)),
        "business_cost": -float(outcome.get("business_cost", 0.0)),
        "overresponse": -float(outcome.get("overresponse", False)),
        "profile_poisoning": -float(outcome.get("fake_confirmed", 0)),
        "poison_defense": float(outcome.get("poison_defense", 0.0)),
        "probe_yield": float(outcome.get("probe_yield", 0.0)),
        "trust_recalibration": float(outcome.get("trust_recalibration_count", 0.0)),
        "verification_cost": -float(outcome.get("verification_cost", 0.0)),
        "delay": -float(outcome.get("delay", 0.0)),
    }
