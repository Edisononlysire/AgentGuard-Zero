from __future__ import annotations

from typing import Any, Dict, List


def _top_belief(action_packet: Dict[str, Any]) -> str | None:
    belief = action_packet.get("belief", {}) or {}
    if not belief:
        return None
    return max(belief, key=lambda k: float(belief.get(k, 0.0)))


def score_trajectory(scenario: Dict[str, Any], history: List[Dict[str, Any]], memory: Dict[str, Any], attack_mitigated: bool, attack_success: bool, business_cost: float, verification_cost: float, high_impact_count: int) -> Dict[str, Any]:
    true_obj = scenario["oracle"]["true_objective"]
    budget = float(scenario.get("defense_constraints", {}).get("business_budget", 5.0))
    max_high = int(scenario.get("defense_constraints", {}).get("max_high_impact_actions", 1))
    final_packet = history[-1]["action_packet"] if history else {}
    correct_intent = _top_belief(final_packet) == true_obj
    confirmed = {item for item in memory.get("confirmed_profile", []) if isinstance(item, str)}
    fake_ids = set()
    for step in history:
        for ev in step.get("observation", {}).get("observed_events", []):
            if ev.get("is_fake"):
                fake_ids.add(ev.get("event_id"))
    fake_confirmed = len(fake_ids & confirmed)
    quarantined = {item for item in memory.get("quarantined_profile", []) if isinstance(item, str)}
    rejected = {item for item in memory.get("rejected_profile", []) if isinstance(item, str)}
    protected_fake = fake_ids & (quarantined | rejected)
    poison_defense = len(protected_fake) / max(1, len(fake_ids))

    active_probe_count = 0
    useful_active_probe_count = 0
    trust_recalibration_count = 0
    for step in history:
        tool_result = step.get("tool_result", {}) or {}
        if tool_result.get("active_probe", False):
            active_probe_count += 1
            if any(
                key in tool_result
                for key in (
                    "verdict",
                    "canary_triggered",
                    "expected_observation",
                    "overresponse_risk",
                )
            ):
                useful_active_probe_count += 1
        trust_update = (step.get("action_packet", {}) or {}).get("trust_update", {}) or {}
        if trust_update.get("source_reliability"):
            trust_recalibration_count += 1
    probe_yield = useful_active_probe_count / max(1, active_probe_count)
    delay = len(history) / max(1, int(getattr(scenario, "max_steps", 0) or len(scenario.get("true_attack", {}).get("phase_schedule", [])) + 2))
    overresponse = business_cost > budget or high_impact_count > max_high
    safe_success = bool(correct_intent and attack_mitigated and not overresponse and fake_confirmed == 0)
    reward = (
        2.0 * float(correct_intent)
        + 3.0 * float(attack_mitigated)
        - 4.0 * float(attack_success)
        - 2.0 * float(overresponse)
        - 1.0 * fake_confirmed
        - 0.5 * business_cost
        - 0.1 * verification_cost
        + 0.75 * probe_yield
        + 0.75 * poison_defense
        - 0.05 * delay
    )
    return {
        "reward": reward,
        "safe_success": safe_success,
        "correct_intent": bool(correct_intent),
        "attack_mitigated": bool(attack_mitigated),
        "attack_success": bool(attack_success),
        "business_cost": business_cost,
        "verification_cost": verification_cost,
        "overresponse": bool(overresponse),
        "fake_confirmed": fake_confirmed,
        "fake_quarantined_or_rejected": len(protected_fake),
        "poison_defense": poison_defense,
        "active_probe_count": active_probe_count,
        "useful_active_probe_count": useful_active_probe_count,
        "probe_yield": probe_yield,
        "trust_recalibration_count": trust_recalibration_count,
        "delay": delay,
    }
