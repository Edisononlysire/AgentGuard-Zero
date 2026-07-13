from __future__ import annotations

from typing import Any


def trust_metrics(env: Any) -> dict[str, float]:
    ledger = list(getattr(env, "oracle_ledger", []))
    claim_state = getattr(env.trust_manager, "claim_trust", {})
    brier: list[float] = []
    false_claims = []
    true_claims = []
    for row in ledger:
        event_id = str(row.get("event_id", ""))
        claim = claim_state.get(event_id, {})
        score = float(claim.get("score", 0.5))
        truth = 1.0 if row.get("truth_value", True) else 0.0
        brier.append((score - truth) ** 2)
        (true_claims if truth else false_claims).append((row, claim))

    false_accepted = [
        item for item in false_claims if item[1].get("status") == "supported" or float(item[1].get("score", 0.0)) >= 0.7
    ]
    false_distrusted = [
        item for item in true_claims if item[1].get("status") == "contradicted" or float(item[1].get("score", 1.0)) <= 0.3
    ]
    betrayal_detected = [
        item for item in false_claims if item[1].get("status") in {"challenged", "contradicted"}
    ]
    divergence_time = int(env.scenario.get("divergence_time", 0))
    change_claims = (
        [item for item in true_claims if int(item[0].get("time", -1)) >= divergence_time]
        if str(env.scenario.get("trajectory_type", "")) == "legitimate_change"
        else []
    )
    change_accepted = [
        item for item in change_claims if item[1].get("status") == "supported" or float(item[1].get("score", 0.0)) >= 0.7
    ]
    detection_delays: list[float] = []
    for row, claim in betrayal_detected:
        detection_delays.append(float(max(0, int(claim.get("updated_at", row.get("time", 0))) - int(row.get("time", 0)))))
    return {
        "claim_trust_brier": float(sum(brier) / max(1, len(brier))),
        "high_impact_false_claim_acceptance": float(len(false_accepted) / max(1, len(false_claims))),
        "false_distrust_rate": float(len(false_distrusted) / max(1, len(true_claims))),
        "betrayal_detection_rate": float(len(betrayal_detected) / max(1, len(false_claims))),
        "betrayal_detection_delay": float(sum(detection_delays) / max(1, len(detection_delays))),
        "true_change_acceptance": float(len(change_accepted) / max(1, len(change_claims))),
    }
