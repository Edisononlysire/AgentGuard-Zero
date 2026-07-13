from __future__ import annotations

from typing import Any, Dict, Iterable

from agentguard_zero.training.coevolution import scenario_fingerprint


def _flag(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(low, min(high, parsed))


def _coverage_score(scenario: Dict[str, Any], task_focus: str) -> float:
    focus = task_focus.upper()
    poisoning = scenario.get("poisoning_plan", {}) or {}
    constraints = scenario.get("defense_constraints", {}) or {}
    trust_steps = len(scenario.get("trust_building_plan", []) or [])
    fake_evidence = poisoning.get("fake_evidence", []) or []
    if focus.startswith("T1"):
        return _flag(fake_evidence and int(constraints.get("verification_budget", 0)) >= 1)
    if focus.startswith("T2"):
        trigger = scenario.get("betrayal_trigger") or {}
        return _flag(trust_steps >= 2 and trigger)
    if focus.startswith("T3"):
        return _flag(len(fake_evidence) >= 2)
    if focus.startswith("T4"):
        assets = scenario.get("network_context", {}).get("assets", []) or []
        critical = max((float(item.get("criticality", 0.0)) for item in assets), default=0.0)
        return _flag(critical >= 0.8 and float(constraints.get("business_budget", 0.0)) > 0.0)
    return 0.5


def compute_dca_reward(
    scenario: Dict[str, Any],
    eval_result: Dict[str, Any],
    seen_fingerprints: Iterable[str] | None = None,
    task_focus: str = "",
) -> Dict[str, float]:
    """Score a DCA scenario using the current VDA's real rollout feedback.

    Invalid, unsafe, or oracle-unsolvable scenarios are hard-gated. A valid
    scenario is rewarded for exposing a current VDA weakness while retaining a
    low-risk verification/probing solution and the requested TMCD coverage.
    """

    checks = eval_result.get("checks", {}) or {}
    format_score = _flag(checks.get("format", {}).get("ok", False))
    valid_score = _flag(checks.get("valid", {}).get("ok", False))
    solvable_score = _flag(checks.get("solvable", {}).get("ok", False))
    safe_score = _flag(checks.get("safe", {}).get("ok", False))
    oracle_solvable = _flag(eval_result.get("oracle_solvable", False))
    hard_gate = format_score * valid_score * solvable_score * safe_score * oracle_solvable

    current_safe_success = _flag(eval_result.get("current_vda_safe_success", False))
    weakness = 1.0 - current_safe_success
    safe_utility = _clamp(eval_result.get("current_vda_safe_utility", 0.0), -1.0, 1.0)
    utility_difficulty = _clamp((0.55 - safe_utility) / 1.10)
    cfc_difficulty = _clamp(eval_result.get("difficulty", 0.0))
    frontier = _clamp(0.45 * weakness + 0.30 * utility_difficulty + 0.25 * cfc_difficulty)

    verification_value = _clamp(eval_result.get("verification_value", 0.0))
    active_probe_need = _clamp(
        max(
            verification_value,
            weakness if int(eval_result.get("current_vda_active_probe_count", 0)) == 0 else 0.0,
        )
    )
    poisoning_strength = _clamp(
        max(
            eval_result.get("poisoning_success_proxy", 0.0),
            _flag(float(eval_result.get("current_vda_fake_confirmed", 0.0)) > 0.0),
        )
    )
    business_challenge = _clamp(
        max(
            eval_result.get("business_pressure", 0.0),
            _flag(eval_result.get("current_vda_overresponse", False)),
        )
    )
    coverage = _coverage_score(scenario, task_focus)

    fingerprint = scenario_fingerprint(scenario)
    repetition = _flag(seen_fingerprints and fingerprint in set(seen_fingerprints))
    ambiguity = _clamp(eval_result.get("ambiguity_penalty", 0.0))
    unsafe = 1.0 - safe_score
    invalid = 1.0 - (format_score * valid_score * solvable_score)

    if not hard_gate:
        overall = -1.0
    else:
        overall = (
            0.34 * frontier
            + 0.18 * weakness
            + 0.16 * active_probe_need
            + 0.10 * poisoning_strength
            + 0.10 * business_challenge
            + 0.12 * coverage
            - 0.35 * repetition
            - 0.25 * ambiguity
        )
        overall = max(-1.0, min(1.0, overall))

    return {
        "overall": float(overall),
        "format": format_score,
        "valid": valid_score,
        "solvable": solvable_score,
        "safe": safe_score,
        "oracle_solvable": oracle_solvable,
        "frontier": frontier,
        "current_vda_weakness": weakness,
        "active_probe_need": active_probe_need,
        "verification_value": verification_value,
        "poisoning_strength": poisoning_strength,
        "business_challenge": business_challenge,
        "coverage": coverage,
        "unsafe_penalty": -unsafe,
        "invalid_penalty": -invalid,
        "ambiguity_penalty": -ambiguity,
        "repetition_penalty": -repetition,
    }
