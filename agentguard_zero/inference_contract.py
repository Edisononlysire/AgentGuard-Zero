"""Frozen inference-contract constants and model-independent quality gates.

These gates validate whether a sampled candidate set is usable by a runtime
governor. They inspect syntax and public-state admissibility only; they never
use TMCD outcomes or model-performance labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


TRAINED_VDA_PROMPT_CONTRACT = "scenario_to_training_row_exact_v1"
GENERIC_EVAL_PROMPT_CONTRACT = "evaluation_system_prompt_v1"
FORMAL_VDA_MAX_NEW_TOKENS = 320

CANDIDATE_QUALITY_MINIMUMS = {
    "candidate_parse_ok_rate": 0.50,
    "candidate_base_admissible_rate": 0.20,
    "decision_parse_coverage": 0.95,
    "decision_base_admissible_coverage": 0.80,
}


def summarize_candidate_quality(
    decisions: Iterable[Iterable[Mapping[str, Any]]],
    *,
    expected_candidates_per_decision: int = 6,
) -> dict[str, Any]:
    """Summarize raw candidate validity without consulting hidden labels."""

    decision_count = 0
    candidate_count = 0
    parse_ok_count = 0
    base_admissible_count = 0
    decisions_with_parse_ok = 0
    decisions_with_base_admissible = 0
    malformed_decision_count = 0

    for raw_candidates in decisions:
        candidates = list(raw_candidates)
        decision_count += 1
        if len(candidates) != expected_candidates_per_decision:
            malformed_decision_count += 1
        candidate_count += len(candidates)
        parse_flags = [bool(item.get("parse_ok")) for item in candidates]
        admissible_flags = [
            bool(item.get("parse_ok")) and bool(item.get("base_admissible"))
            for item in candidates
        ]
        parse_ok_count += sum(parse_flags)
        base_admissible_count += sum(admissible_flags)
        decisions_with_parse_ok += int(any(parse_flags))
        decisions_with_base_admissible += int(any(admissible_flags))

    rates = {
        "candidate_parse_ok_rate": parse_ok_count / max(1, candidate_count),
        "candidate_base_admissible_rate": base_admissible_count
        / max(1, candidate_count),
        "decision_parse_coverage": decisions_with_parse_ok
        / max(1, decision_count),
        "decision_base_admissible_coverage": decisions_with_base_admissible
        / max(1, decision_count),
    }
    failures = [
        name
        for name, minimum in CANDIDATE_QUALITY_MINIMUMS.items()
        if rates[name] + 1.0e-12 < minimum
    ]
    if decision_count <= 0:
        failures.append("nonempty_decision_set")
    if malformed_decision_count:
        failures.append("candidate_count_per_decision")

    return {
        "status": "accepted" if not failures else "rejected",
        "accepted": not failures,
        "expected_candidates_per_decision": expected_candidates_per_decision,
        "decision_count": decision_count,
        "candidate_count": candidate_count,
        "parse_ok_count": parse_ok_count,
        "base_admissible_count": base_admissible_count,
        "decisions_with_parse_ok": decisions_with_parse_ok,
        "decisions_with_base_admissible": decisions_with_base_admissible,
        "malformed_decision_count": malformed_decision_count,
        **rates,
        "minimums": dict(CANDIDATE_QUALITY_MINIMUMS),
        "failures": failures,
        "selection_independent": True,
        "hidden_label_access": False,
    }


def require_candidate_quality(summary: Mapping[str, Any], *, context: str) -> None:
    """Reject missing, weakened, or failed candidate-quality evidence."""

    if summary.get("minimums") != CANDIDATE_QUALITY_MINIMUMS:
        raise ValueError(f"{context} candidate-quality minima mismatch")
    if summary.get("accepted") is not True or summary.get("status") != "accepted":
        raise ValueError(
            f"{context} candidate-quality gate failed: "
            f"{list(summary.get('failures', []))}"
        )
    if summary.get("selection_independent") is not True:
        raise ValueError(f"{context} candidate-quality gate is selection-dependent")
    if summary.get("hidden_label_access") is not False:
        raise ValueError(f"{context} candidate-quality gate used hidden labels")
