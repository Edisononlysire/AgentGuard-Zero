from __future__ import annotations

import copy

from scripts.build_v11_t12_native_v2 import (
    ATTACK_STAGES,
    DECISION_FAMILIES,
    _apply_lifecycle_supervision,
    _candidate_set_hash,
    _complete_chain_rows,
    _max_flow_cell_quotas,
)


def _candidate(key: str, family: str) -> dict:
    return {
        "candidate_key": key,
        "semantic_id": f"semantic-{key}",
        "action_family": family,
        "compiled_packet": {},
    }


def test_lifecycle_supervision_preserves_unconstrained_teacher_target() -> None:
    row = {
        "target_candidate_id": "teacher",
        "target_candidate_key": "teacher",
        "target_semantic_id": "semantic-teacher",
        "target_family": "active_probe",
        "causal_target_candidate_id": "stop",
        "causal_target_family": "observe",
        "candidates": [
            _candidate("teacher", "active_probe"),
            _candidate("stop", "observe"),
        ],
        "hard_negative_candidate_ids": ["stop"],
        "trajectory_negative_candidate_ids": ["stop"],
        "audit": {},
    }
    assert _apply_lifecycle_supervision(row) is True
    assert row["target_candidate_id"] == "stop"
    assert row["target_family"] == "observe"
    assert row["target_semantic_id"] == "semantic-stop"
    assert row["hard_negative_candidate_ids"] == []
    assert row["trajectory_negative_candidate_ids"] == []
    assert row["audit"]["unconstrained_teacher_target_candidate_id"] == "teacher"
    assert (
        row["audit"]["supervision_target_override"]
        == "public_lifecycle_family_then_teacher_q_v1"
    )


def test_lifecycle_supervision_rejects_missing_target_without_relabelling() -> None:
    row = {
        "target_candidate_id": "teacher",
        "target_candidate_key": "teacher",
        "target_semantic_id": "semantic-teacher",
        "target_family": "active_probe",
        "causal_target_candidate_id": None,
        "causal_target_family": None,
        "candidates": [_candidate("teacher", "active_probe")],
        "audit": {},
    }
    original = copy.deepcopy(row)
    assert _apply_lifecycle_supervision(row) is False
    assert row == original


def test_family_stage_flow_has_exact_margins() -> None:
    family_remaining = {family: 7 for family in DECISION_FAMILIES}
    stage_remaining = {stage: 7 for stage in ATTACK_STAGES}
    capacities = {
        (family, stage): 7
        for family in DECISION_FAMILIES
        for stage in ATTACK_STAGES
    }
    quotas = _max_flow_cell_quotas(
        capacities,
        family_remaining,
        stage_remaining,
    )
    assert {
        family: sum(
            quotas[(family, stage)] for stage in ATTACK_STAGES
        )
        for family in DECISION_FAMILIES
    } == family_remaining
    assert {
        stage: sum(
            quotas[(family, stage)] for family in DECISION_FAMILIES
        )
        for stage in ATTACK_STAGES
    } == stage_remaining


def test_complete_chain_requires_all_five_ordered_families() -> None:
    rows = []
    for step, family in enumerate(DECISION_FAMILIES[1:] + DECISION_FAMILIES[:1]):
        rows.append(
            {
                "trajectory_id": "complete",
                "trajectory_step": step,
                "record_id": f"complete-{step}",
                "causal_target_family": family,
                "semantic_scenario_fingerprint": "scenario-a",
            }
        )
    incomplete = copy.deepcopy(rows[:-1])
    for row in incomplete:
        row["trajectory_id"] = "incomplete"
        row["record_id"] = "incomplete-" + row["record_id"]
    chains = _complete_chain_rows([*rows, *incomplete])
    assert len(chains) == 1
    assert [row["causal_target_family"] for row in chains[0]] == [
        "passive_verification",
        "active_probe",
        "trust",
        "mitigation",
        "observe",
    ]


def test_candidate_set_hash_is_deterministic_and_order_sensitive() -> None:
    first = {
        "candidates": [
            _candidate("a", "observe"),
            _candidate("b", "mitigation"),
        ]
    }
    second = copy.deepcopy(first)
    assert _candidate_set_hash(first) == _candidate_set_hash(second)
    second["candidates"].reverse()
    assert _candidate_set_hash(first) != _candidate_set_hash(second)
