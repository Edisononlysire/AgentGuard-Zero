"""Deterministic canonical recovery scenarios with public-equivalent hidden worlds."""

from __future__ import annotations

import copy
from typing import Any

from agentguard_zero.env.checker import full_check
from agentguard_zero.protocol import TASK_FAMILY_MAP
from agentguard_zero.schemas.scenario_schema import OBJECTIVES
from agentguard_zero.schemas.scenario_schema_v2 import (
    minimal_example_v2,
    paired_counterpart_v2,
    public_prefix_hash,
)


def _rename_sources_and_events(scenario: dict[str, Any], suffix: str) -> None:
    source_mapping = {
        str(profile["source_id"]): f"{profile['source_id']}-{suffix}"
        for profile in scenario.get("source_profiles", [])
    }
    for profile in scenario.get("source_profiles", []):
        profile["source_id"] = source_mapping[str(profile["source_id"])]
    actionable = str(scenario.get("oracle", {}).get("first_actionable_event_id", ""))
    event_mapping: dict[str, str] = {}
    for index, event in enumerate(scenario.get("event_schedule", [])):
        old = str(event.get("event_id", ""))
        event["event_id"] = f"event-{suffix}-{index}"
        event_mapping[old] = event["event_id"]
        source = str(event.get("source_id", ""))
        event["source_id"] = source_mapping.get(source, source)
        event["source"] = source_mapping.get(source, source)
    if actionable in event_mapping:
        scenario["oracle"]["first_actionable_event_id"] = event_mapping[actionable]


def _base_scenario(task_id: str, group_index: int) -> dict[str, Any]:
    scenario = minimal_example_v2(trajectory_type="betrayal")
    suffix = f"recovery-{task_id.lower()}-{group_index:04d}"
    scenario["scenario_id"] = f"TMCD-RECOVERY-{task_id}-{group_index:04d}-A"
    scenario["scenario_family"] = TASK_FAMILY_MAP[task_id]
    scenario["split"] = "train"
    scenario["metadata"] = {
        "task_id": task_id,
        "manipulation_family": TASK_FAMILY_MAP[task_id],
        "canonical_recovery": True,
        "experiment_variant": "full",
    }
    _rename_sources_and_events(scenario, suffix)

    if task_id == "T1":
        for key in ("pair_id", "prefix_hash", "divergence_time", "trajectory_type"):
            scenario.pop(key, None)
        event = scenario["event_schedule"][1]
        event["truth_value"] = False
        event["is_fake"] = True
        event["spoofability"] = 0.60
        scenario["oracle"]["actionable_from"] = int(event["time"])
        scenario["oracle"]["first_actionable_event_id"] = event["event_id"]
    elif task_id == "T2":
        scenario["pair_id"] = f"pair-recovery-{group_index:04d}"
        scenario["trajectory_type"] = "betrayal"
        scenario["prefix_hash"] = public_prefix_hash(scenario)
    elif task_id == "T3":
        for key in ("pair_id", "prefix_hash", "divergence_time", "trajectory_type"):
            scenario.pop(key, None)
        poison = copy.deepcopy(scenario["event_schedule"][2])
        poison["event_id"] = f"event-{suffix}-poison"
        poison["time"] = 1
        poison["claim_semantics"]["predicate"] = "asset_risk_level"
        poison["claim_semantics"]["object"] = "low"
        poison["objective_hint"] = "persistence"
        poison["truth_value"] = False
        poison["is_fake"] = True
        scenario["event_schedule"].insert(2, poison)
        scenario["oracle"]["actionable_from"] = 1
        scenario["oracle"]["first_actionable_event_id"] = poison["event_id"]
    else:
        for key in ("pair_id", "prefix_hash", "divergence_time", "trajectory_type"):
            scenario.pop(key, None)
        scenario["defense_constraints"]["business_budget"] = 0.9
        pressure = scenario["event_schedule"][2]
        pressure["truth_value"] = False
        pressure["is_fake"] = True
        pressure["claim"] = "urgent isolation requested for critical database"
        scenario["oracle"]["actionable_from"] = int(pressure["time"])
        scenario["oracle"]["first_actionable_event_id"] = pressure["event_id"]
    return scenario


def _objective_counterpart(scenario: dict[str, Any]) -> dict[str, Any]:
    counterpart = copy.deepcopy(scenario)
    current = str(counterpart["true_attack"]["objective"])
    alternative = (
        "sabotage"
        if current != "sabotage"
        else next(item for item in sorted(OBJECTIVES) if item != current)
    )
    counterpart["scenario_id"] = str(counterpart["scenario_id"]).rsplit("-", 1)[0] + "-B"
    counterpart["true_attack"]["objective"] = alternative
    counterpart["oracle"]["true_objective"] = alternative
    counterpart["oracle"]["success_condition"] = f"prevent_{alternative}"
    for event in counterpart.get("event_schedule", []):
        claimed = str((event.get("claim_semantics", {}) or {}).get("object", ""))
        event["truth_value"] = claimed == alternative
        event["is_fake"] = claimed != alternative
        event["spoofability"] = 0.25 if claimed == alternative else 0.75
    return counterpart


def canonical_recovery_group(task_id: str, group_index: int) -> list[dict[str, Any]]:
    if task_id not in TASK_FAMILY_MAP:
        raise ValueError(f"unsupported task: {task_id}")
    first = _base_scenario(task_id, group_index)
    if task_id == "T2":
        second = paired_counterpart_v2(first)
        second["scenario_id"] = f"TMCD-RECOVERY-{task_id}-{group_index:04d}-B"
        metadata = dict(second.get("metadata", {}) or {})
        metadata["canonical_recovery"] = True
        metadata["experiment_variant"] = "full"
        second["metadata"] = metadata
    else:
        second = _objective_counterpart(first)
    group = [first, second]
    for scenario in group:
        checks = full_check(scenario)
        if not checks.get("all_ok", False):
            raise ValueError(
                f"invalid canonical recovery scenario {scenario['scenario_id']}: {checks}"
            )
    return group


def canonical_recovery_suite(
    *,
    scenario_count: int,
    group_offset: int = 0,
) -> list[list[dict[str, Any]]]:
    if scenario_count <= 0 or scenario_count % 8:
        raise ValueError("scenario_count must be a positive multiple of eight")
    groups_per_task = scenario_count // 8
    return [
        canonical_recovery_group(task_id, group_offset + index)
        for task_id in ("T1", "T2", "T3", "T4")
        for index in range(groups_per_task)
    ]
