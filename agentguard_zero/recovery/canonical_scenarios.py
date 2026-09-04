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


def _diversify_public_semantics(
    scenario: dict[str, Any], task_id: str, group_index: int
) -> None:
    """Vary causal public features without introducing identifier shortcuts.

    The canonical suite is used for both the finite learnability gate and later
    held-out evaluation.  Renaming scenarios, events, or sources is not enough:
    candidate semantic normalization intentionally removes those identifiers.
    These variants therefore change real public decision inputs while retaining
    the same task contract and hidden-world pairing.
    """

    objective_index = group_index % len(OBJECTIVES)
    true_objective = OBJECTIVES[objective_index]
    false_objective = OBJECTIVES[(objective_index + 1 + group_index % 2) % len(OBJECTIVES)]
    if false_objective == true_objective:
        false_objective = OBJECTIVES[(objective_index + 1) % len(OBJECTIVES)]
    scenario["true_attack"]["objective"] = true_objective
    scenario["oracle"]["true_objective"] = true_objective
    scenario["oracle"]["success_condition"] = f"prevent_{true_objective}"

    assets = list((scenario.get("network_context") or {}).get("assets") or [])
    for asset_index, asset in enumerate(assets):
        base = (0.18, 0.55, 0.82)[min(asset_index, 2)]
        asset["criticality"] = round(
            min(0.99, base + 0.015 * ((group_index + 3 * asset_index) % 11)),
            3,
        )

    profiles = list(scenario.get("source_profiles") or [])
    for profile_index, profile in enumerate(profiles):
        profile["public_prior"] = round(
            0.36 + 0.055 * ((group_index + 2 * profile_index) % 5),
            3,
        )

    constraints = scenario["defense_constraints"]
    if task_id == "T4":
        constraints["business_budget"] = round(
            0.72 + 0.045 * (group_index % 7), 3
        )
    else:
        constraints["business_budget"] = round(
            3.6 + 0.18 * (group_index % 7), 3
        )
    minimum_verification = 6 if task_id == "T3" else 4
    constraints["verification_budget"] = minimum_verification + group_index % 3

    event_types = ("host_alert", "network_alert", "identity_alert", "audit_alert")
    for event_index, event in enumerate(scenario.get("event_schedule") or []):
        semantics = event.get("claim_semantics") or {}
        if str(semantics.get("predicate", "")) == "attack_objective":
            objective = (
                true_objective
                if bool(event.get("truth_value", not event.get("is_fake", False)))
                else false_objective
            )
            semantics["object"] = objective
            event["objective_hint"] = objective
            event["claim"] = (
                f"telemetry indicates {objective} activity against "
                f"{semantics.get('entity_id', event.get('entity_id', 'asset'))}"
            )
        event["claim_semantics"] = semantics
        event["type"] = event_types[(group_index + event_index) % len(event_types)]

    metadata = dict(scenario.get("metadata") or {})
    metadata["public_semantic_variant"] = {
        "objective": objective_index,
        "criticality": group_index % 11,
        "business_budget": group_index % 7,
        "source_prior": group_index % 5,
        "verification_budget": group_index % 3,
    }
    scenario["metadata"] = metadata
    if task_id == "T2":
        scenario["prefix_hash"] = public_prefix_hash(scenario)


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
        scenario["defense_constraints"]["require_active_probe_for_mitigation"] = True
        scenario["true_attack"]["initial_pressure"] = 0.40
    elif task_id == "T2":
        scenario["pair_id"] = f"pair-recovery-{group_index:04d}"
        scenario["trajectory_type"] = "betrayal"
        scenario["defense_constraints"]["require_trust_update_for_mitigation"] = True
        scenario["true_attack"]["initial_pressure"] = 0.40
        scenario["prefix_hash"] = public_prefix_hash(scenario)
    elif task_id == "T3":
        for key in ("pair_id", "prefix_hash", "divergence_time", "trajectory_type"):
            scenario.pop(key, None)
        template = copy.deepcopy(scenario["event_schedule"][0])
        source_a = str(scenario["event_schedule"][0]["source_id"])
        source_b = str(scenario["event_schedule"][3]["source_id"])
        schedule = []
        specifications = (
            (0, "profile-low", source_a, "asset_risk_level", "low", False),
            (1, "profile-high", source_b, "asset_risk_level", "high", True),
            (2, "profile-safe", source_a, "compromise_state", "safe", False),
            (3, "history-dependent", source_b, "attack_objective", "exfiltration", True),
            (4, "final-decision", source_b, "threat_state", "active", True),
            (5, "delayed-recall", source_b, "threat_state", "active", True),
        )
        for time, name, source, predicate, object_value, truth in specifications:
            event = copy.deepcopy(template)
            event["event_id"] = f"event-{suffix}-{name}"
            event["time"] = time
            event["source_id"] = source
            event["source"] = source
            event["entity_id"] = "database"
            event["claim"] = f"database {predicate} is {object_value}"
            event["claim_semantics"] = {
                "entity_id": "database",
                "predicate": predicate,
                "object": object_value,
                "scope": "cyber_defense",
            }
            event["objective_hint"] = (
                object_value if object_value in OBJECTIVES else "exfiltration"
            )
            event["truth_value"] = truth
            event["is_fake"] = not truth
            event["spoofability"] = 0.75 if not truth else 0.20
            schedule.append(event)
        scenario["event_schedule"] = schedule
        scenario["true_attack"]["phase_schedule"] = [
            "initial_access",
            "discovery",
            "lateral_movement",
            "collection",
            "collection",
            "exfiltration",
        ]
        scenario["defense_constraints"]["horizon"] = 7
        scenario["defense_constraints"]["verification_budget"] = 6
        scenario["defense_constraints"]["require_memory_use_for_mitigation"] = True
        scenario["oracle"]["actionable_from"] = 0
        scenario["oracle"]["first_actionable_event_id"] = schedule[0]["event_id"]
    else:
        for key in ("pair_id", "prefix_hash", "divergence_time", "trajectory_type"):
            scenario.pop(key, None)
        scenario["defense_constraints"]["business_budget"] = 0.9
        scenario["defense_constraints"]["horizon"] = 7
        pressure = scenario["event_schedule"][2]
        pressure["truth_value"] = False
        pressure["is_fake"] = True
        pressure["claim"] = "urgent isolation requested for critical database"
        scenario["oracle"]["actionable_from"] = int(pressure["time"])
        scenario["oracle"]["first_actionable_event_id"] = pressure["event_id"]
        scenario["defense_constraints"]["require_impact_probe_for_mitigation"] = True
        scenario["defense_constraints"][
            "require_profile_memory_use_for_mitigation"
        ] = True
        scenario["true_attack"]["initial_pressure"] = 0.40
    _diversify_public_semantics(scenario, task_id, group_index)
    return scenario


def _robust_hidden_counterpart(scenario: dict[str, Any]) -> dict[str, Any]:
    counterpart = copy.deepcopy(scenario)
    counterpart["scenario_id"] = str(counterpart["scenario_id"]).rsplit("-", 1)[0] + "-B"
    for index, event in enumerate(counterpart.get("event_schedule", [])):
        base = float(event.get("spoofability", 0.5))
        event["spoofability"] = max(0.05, min(0.95, base + (0.12 if index % 2 else -0.12)))
    phases = list(counterpart.get("true_attack", {}).get("phase_schedule", []))
    if len(phases) >= 3:
        phases[1], phases[2] = phases[2], phases[1]
        counterpart["true_attack"]["phase_schedule"] = phases
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
        second = _robust_hidden_counterpart(first)
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
