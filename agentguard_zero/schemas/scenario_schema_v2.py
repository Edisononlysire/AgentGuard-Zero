from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from agentguard_zero.schemas.scenario_schema import OBJECTIVES, PHASES
from agentguard_zero.protocol import (
    PRIVILEGED_METADATA_FIELDS,
    RAW_EVENT_ALLOWED_FIELDS,
    RAW_EVENT_FORBIDDEN_SIGNAL_FIELDS,
    RAW_EVENT_RESERVED_TYPES,
)
from agentguard_zero.world.public_projector import project_event


PROTOCOL_VERSION = "tmcd-v2"
SCHEMA_VERSION = 4
DISTRIBUTIONS = {"clean", "id", "ood", "composition"}
SOURCE_MODES = {"honest", "deceptive", "mixed", "legitimate_change", "recovered"}
TRAIN_FAMILIES = {
    "active_probe_ambiguity",
    "trust_betrayal",
    "profile_poisoning",
    "business_overresponse",
}
OOD_FAMILIES = {
    "cross_source_sleeper_betrayal",
    "poison_recovery_reentry",
    "trust_memory_feedback_composition",
}


def _event(
    time: int,
    event_id: str,
    source_id: str,
    claim: str,
    *,
    objective: str,
    truth_value: bool,
    entity_id: str = "database",
    predicate: str = "attack_objective",
    event_type: str = "host_alert",
    spoofability: float = 0.2,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "time": int(time),
        "type": event_type,
        "source_id": source_id,
        "source": source_id,
        "entity_id": entity_id,
        "claim": claim,
        "claim_semantics": {
            "entity_id": entity_id,
            "predicate": predicate,
            "object": objective,
            "scope": "cyber_defense",
        },
        "objective_hint": objective,
        "truth_value": bool(truth_value),
        "is_fake": not bool(truth_value),
        "spoofability": float(spoofability),
    }


def minimal_example_v2(*, trajectory_type: str = "betrayal") -> dict[str, Any]:
    legitimate_change = trajectory_type == "legitimate_change"
    third_objective = "exfiltration" if legitimate_change else "sabotage"
    third_truth = legitimate_change
    scenario = {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "scenario_id": f"TMCD-V2-T2-{trajectory_type}",
        "scenario_family": "trust_betrayal",
        "distribution": "id",
        "split": "train",
        "pair_id": "pair-minimal-t2",
        "divergence_time": 2,
        "trajectory_type": trajectory_type,
        "network_context": {
            "assets": [
                {"id": "user_pc", "zone": "office", "criticality": 0.2},
                {"id": "file_server", "zone": "server", "criticality": 0.7},
                {"id": "database", "zone": "server", "criticality": 0.95},
            ],
            "reachable_edges": [["user_pc", "file_server"], ["file_server", "database"]],
        },
        "true_attack": {
            "objective": "exfiltration",
            "entry": "user_pc",
            "target": "database",
            "phase_schedule": ["initial_access", "discovery", "collection", "exfiltration"],
        },
        "source_profiles": [
            {
                "source_id": "sensor-A",
                "public_prior": 0.55,
                "behavior_schedule": [
                    {"start_time": 0, "end_time": 1, "mode": "honest", "truth_rate": 1.0},
                    {
                        "start_time": 2,
                        "end_time": 3,
                        "mode": "legitimate_change" if legitimate_change else "deceptive",
                        "truth_rate": 1.0 if legitimate_change else 0.0,
                    },
                ],
            },
            {
                "source_id": "sensor-B",
                "public_prior": 0.50,
                "behavior_schedule": [
                    {"start_time": 0, "end_time": 3, "mode": "honest", "truth_rate": 1.0}
                ],
            },
        ],
        "event_schedule": [
            _event(0, "event-prefix-0", "sensor-A", "database collection activity", objective="exfiltration", truth_value=True),
            _event(1, "event-prefix-1", "sensor-A", "database access path observed", objective="exfiltration", truth_value=True),
            _event(
                2,
                f"event-divergence-{trajectory_type}",
                "sensor-A",
                "database objective claim changed",
                objective=third_objective,
                truth_value=third_truth,
                spoofability=0.78 if not third_truth else 0.25,
            ),
            _event(3, f"event-cross-{trajectory_type}", "sensor-B", "independent database observation", objective="exfiltration", truth_value=True),
        ],
        "defense_constraints": {
            "business_budget": 5.0,
            "verification_budget": 4,
            "max_high_impact_actions": 1,
        },
        "oracle": {
            "true_objective": "exfiltration",
            "success_condition": "prevent_exfiltration",
            "overresponse_condition": "isolate_database_without_verified_risk",
            "actionable_from": 2,
            "first_actionable_event_id": f"event-divergence-{trajectory_type}",
        },
        "metadata": {
            "task_id": "T2",
            "manipulation_family": "trust_betrayal",
            "trajectory_type": trajectory_type,
        },
    }
    scenario["prefix_hash"] = public_prefix_hash(scenario)
    return scenario


def public_prefix_hash(scenario: dict[str, Any]) -> str:
    def safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def safe_prior(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    divergence = safe_int(scenario.get("divergence_time", 0), 0)
    raw_events = scenario.get("event_schedule", [])
    if not isinstance(raw_events, list):
        raw_events = []
    events = [
        project_event(event)
        for event in raw_events
        if isinstance(event, dict)
        and safe_int(event.get("time", divergence), divergence) < divergence
    ]
    raw_profiles = scenario.get("source_profiles", [])
    if not isinstance(raw_profiles, list):
        raw_profiles = []
    payload = {
        "network_context": scenario.get("network_context", {}),
        "defense_constraints": scenario.get("defense_constraints", {}),
        "source_profiles": [
            {
                "source_id": str(profile.get("source_id", "")),
                "public_prior": safe_prior(profile.get("public_prior", 0.5)),
            }
            for profile in raw_profiles
            if isinstance(profile, dict)
        ],
        "events": events,
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def paired_minimal_examples_v2() -> tuple[dict[str, Any], dict[str, Any]]:
    betrayal = minimal_example_v2(trajectory_type="betrayal")
    change = minimal_example_v2(trajectory_type="legitimate_change")
    return copy.deepcopy(betrayal), copy.deepcopy(change)


def paired_counterpart_v2(scenario: dict[str, Any]) -> dict[str, Any]:
    """Create the opposite T2 branch while preserving the full public prefix."""

    if scenario.get("scenario_family") != "trust_betrayal":
        raise ValueError("paired counterparts are only defined for trust_betrayal")
    counterpart = copy.deepcopy(scenario)
    divergence = int(counterpart.get("divergence_time", 0))
    current = str(counterpart.get("trajectory_type", "betrayal"))
    target_type = "legitimate_change" if current == "betrayal" else "betrayal"
    true_objective = str(counterpart.get("oracle", {}).get("true_objective", "exfiltration"))
    false_objective = next(item for item in sorted(OBJECTIVES) if item != true_objective)
    counterpart["trajectory_type"] = target_type
    counterpart["pair_id"] = str(counterpart.get("pair_id") or f"pair-{counterpart.get('scenario_id', 't2')}")
    counterpart["scenario_id"] = f"{counterpart.get('scenario_id', 'TMCD-V2-T2')}-{target_type}"
    divergence_source = next(
        (
            str(event.get("source_id", ""))
            for event in counterpart.get("event_schedule", [])
            if int(event.get("time", -1)) == divergence
        ),
        "",
    )
    for profile in counterpart.get("source_profiles", []):
        if str(profile.get("source_id", "")) != divergence_source:
            continue
        for segment in profile.get("behavior_schedule", []):
            if int(segment.get("end_time", -1)) < divergence:
                continue
            segment["mode"] = "legitimate_change" if target_type == "legitimate_change" else "deceptive"
            segment["truth_rate"] = 1.0 if target_type == "legitimate_change" else 0.0
            break
    for event in counterpart.get("event_schedule", []):
        if int(event.get("time", -1)) < divergence:
            continue
        event["event_id"] = f"{event.get('event_id', 'event')}-{target_type}"
        if int(event.get("time", -1)) == divergence:
            objective = true_objective if target_type == "legitimate_change" else false_objective
            event["objective_hint"] = objective
            event.setdefault("claim_semantics", {})["object"] = objective
            event["truth_value"] = target_type == "legitimate_change"
            event["is_fake"] = target_type != "legitimate_change"
            event["spoofability"] = 0.25 if target_type == "legitimate_change" else 0.78
            counterpart.setdefault("oracle", {})[
                "first_actionable_event_id"
            ] = event["event_id"]
    metadata = dict(counterpart.get("metadata", {}) or {})
    metadata["trajectory_type"] = target_type
    metadata["paired_counterpart"] = True
    counterpart["metadata"] = metadata
    counterpart["prefix_hash"] = public_prefix_hash(counterpart)
    return counterpart


def validate_scenario_v2(scenario: dict[str, Any]) -> tuple[bool, str]:
    if scenario.get("protocol_version") != PROTOCOL_VERSION or int(scenario.get("schema_version", 0)) != SCHEMA_VERSION:
        return False, "invalid_protocol_or_schema_version"
    required = (
        "scenario_id",
        "scenario_family",
        "distribution",
        "network_context",
        "true_attack",
        "source_profiles",
        "event_schedule",
        "defense_constraints",
        "oracle",
    )
    for key in required:
        if key not in scenario:
            return False, f"missing_{key}"
    metadata = scenario.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        return False, "invalid_metadata"
    privileged = sorted(PRIVILEGED_METADATA_FIELDS & set(metadata))
    if privileged:
        return False, f"privileged_metadata_forbidden:{privileged[0]}"
    if scenario.get("distribution") not in DISTRIBUTIONS:
        return False, "invalid_distribution"
    if scenario.get("true_attack", {}).get("objective") not in OBJECTIVES:
        return False, "invalid_true_objective"
    if any(phase not in PHASES for phase in scenario.get("true_attack", {}).get("phase_schedule", [])):
        return False, "invalid_attack_phase"
    if scenario.get("split") in {"train", "dev", "xplay"} and scenario.get("scenario_family") in OOD_FAMILIES:
        return False, "ood_family_in_training_split"
    if scenario.get("scenario_family") == "trust_betrayal":
        if not str(scenario.get("pair_id", "")).strip():
            return False, "missing_pair_id"
        divergence = scenario.get("divergence_time")
        if isinstance(divergence, bool) or not isinstance(divergence, int) or divergence < 0:
            return False, "invalid_divergence_time"
        if scenario.get("trajectory_type") not in {"betrayal", "legitimate_change"}:
            return False, "invalid_trajectory_type"
        if not str(scenario.get("prefix_hash", "")).strip():
            return False, "missing_prefix_hash"
    source_ids: set[str] = set()
    for profile in scenario.get("source_profiles", []):
        source_id = str(profile.get("source_id", ""))
        if not source_id or source_id in source_ids:
            return False, "invalid_or_duplicate_source"
        source_ids.add(source_id)
        prior = profile.get("public_prior", 0.5)
        if isinstance(prior, bool) or not isinstance(prior, (int, float)) or not 0.0 <= float(prior) <= 1.0:
            return False, "invalid_public_prior"
        for segment in profile.get("behavior_schedule", []):
            if segment.get("mode") not in SOURCE_MODES:
                return False, "invalid_source_mode"
            if int(segment.get("start_time", -1)) < 0 or int(segment.get("end_time", -1)) < int(segment.get("start_time", 0)):
                return False, "invalid_behavior_segment"
    event_ids: set[str] = set()
    event_times: dict[str, int] = {}
    for event in scenario.get("event_schedule", []):
        if not isinstance(event, dict):
            return False, "invalid_raw_event"
        forbidden_signals = sorted(
            RAW_EVENT_FORBIDDEN_SIGNAL_FIELDS & set(event)
        )
        if forbidden_signals:
            return False, f"tool_signal_in_raw_event:{forbidden_signals[0]}"
        extra_fields = sorted(set(event) - RAW_EVENT_ALLOWED_FIELDS)
        if extra_fields:
            return False, f"forbidden_raw_event_field:{extra_fields[0]}"
        event_type = str(event.get("type", "")).strip().lower()
        if not event_type:
            return False, "missing_raw_event_type"
        if (
            event_type.startswith("tool:")
            or event_type.endswith("_result")
            or event_type in RAW_EVENT_RESERVED_TYPES
        ):
            return False, f"tool_result_type_in_raw_event:{event_type}"
        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in event_ids:
            return False, "invalid_or_duplicate_event"
        event_ids.add(event_id)
        event_time = event.get("time", -1)
        if isinstance(event_time, bool) or not isinstance(event_time, int) or event_time < 0:
            return False, "invalid_event_time"
        event_times[event_id] = int(event_time)
        if str(event.get("source_id", "")) not in source_ids:
            return False, "event_source_not_profiled"
        semantics = event.get("claim_semantics", {})
        if any(not str(semantics.get(key, "")).strip() for key in ("entity_id", "predicate", "object", "scope")):
            return False, "missing_claim_semantics"
    oracle = scenario.get("oracle", {}) or {}
    actionable_from = oracle.get("actionable_from")
    actionable_event_id = str(oracle.get("first_actionable_event_id", ""))
    if isinstance(actionable_from, bool) or not isinstance(actionable_from, int) or actionable_from < 0:
        return False, "invalid_or_missing_actionable_from"
    if actionable_event_id not in event_times:
        return False, "invalid_or_missing_first_actionable_event_id"
    if event_times[actionable_event_id] != actionable_from:
        return False, "actionable_time_event_mismatch"
    return True, "ok"


def validate_pair_v2(first: dict[str, Any], second: dict[str, Any]) -> tuple[bool, str]:
    if first.get("pair_id") != second.get("pair_id"):
        return False, "pair_id_mismatch"
    if int(first.get("divergence_time", -1)) != int(second.get("divergence_time", -1)):
        return False, "divergence_time_mismatch"
    if public_prefix_hash(first) != public_prefix_hash(second):
        return False, "public_prefix_mismatch"
    kinds = {str(first.get("trajectory_type")), str(second.get("trajectory_type"))}
    if kinds != {"betrayal", "legitimate_change"}:
        return False, "invalid_pair_types"
    return True, "ok"
