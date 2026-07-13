from __future__ import annotations

import json
from collections import deque
from typing import Any

from agentguard_zero.schemas.scenario_schema_v2 import (
    OOD_FAMILIES,
    TRAIN_FAMILIES,
    public_prefix_hash,
    validate_scenario_v2,
)
from agentguard_zero.world.public_projector import forbidden_public_paths, project_event


def _reachable(edges: list[Any], source: str, target: str) -> bool:
    graph: dict[str, list[str]] = {}
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            graph.setdefault(str(edge[0]), []).append(str(edge[1]))
    queue = deque([source])
    seen = {source}
    while queue:
        current = queue.popleft()
        if current == target:
            return True
        for neighbour in graph.get(current, []):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return False


def format_checker_v2(scenario: dict[str, Any]) -> tuple[bool, str]:
    return validate_scenario_v2(scenario)


def validity_checker_v2(scenario: dict[str, Any]) -> tuple[bool, str]:
    valid, reason = validate_scenario_v2(scenario)
    if not valid:
        return valid, reason
    family = str(scenario.get("scenario_family", ""))
    distribution = str(scenario.get("distribution", ""))
    split = str(scenario.get("split", ""))
    if split in {"train", "dev", "xplay"} and family not in TRAIN_FAMILIES:
        return False, "non_training_family_in_training_split"
    if distribution == "ood" and family not in OOD_FAMILIES:
        return False, "ood_distribution_requires_frozen_ood_family"
    if family in OOD_FAMILIES and distribution != "ood":
        return False, "ood_family_wrong_distribution"
    if family == "trust_betrayal":
        required = ("pair_id", "divergence_time", "trajectory_type")
        if any(not str(scenario.get(key, "")).strip() for key in required):
            return False, "t2_pair_metadata_missing"
        if scenario.get("trajectory_type") not in {"betrayal", "legitimate_change"}:
            return False, "invalid_t2_trajectory_type"
        declared = str(scenario.get("prefix_hash", ""))
        if declared and declared != public_prefix_hash(scenario):
            return False, "stale_prefix_hash"
        divergence = int(scenario.get("divergence_time", -1))
        prefix_events = [
            item for item in scenario.get("event_schedule", [])
            if int(item.get("time", -1)) < divergence
        ]
        if len(prefix_events) < 2 or any(not bool(item.get("truth_value", True)) for item in prefix_events):
            return False, "t2_requires_two_truthful_prefix_events"
        divergence_events = [
            item for item in scenario.get("event_schedule", [])
            if int(item.get("time", -1)) == divergence
        ]
        if not divergence_events:
            return False, "t2_missing_divergence_event"
        expected_truth = scenario.get("trajectory_type") == "legitimate_change"
        if all(bool(item.get("truth_value", True)) != expected_truth for item in divergence_events):
            return False, "t2_divergence_truth_mismatch"
    for event in scenario.get("event_schedule", []):
        public_event = project_event(event)
        leaked = forbidden_public_paths(public_event)
        if leaked:
            return False, f"public_projection_leak:{leaked[0]}"
    events = scenario.get("event_schedule", []) or []
    false_events = [item for item in events if not bool(item.get("truth_value", True))]
    if family == "active_probe_ambiguity" and not false_events:
        return False, "t1_requires_ambiguous_manipulated_event"
    if family == "profile_poisoning":
        false_claims = {
            json.dumps(item.get("claim_semantics", {}), sort_keys=True, separators=(",", ":"))
            for item in false_events
        }
        if len(false_claims) < 2:
            return False, "t3_requires_two_distinct_poison_claims"
        first_false = min(int(item.get("time", 0)) for item in false_events)
        if not any(bool(item.get("truth_value", True)) and int(item.get("time", 0)) > first_false for item in events):
            return False, "t3_requires_later_clean_decision_phase"
    if family == "business_overresponse":
        target = str(scenario.get("true_attack", {}).get("target", ""))
        criticality = max(
            (
                float(item.get("criticality", 0.0))
                for item in scenario.get("network_context", {}).get("assets", [])
                if str(item.get("id", "")) == target
            ),
            default=0.0,
        )
        if criticality < 0.8 or not false_events:
            return False, "t4_requires_false_pressure_on_critical_target"
    return True, "ok"


def solvability_checker_v2(scenario: dict[str, Any]) -> tuple[bool, str]:
    valid, reason = validity_checker_v2(scenario)
    if not valid:
        return valid, reason
    assets = {
        str(item.get("id"))
        for item in scenario.get("network_context", {}).get("assets", [])
        if isinstance(item, dict)
    }
    attack = scenario.get("true_attack", {})
    entry = str(attack.get("entry", ""))
    target = str(attack.get("target", ""))
    if entry not in assets or target not in assets:
        return False, "entry_or_target_not_in_assets"
    if not _reachable(scenario.get("network_context", {}).get("reachable_edges", []), entry, target):
        return False, "attack_path_not_reachable"
    constraints = scenario.get("defense_constraints", {})
    if float(constraints.get("business_budget", 0.0)) <= 0.0:
        return False, "invalid_business_budget"
    if int(constraints.get("verification_budget", 0)) < 1:
        return False, "invalid_verification_budget"
    events = scenario.get("event_schedule", []) or []
    if not events:
        return False, "empty_event_schedule"
    sources = {str(item.get("source_id", "")) for item in events}
    if scenario.get("distribution") != "clean" and len(sources) < 2:
        return False, "no_independent_public_source"
    false_times = [int(item.get("time", 0)) for item in events if not bool(item.get("truth_value", True))]
    if false_times:
        first_false = min(false_times)
        later_public = [item for item in events if int(item.get("time", 0)) > first_false]
        if not later_public:
            return False, "no_post_manipulation_discriminating_observation"
    return True, "ok"


def safety_checker_v2(scenario: dict[str, Any]) -> tuple[bool, str]:
    text = json.dumps(scenario, ensure_ascii=True).lower()
    banned = (
        "shellcode",
        "exploit code",
        "malware binary",
        "ransomware binary",
        "reverse shell",
        "credential dump command",
        "real ip",
    )
    for token in banned:
        if token in text:
            return False, f"unsafe_content:{token}"
    return True, "ok"


def full_check_v2(scenario: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, checker in (
        ("format", format_checker_v2),
        ("valid", validity_checker_v2),
        ("solvable", solvability_checker_v2),
        ("safe", safety_checker_v2),
    ):
        try:
            ok, message = checker(scenario)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            ok, message = False, f"checker_exception:{type(exc).__name__}"
        checks[name] = {"ok": bool(ok), "message": message}
    checks["all_ok"] = all(checks[name]["ok"] for name in ("format", "valid", "solvable", "safe"))
    return checks
