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
    for event in scenario.get("event_schedule", []):
        public_event = project_event(event)
        leaked = forbidden_public_paths(public_event)
        if leaked:
            return False, f"public_projection_leak:{leaked[0]}"
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
