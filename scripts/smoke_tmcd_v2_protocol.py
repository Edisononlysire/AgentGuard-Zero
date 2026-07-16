#!/usr/bin/env python3
"""Run a deterministic 256-scenario CPU gate for TMCD Protocol v2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.env.checker import full_check
from agentguard_zero.env.cyber_env_v2 import CyberDefenseEnvV2
from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.protocol import TASK_FAMILY_MAP
from agentguard_zero.schemas.action_schema_v4 import DEFAULT_ACTION_PACKET_V4, validate_action_packet_v4
from agentguard_zero.schemas.scenario_schema_v2 import (
    paired_counterpart_v2,
    minimal_example_v2,
    public_prefix_hash,
    validate_pair_v2,
)
from agentguard_zero.training.coevolution import atomic_write_json, scenario_fingerprint, utc_now
from agentguard_zero.world.public_projector import assert_public


TASK_FAMILIES = TASK_FAMILY_MAP


def _rename_sources(scenario: dict[str, Any], suffix: str) -> None:
    mapping = {
        str(profile["source_id"]): f"{profile['source_id']}-{suffix}"
        for profile in scenario.get("source_profiles", [])
    }
    for profile in scenario.get("source_profiles", []):
        profile["source_id"] = mapping[str(profile["source_id"])]
    for event in scenario.get("event_schedule", []):
        old = str(event.get("source_id", ""))
        event["source_id"] = mapping.get(old, old)
        event["source"] = mapping.get(old, old)


def make_scenario(task_id: str, index: int) -> dict[str, Any]:
    scenario = minimal_example_v2(trajectory_type="betrayal")
    suffix = f"{task_id.lower()}-{index:03d}"
    scenario["scenario_id"] = f"TMCD-V2-SMOKE-{task_id}-{index:03d}"
    scenario["scenario_family"] = TASK_FAMILIES[task_id]
    scenario["split"] = "smoke"
    scenario["metadata"] = {
        "task_id": task_id,
        "manipulation_family": TASK_FAMILIES[task_id],
        "smoke_index": index,
        "experiment_variant": ("full", "append_only_memory", "no_state_layer")[index % 3],
    }
    _rename_sources(scenario, suffix)
    actionable_event_id = str(
        scenario.get("oracle", {}).get("first_actionable_event_id", "")
    )
    renamed_events: dict[str, str] = {}
    for event_index, event in enumerate(scenario.get("event_schedule", [])):
        old_event_id = str(event.get("event_id", ""))
        event["event_id"] = f"event-{suffix}-{event_index}"
        renamed_events[old_event_id] = event["event_id"]
        event["entity_id"] = "database"
        event["claim_semantics"]["entity_id"] = "database"
    if actionable_event_id in renamed_events:
        scenario["oracle"]["first_actionable_event_id"] = renamed_events[
            actionable_event_id
        ]

    if task_id == "T1":
        for key in ("pair_id", "prefix_hash", "divergence_time", "trajectory_type"):
            scenario.pop(key, None)
        scenario["event_schedule"][1]["truth_value"] = False
        scenario["event_schedule"][1]["is_fake"] = True
        scenario["event_schedule"][1]["spoofability"] = 0.55 + 0.01 * (index % 20)
        scenario["oracle"]["actionable_from"] = int(
            scenario["event_schedule"][1]["time"]
        )
        scenario["oracle"]["first_actionable_event_id"] = scenario[
            "event_schedule"
        ][1]["event_id"]
    elif task_id == "T2":
        scenario["pair_id"] = f"pair-smoke-{index // 2:03d}"
        scenario["trajectory_type"] = "betrayal"
        scenario["prefix_hash"] = public_prefix_hash(scenario)
    elif task_id == "T3":
        for key in ("pair_id", "prefix_hash", "divergence_time", "trajectory_type"):
            scenario.pop(key, None)
        second = copy.deepcopy(scenario["event_schedule"][2])
        second["event_id"] = f"event-{suffix}-poison-2"
        second["time"] = 1
        second["claim_semantics"]["predicate"] = "asset_risk_level"
        second["claim_semantics"]["object"] = "low"
        second["truth_value"] = False
        second["is_fake"] = True
        scenario["event_schedule"].insert(2, second)
        scenario["oracle"]["actionable_from"] = int(second["time"])
        scenario["oracle"]["first_actionable_event_id"] = second["event_id"]
    else:
        for key in ("pair_id", "prefix_hash", "divergence_time", "trajectory_type"):
            scenario.pop(key, None)
        scenario["defense_constraints"]["business_budget"] = 0.8 + 0.05 * (index % 4)
        scenario["event_schedule"][2]["truth_value"] = False
        scenario["event_schedule"][2]["is_fake"] = True
        scenario["event_schedule"][2]["claim"] = "urgent isolation requested for critical database"
    return scenario


def _belief_from_public(event: dict[str, Any]) -> dict[str, float]:
    objective = str(event.get("objective_hint", "exfiltration"))
    values = {key: 0.05 for key in ("exfiltration", "sabotage", "persistence", "credential_theft")}
    if objective not in values:
        objective = "exfiltration"
    values[objective] = 0.85
    return values


def public_policy(observation: dict[str, Any], *, variant: str, turn: int) -> dict[str, Any]:
    packet = copy.deepcopy(DEFAULT_ACTION_PACKET_V4)
    events = observation.get("observed_events", []) or []
    if not events:
        return packet
    event = events[0]
    event_id = str(event.get("event_id", ""))
    source_id = str(event.get("source_id") or event.get("source") or "unknown")
    evidence_id = str(event.get("evidence_id", ""))
    refs = [evidence_id] if evidence_id else []
    packet["belief"] = _belief_from_public(event)
    packet["evidence_assessment"] = [
        {"event_id": event_id, "status": "challenged", "suspected_poisoning": True}
    ]
    packet["trust_operations"] = [
        {
            "op": "challenge",
            "source_id": source_id,
            "event_id": event_id,
            "evidence_refs": refs,
        }
    ]
    packet["memory_operations"] = [
        {
            "op": "ingest",
            "claim": copy.deepcopy(event.get("claim_semantics", {})),
            "source_ids": [source_id],
            "evidence_refs": refs,
            "target_status": "confirmed" if variant == "append_only_memory" and turn % 2 else "quarantined",
        }
    ]
    packet["tool_call"] = {"name": "SourceChallenge", "args": {"event_id": event_id}}
    packet["uncertainty"] = 0.65
    packet["safety_check"] = {
        "business_risk": 0.2,
        "overresponse_risk": 0.2,
        "justification": "public evidence probe",
    }
    packet["response"] = {
        "tier": "L1",
        "action": "LimitSession" if turn >= 2 else "Observe",
        "target": str(event.get("entity_id", "none")),
    }
    valid, reason = validate_action_packet_v4(packet)
    if not valid:
        raise AssertionError(f"invalid smoke action: {reason}")
    return packet


def run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    checks = full_check(scenario)
    if not checks.get("all_ok", False):
        raise AssertionError(f"hard gate failed: {scenario['scenario_id']} {checks}")
    env = CyberDefenseEnvV2(scenario, max_steps=4)
    variant = str(scenario.get("metadata", {}).get("experiment_variant", "full"))
    done = False
    observations = 0
    while not done:
        observation = env.observe()
        observations += 1
        assert_public(observation)
        if "reward" in json.dumps(observation, ensure_ascii=True):
            raise AssertionError("reward leaked into observation")
        action = public_policy(observation, variant=variant, turn=env.t)
        next_observation, tool_result, done = env.step(action)
        assert_public(tool_result)
        assert_public(next_observation)
    score = score_trajectory_v2(env)
    if not math.isfinite(float(score["reward"])):
        raise AssertionError("non-finite trajectory reward")
    assert_public(env.history)
    return {
        "scenario_id": scenario["scenario_id"],
        "task_id": scenario["metadata"]["task_id"],
        "variant": variant,
        "fingerprint": scenario_fingerprint(scenario),
        "steps": len(env.history),
        "observations": observations,
        "reward": float(score["reward"]),
        "attack_mitigated": bool(score["attack_mitigated"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.count % 4:
        raise SystemExit("--count must be a positive multiple of four")
    per_task = args.count // 4
    scenarios: list[dict[str, Any]] = []
    for task_id in TASK_FAMILIES:
        for index in range(per_task):
            scenarios.append(make_scenario(task_id, index))

    # Every T2 betrayal branch must have a valid legitimate-change pair.
    pair_checks = 0
    for scenario in [item for item in scenarios if item["metadata"]["task_id"] == "T2"]:
        counterpart = paired_counterpart_v2(scenario)
        ok, reason = validate_pair_v2(scenario, counterpart)
        if not ok:
            raise AssertionError(f"T2 pair gate failed: {reason}")
        pair_checks += 1

    results = [run_scenario(scenario) for scenario in scenarios]
    fingerprints = [item["fingerprint"] for item in results]
    if len(fingerprints) != len(set(fingerprints)):
        raise AssertionError("smoke scenarios are not unique")
    report = {
        "protocol_version": "tmcd-v2",
        "kind": "protocol_smoke_report",
        "created_at": utc_now(),
        "count": len(results),
        "pair_checks": pair_checks,
        "task_counts": dict(Counter(item["task_id"] for item in results)),
        "variant_counts": dict(Counter(item["variant"] for item in results)),
        "mean_reward": sum(item["reward"] for item in results) / len(results),
        "result_digest": hashlib.sha256(
            json.dumps(results, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "pid": os.getpid(),
    }
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
