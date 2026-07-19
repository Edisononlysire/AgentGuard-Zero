#!/usr/bin/env python3
"""Gate T1-T4 on Full-Teacher utility versus explicit skill ablations."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.canonical_scenarios import canonical_recovery_group
from agentguard_zero.recovery.public_teacher import (
    PublicStateRobustTeacher,
    public_state_digest,
)
from agentguard_zero.training.coevolution import atomic_write_json, utc_now


SKILL_BY_TASK = {
    "T1": "active_probe",
    "T2": "trust",
    "T3": "memory",
    "T4": "business_response",
}


def _uses_skill(packet: dict[str, Any], category: str, skill: str) -> bool:
    tool = str((packet.get("tool_call") or {}).get("name", ""))
    if skill == "active_probe":
        return category == "active_probe"
    if skill == "trust":
        return bool(packet.get("trust_operations"))
    if skill == "memory":
        return bool(packet.get("memory_operations") or packet.get("memory_usage"))
    return tool in {"BusinessImpactEstimator", "ShadowActionProbe"}


def identify(task_id: str, group_index: int, *, skill: str) -> dict[str, float]:
    worlds = [
        instantiate_scenario(copy.deepcopy(row))
        for row in canonical_recovery_group(task_id, group_index)
    ]
    full_teacher = PublicStateRobustTeacher(beam_width=16, max_candidates=64)
    no_skill_teacher = PublicStateRobustTeacher(
        beam_width=16,
        max_candidates=64,
        disabled_skills=(skill,),
    )
    for step in range(min(env.max_steps for env in worlds)):
        full = full_teacher.decide(worlds, horizon=3, enforce_min_worlds=True)
        if _uses_skill(full.selected_packet, full.selected_category, skill):
            ablated = no_skill_teacher.decide(
                worlds, horizon=3, enforce_min_worlds=True
            )
            return {
                "full_robust_utility": full.robust_value,
                "no_skill_robust_utility": ablated.robust_value,
                "utility_gap": full.robust_value - ablated.robust_value,
                "decision_step": float(step),
                "world_count": float(len(worlds)),
            }
        for env in worlds:
            env.step(copy.deepcopy(full.selected_packet))
        if any(env.attack_mitigated or env.attack_success for env in worlds):
            break
        if len({public_state_digest(env.observe()) for env in worlds}) != 1:
            break
    return {
        "full_robust_utility": -1.0e9,
        "no_skill_robust_utility": -1.0e9,
        "utility_gap": -1.0e9,
        "decision_step": -1.0,
        "world_count": float(len(worlds)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-task", type=int, default=1)
    parser.add_argument("--group-offset", type=int, default=30000)
    parser.add_argument("--utility-gap", type=float, default=0.10)
    parser.add_argument("--safe-success-gap", type=float, default=0.10)
    args = parser.parse_args()

    tasks: dict[str, Any] = {}
    for task_offset, (task_id, skill) in enumerate(SKILL_BY_TASK.items()):
        rows = []
        for index in range(args.groups_per_task):
            group_index = args.group_offset + task_offset * 1000 + index
            rows.append(identify(task_id, group_index, skill=skill))
        utility_gap = mean(row["utility_gap"] for row in rows)
        tasks[task_id] = {
            "skill": skill,
            "diagnostics": rows,
            "core_utility_gap": utility_gap,
            "safe_success_gap": None,
            "accepted": utility_gap >= args.utility_gap,
        }
    payload = {
        "schema_version": 1,
        "kind": "candidate_skill_identifiability_gate",
        "created_at": utc_now(),
        "thresholds": {
            "core_utility_gap": args.utility_gap,
            "safe_success_gap": args.safe_success_gap,
            "rule": "same-state H=3 robust utility gap; terminal safe-success gap optional",
        },
        "tasks": tasks,
        "accepted": all(row["accepted"] for row in tasks.values()),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
