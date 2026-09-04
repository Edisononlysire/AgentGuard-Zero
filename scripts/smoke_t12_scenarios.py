#!/usr/bin/env python3
"""Generate and audit a small CPU-only T1/T2 scenario sample."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.cyber_grounding import (
    cyber_grounded_recovery_group_v2,
)
from agentguard_zero.env.checker import full_check
from agentguard_zero.world.public_projector import assert_public, project_public


FORBIDDEN_PUBLIC_KEYS = {
    "hidden_world",
    "is_fake",
    "oracle",
    "oracle_ledger",
    "task_id",
    "teacher_q",
    "teacher_q_values",
    "true_attack",
    "truth_value",
}


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(map(str, value))
        for item in value.values():
            result.update(_keys(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_keys(item))
        return result
    return set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-per-task", type=int, default=2)
    args = parser.parse_args()
    if args.groups_per_task <= 0:
        parser.error("--groups-per-task must be positive")

    task_counts: Counter[str] = Counter()
    scenario_count = 0
    for task in ("T1", "T2"):
        for group_index in range(args.groups_per_task):
            group = cyber_grounded_recovery_group_v2(task, group_index)
            if len(group) < 2:
                raise RuntimeError(f"{task} group {group_index} is not paired")
            for scenario in group:
                checks = full_check(scenario)
                if not checks.get("all_ok", False):
                    raise RuntimeError(
                        f"invalid scenario {scenario.get('scenario_id')}: {checks}"
                    )
                public = project_public(scenario)
                assert_public(public)
                leaked = sorted(_keys(public) & FORBIDDEN_PUBLIC_KEYS)
                if leaked:
                    raise RuntimeError(
                        f"public projection leaked forbidden keys: {leaked}"
                    )
                task_counts[task] += 1
                scenario_count += 1

    print(
        json.dumps(
            {
                "accepted": True,
                "groups_per_task": args.groups_per_task,
                "scenario_count": scenario_count,
                "scenario_count_by_task": dict(task_counts),
                "hidden_truth_leakage": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
