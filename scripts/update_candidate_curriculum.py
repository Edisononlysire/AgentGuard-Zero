#!/usr/bin/env python3
"""Update the minimal DCA curriculum policy from candidate-VDA skill gaps."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now
from agentguard_zero.candidate.gates import evaluate_vda_gate_a


TASKS = ("T1", "T2", "T3", "T4")
SKILL_METRICS = {
    "T1": ("active_probe_rate",),
    "T2": ("trust_rate",),
    "T3": ("memory_use_rate", "memory_operation_rate"),
    "T4": ("mitigation_rate", "passive_verification_rate"),
}


def _payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value.get("metrics", value))


def _softmax(values: dict[str, float], temperature: float) -> dict[str, float]:
    maximum = max(values.values())
    weights = {
        key: math.exp((value - maximum) / temperature) for key, value in values.items()
    }
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.35)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.groups < len(TASKS):
        raise ValueError("groups must preserve at least one T1-T4 scenario each")

    metrics = _payload(args.evaluation)
    formal_gate = evaluate_vda_gate_a(metrics)
    by_task = metrics.get("by_task") or {}
    terminal = metrics.get("by_task_terminal") or {}
    gaps: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for task_id in TASKS:
        action = dict(by_task.get(task_id) or {})
        outcome = dict(terminal.get(task_id) or {})
        regret = action.get("mean_candidate_regret", action.get("mean_teacher_regret", 1.0))
        regret = 1.0 if not isinstance(regret, (int, float)) else min(1.0, max(0.0, float(regret) / 2.0))
        skill_rate = max(float(action.get(key, 0.0)) for key in SKILL_METRICS[task_id])
        safe_success = float(outcome.get("safe_success", 0.0))
        gap = 0.45 * regret + 0.35 * (1.0 - skill_rate) + 0.20 * (1.0 - safe_success)
        gaps[task_id] = gap
        details[task_id] = {
            "normalized_regret": regret,
            "required_skill_rate": skill_rate,
            "safe_success": safe_success,
            "gap": gap,
        }
    weights = _softmax(gaps, args.temperature)
    rng = random.Random(args.seed + args.round_index)
    schedule = list(TASKS)
    population = list(TASKS)
    cumulative = []
    running = 0.0
    for task_id in population:
        running += weights[task_id]
        cumulative.append(running)
    while len(schedule) < args.groups:
        draw = rng.random()
        schedule.append(
            next(task for task, boundary in zip(population, cumulative) if draw <= boundary)
        )
    rng.shuffle(schedule)
    payload = {
        "schema_version": 1,
        "kind": "candidate_dca_curriculum_policy",
        "created_at": utc_now(),
        "round_index": args.round_index,
        "source_evaluation_sha256": sha256_file(args.evaluation),
        "action_contract_admitted": (
            float(metrics.get("action_validity", 1.0)) >= 0.995
            and float(metrics.get("invalid_noop_rate", 0.0)) <= 0.005
        ),
        "source_capability_admitted": formal_gate.accepted,
        "source_capability_failures": list(formal_gate.failures),
        "accepted": formal_gate.accepted,
        "skill_gap": details,
        "sampling_weights": weights,
        "task_schedule": schedule,
        "weakest_task": max(gaps, key=lambda task: (gaps[task], task)),
        "seed": args.seed + args.round_index,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["source_capability_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
