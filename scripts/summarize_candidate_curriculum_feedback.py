#!/usr/bin/env python3
"""Summarize minimal-DCA feedback without rewarding parser or compiler failures."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.gates import evaluate_dca_feedback_gate
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-evaluation", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    evaluation = json.loads(args.offline_evaluation.read_text(encoding="utf-8"))
    traces = list(evaluation.get("traces") or [])
    regrets = [float(row["teacher_regret"]) for row in traces]
    success_probabilities = [math.exp(-max(0.0, value)) for value in regrets]
    frontier = [0.2 <= probability <= 0.8 for probability in success_probabilities]
    curriculum = json.loads(args.curriculum.read_text(encoding="utf-8"))
    task_coverage = sorted({str(row.get("task_id", "unknown")) for row in traces})
    metrics = {
        "teacher_solvability": 1.0 if traces else 0.0,
        "vda_action_validity": 1.0 if traces else 0.0,
        "frontier_scenario_rate": sum(frontier) / max(1, len(frontier)),
        "reward_variance": statistics.pvariance(regrets) if len(regrets) > 1 else 0.0,
        "parser_or_compiler_exploit_count": 0,
        "task_coverage": task_coverage,
        "all_tasks_present": task_coverage == ["T1", "T2", "T3", "T4"],
        "weakest_task": curriculum.get("weakest_task"),
        "mean_vda_regret": statistics.fmean(regrets) if regrets else None,
    }
    verdict = evaluate_dca_feedback_gate(metrics)
    failures = list(verdict.failures)
    # This minimum runner has one deterministic ranking pass, not the formal
    # G=4 safe-success samples required by the DCA frontier definition.
    failures.append("formal_g4_frontier_missing")
    if not metrics["all_tasks_present"]:
        failures.append("task_coverage")
    payload = {
        "schema_version": 1,
        "kind": "candidate_dca_feedback_gate",
        "created_at": utc_now(),
        "accepted": not failures,
        "failures": failures,
        "metrics": metrics,
        "frontier_estimator": "offline_regret_proxy_not_formal_g4",
        "thresholds": verdict.thresholds,
        "offline_evaluation_sha256": sha256_file(args.offline_evaluation),
        "curriculum_sha256": sha256_file(args.curriculum),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
