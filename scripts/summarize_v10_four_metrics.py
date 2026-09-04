#!/usr/bin/env python3
"""Summarize the four primary terminal metrics from a merged rollout."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


TASKS = ("T1", "T2", "T3", "T4")
METRICS = (
    "safe_success",
    "attack_mitigation",
    "attack_success",
    "intent_accuracy",
)


def terminal_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for trace in payload.get("traces", []) or []:
        for execution in trace.get("execution_results", []) or []:
            if not bool(execution.get("terminal", False)):
                continue
            fingerprint = str(execution.get("scenario_fingerprint", ""))
            if not fingerprint or fingerprint in seen:
                raise RuntimeError("terminal scenario fingerprints must be unique")
            seen.add(fingerprint)
            score = execution.get("terminal_score") or {}
            rows.append(
                {
                    "task_id": str(
                        execution.get("terminal_task_id", trace.get("task_id", ""))
                    ),
                    "safe_success": float(bool(score.get("safe_success", False))),
                    "attack_mitigation": float(
                        bool(score.get("attack_mitigated", False))
                    ),
                    "attack_success": float(bool(score.get("attack_success", False))),
                    "intent_accuracy": float(bool(score.get("correct_intent", False))),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("rollout has no terminal trajectories")

    def metrics(group: list[dict[str, Any]]) -> dict[str, float]:
        return {
            name: float(statistics.mean(float(row[name]) for row in group))
            if group
            else 0.0
            for name in METRICS
        }

    by_task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task_rows[str(row["task_id"])].append(row)
    return {
        "scenario_count": len(rows),
        "overall": metrics(rows),
        "by_task": {task: metrics(by_task_rows[task]) for task in TASKS},
        "scenario_count_by_task": {
            task: len(by_task_rows[task]) for task in TASKS
        },
        "direction": {
            "safe_success": "higher_is_better",
            "attack_mitigation": "higher_is_better",
            "attack_success": "lower_is_better",
            "intent_accuracy": "higher_is_better",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite " + str(args.output))
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "kind": "v10_four_primary_trajectory_metrics",
        "source": str(args.input.resolve()),
        **summarize(terminal_rows(payload)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
