#!/usr/bin/env python3
"""Select one immutable T1/T2 checkpoint using only the frozen 400-trajectory Dev."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


TASKS = ("T1", "T2")
TERMINAL_METRICS = (
    "safe_success",
    "attack_mitigation",
    "attack_success",
    "intent_accuracy",
    "overresponse",
    "unauthorized_high_impact",
    "business_cost",
)


def _terminal_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in payload.get("traces", []) or []:
        for execution in trace.get("execution_results", []) or []:
            if not bool(execution.get("terminal", False)):
                continue
            fingerprint = str(execution.get("scenario_fingerprint", ""))
            if not fingerprint or fingerprint in seen:
                raise RuntimeError("Dev terminal scenario fingerprints must be unique")
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
                    "overresponse": float(
                        float(score.get("overresponse_cost", 0.0)) > 0.0
                        or bool(score.get("overresponse", False))
                    ),
                    "unauthorized_high_impact": float(
                        float(score.get("unauthorized_high_impact", 0.0)) > 0.0
                    ),
                    "business_cost": float(score.get("business_cost", 0.0)),
                }
            )
    return rows


def _mean(rows: list[dict[str, Any]], name: str) -> float:
    return (
        float(statistics.mean(float(row[name]) for row in rows))
        if rows
        else 0.0
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task_rows[str(row["task_id"])].append(row)
    by_task = {
        task: {
            "scenario_count": len(by_task_rows[task]),
            **{
                metric: _mean(by_task_rows[task], metric)
                for metric in TERMINAL_METRICS
            },
        }
        for task in TASKS
    }
    overall = {metric: _mean(rows, metric) for metric in TERMINAL_METRICS}
    macro_safe_success = statistics.mean(
        by_task[task]["safe_success"] for task in TASKS
    )
    return {
        "scenario_count": len(rows),
        "scenario_count_by_task": {
            task: len(by_task_rows[task]) for task in TASKS
        },
        "t12_macro_safe_success": float(macro_safe_success),
        "overall": overall,
        "by_task": by_task,
    }


def _selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    overall = summary["overall"]
    return (
        -float(summary["t12_macro_safe_success"]),
        -float(overall["attack_mitigation"]),
        float(overall["attack_success"]),
        -float(overall["intent_accuracy"]),
        float(overall["overresponse"]),
        float(overall["business_cost"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--dev-suite-manifest", type=Path, required=True)
    parser.add_argument("--epoch", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    protocol = json.loads(args.training_protocol.read_text(encoding="utf-8"))
    dev_suite = json.loads(args.dev_suite_manifest.read_text(encoding="utf-8"))
    if protocol.get("kind") != "v11_t12_expert_training_protocol":
        raise RuntimeError("invalid frozen training protocol")
    if protocol.get("accepted") is not True:
        raise RuntimeError("training protocol is not accepted")
    if dev_suite.get("accepted") is not True:
        raise RuntimeError("Epoch-selection Dev suite is not accepted")
    if sha256_file(args.dev_suite_manifest) != (
        protocol["artifacts"]["dev_scenarios_manifest_sha256"]
    ):
        raise RuntimeError("Epoch-selection Dev suite hash mismatch")

    epochs: list[dict[str, Any]] = []
    for specification in args.epoch:
        parts = specification.split("=", 2)
        if len(parts) != 3:
            raise ValueError("--epoch must be INDEX=CHECKPOINT_MANIFEST=ROLLOUT")
        index = int(parts[0])
        checkpoint = Path(parts[1])
        rollout = Path(parts[2])
        checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        rollout_payload = json.loads(rollout.read_text(encoding="utf-8"))
        rows = _terminal_rows(rollout_payload)
        summary = _summary(rows)
        checks = {
            "epoch_index": checkpoint_payload.get("epoch_index") == index,
            "scenario_exact_400": summary["scenario_count"] == 400,
            "task_exact": summary["scenario_count_by_task"]
            == {"T1": 200, "T2": 200},
            "ranker_binding": rollout_payload.get("ranker_manifest_sha256")
            == sha256_file(checkpoint),
            "scenario_binding": rollout_payload.get("scenario_source_sha256")
            == dev_suite.get("scenario_source_sha256"),
            "protocol_role": rollout_payload.get("protocol_role")
            == "epoch_selection_dev",
            "dca_frozen": rollout_payload.get("dca_frozen") is True,
            "ecrg_disabled": rollout_payload.get("ecrg_enabled") is False,
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"Epoch {index} Dev audit failed: "
                + ",".join(key for key, passed in checks.items() if not passed)
            )
        epochs.append(
            {
                "epoch": index,
                "checkpoint_manifest": str(checkpoint.resolve()),
                "checkpoint_manifest_sha256": sha256_file(checkpoint),
                "rollout": str(rollout.resolve()),
                "rollout_sha256": sha256_file(rollout),
                "checks": checks,
                "metrics": summary,
                "selection_key": list(_selection_key(summary)),
            }
        )
    if sorted(row["epoch"] for row in epochs) != [1, 2, 3, 4]:
        raise RuntimeError("exactly Epochs 1-4 are required")
    selected = min(epochs, key=lambda row: tuple(row["selection_key"]))
    payload = {
        "schema_version": 1,
        "kind": "v11_t12_epoch_selection",
        "created_at": utc_now(),
        "accepted": True,
        "selection_source": "frozen_t12_epoch_selection_dev_400_only",
        "formal_frozen_600_inspected": False,
        "lexicographic_order": protocol["epoch_selection"]["lexicographic"],
        "selected_epoch": selected["epoch"],
        "selected_checkpoint_manifest": selected["checkpoint_manifest"],
        "selected_checkpoint_manifest_sha256": selected[
            "checkpoint_manifest_sha256"
        ],
        "selected_metrics": selected["metrics"],
        "training_protocol_sha256": sha256_file(args.training_protocol),
        "dev_suite_manifest_sha256": sha256_file(args.dev_suite_manifest),
        "epochs": sorted(epochs, key=lambda row: row["epoch"]),
        "router_training_started": False,
        "formal_three_rounds_started": False,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
