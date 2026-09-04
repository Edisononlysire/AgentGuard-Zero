#!/usr/bin/env python3
"""Paired replacement gate for the selected T1/T2 expert versus old V9."""

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
CORE = ("safe_success", "attack_mitigation", "attack_success", "intent_accuracy")


def _terminal_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for trace in payload.get("traces", []) or []:
        for execution in trace.get("execution_results", []) or []:
            if not bool(execution.get("terminal", False)):
                continue
            fingerprint = str(execution.get("scenario_fingerprint", ""))
            if not fingerprint or fingerprint in result:
                raise RuntimeError("terminal fingerprints must be unique")
            score = execution.get("terminal_score") or {}
            result[fingerprint] = {
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
    return result


def _mean(rows: list[dict[str, Any]], metric: str) -> float:
    return float(statistics.mean(row[metric] for row in rows)) if rows else 0.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    metrics = (*CORE, "overresponse", "unauthorized_high_impact", "business_cost")
    return {
        "scenario_count": len(rows),
        "overall": {metric: _mean(rows, metric) for metric in metrics},
        "by_task": {
            task: {
                "scenario_count": len(by_task[task]),
                **{metric: _mean(by_task[task], metric) for metric in metrics},
            }
            for task in TASKS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--suite", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    protocol = json.loads(args.training_protocol.read_text())
    selection = json.loads(args.selection.read_text())
    suite_manifest = json.loads(args.suite_manifest.read_text())
    if protocol.get("accepted") is not True or selection.get("accepted") is not True:
        raise RuntimeError("accepted training protocol and Epoch selection are required")
    if sha256_file(args.suite_manifest) != protocol["artifacts"][
        "frozen_suite_manifest_sha256"
    ]:
        raise RuntimeError("frozen suite manifest hash mismatch")

    all_new: list[dict[str, Any]] = []
    all_old: list[dict[str, Any]] = []
    per_suite: dict[str, Any] = {}
    selected_hash = selection["selected_checkpoint_manifest_sha256"]
    old_hash = protocol["artifacts"]["old_v9_manifest_sha256"]
    for specification in args.suite:
        suite_id, new_path_text, old_path_text = specification.split("=", 2)
        new_path, old_path = Path(new_path_text), Path(old_path_text)
        new_payload = json.loads(new_path.read_text())
        old_payload = json.loads(old_path.read_text())
        suite_entry = next(
            row for row in suite_manifest["suites"] if row["suite_id"] == suite_id
        )
        for label, payload, expected_ranker in (
            ("new", new_payload, selected_hash),
            ("old", old_payload, old_hash),
        ):
            if payload.get("ranker_manifest_sha256") != expected_ranker:
                raise RuntimeError(f"{suite_id}/{label} ranker hash mismatch")
            if payload.get("scenario_source_sha256") != suite_entry[
                "scenario_source_sha256"
            ]:
                raise RuntimeError(f"{suite_id}/{label} scenario hash mismatch")
            if (payload.get("metrics") or {}).get("scenario_count") != 200:
                raise RuntimeError(f"{suite_id}/{label} scenario count mismatch")
            if payload.get("dca_frozen") is not True or payload.get("ecrg_enabled") is not False:
                raise RuntimeError(f"{suite_id}/{label} protocol controls failed")
        new_map, old_map = _terminal_map(new_payload), _terminal_map(old_payload)
        if set(new_map) != set(old_map) or len(new_map) != 200:
            raise RuntimeError(f"{suite_id} is not a paired 200-trajectory comparison")
        new_rows = [new_map[key] for key in sorted(new_map)]
        old_rows = [old_map[key] for key in sorted(old_map)]
        all_new.extend(new_rows)
        all_old.extend(old_rows)
        per_suite[suite_id] = {
            "new": _summary(new_rows),
            "old": _summary(old_rows),
            "new_metrics_sha256": sha256_file(new_path),
            "old_metrics_sha256": sha256_file(old_path),
        }

    new_summary, old_summary = _summary(all_new), _summary(all_old)
    eps = 1.0e-12
    checks = {
        "paired_total_600": len(all_new) == len(all_old) == 600,
        "pooled_safe_not_lower": new_summary["overall"]["safe_success"] + eps
        >= old_summary["overall"]["safe_success"],
        "pooled_mitigation_not_lower": new_summary["overall"][
            "attack_mitigation"
        ]
        + eps
        >= old_summary["overall"]["attack_mitigation"],
        "pooled_attack_success_not_higher": new_summary["overall"][
            "attack_success"
        ]
        <= old_summary["overall"]["attack_success"] + eps,
        "pooled_overresponse_not_higher": new_summary["overall"]["overresponse"]
        <= old_summary["overall"]["overresponse"] + eps,
        "pooled_unauthorized_not_higher": new_summary["overall"][
            "unauthorized_high_impact"
        ]
        <= old_summary["overall"]["unauthorized_high_impact"] + eps,
    }
    one_trajectory = 1.0 / 50.0
    for suite_id, result in per_suite.items():
        for task in TASKS:
            for metric in CORE:
                new_value = result["new"]["by_task"][task][metric]
                old_value = result["old"]["by_task"][task][metric]
                if metric == "attack_success":
                    passed = new_value <= old_value + one_trajectory + eps
                else:
                    passed = new_value + one_trajectory + eps >= old_value
                checks[f"{suite_id}_{task}_{metric}_within_one_trajectory"] = passed
            checks[f"{suite_id}_{task}_overresponse_not_higher"] = (
                result["new"]["by_task"][task]["overresponse"]
                <= result["old"]["by_task"][task]["overresponse"] + eps
            )
            checks[f"{suite_id}_{task}_unauthorized_not_higher"] = (
                result["new"]["by_task"][task]["unauthorized_high_impact"]
                <= result["old"]["by_task"][task]["unauthorized_high_impact"] + eps
            )
    accepted = all(checks.values())
    payload = {
        "schema_version": 1,
        "kind": "v11_t12_expert_retention_gate",
        "created_at": utc_now(),
        "accepted": accepted,
        "router_training_authorized": accepted,
        "checks": checks,
        "failures": [key for key, passed in checks.items() if not passed],
        "one_trajectory_tolerance_per_task_per_suite": one_trajectory,
        "new_t12": new_summary,
        "old_v9": old_summary,
        "per_suite": per_suite,
        "selection_sha256": sha256_file(args.selection),
        "training_protocol_sha256": sha256_file(args.training_protocol),
        "suite_manifest_sha256": sha256_file(args.suite_manifest),
        "test_result_used_for_retraining": False,
        "formal_three_rounds_started": False,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
