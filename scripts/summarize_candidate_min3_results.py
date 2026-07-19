#!/usr/bin/env python3
"""Aggregate the three-round candidate pilot into a compact auditable result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


KEYS = (
    "safe_utility",
    "safe_success",
    "attack_mitigation",
    "probe_yield",
    "mean_candidate_regret",
    "active_probe_rate",
    "memory_operation_rate",
    "memory_use_rate",
)


def _metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("metrics", payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    baseline_path = args.evaluation_root / "round_0/fixed_active/metrics.json"
    baseline = _metrics(baseline_path)
    rows = []
    input_hashes = {"round_0_fixed": sha256_file(baseline_path)}
    for index in range(1, 4):
        directory = args.evaluation_root / f"round_{index}"
        fixed_path = directory / "fixed_end/metrics.json"
        fresh_start_path = directory / "fresh_start.json"
        fresh_end_path = directory / "fresh_end.json"
        feedback_path = directory / "dca_feedback_gate.json"
        round_gate_path = directory / "round_gate.json"
        training_path = directory / "training_summary.json"
        fixed = _metrics(fixed_path)
        fresh_start = _metrics(fresh_start_path)
        fresh_end = _metrics(fresh_end_path)
        feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
        round_gate = json.loads(round_gate_path.read_text(encoding="utf-8"))
        training = json.loads(training_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "round_index": index,
                "fixed": {key: fixed.get(key) for key in KEYS},
                "delta_safe_utility_vs_baseline": float(fixed["safe_utility"])
                - float(baseline["safe_utility"]),
                "fresh": {
                    "mean_regret_start": fresh_start["mean_teacher_regret"],
                    "mean_regret_end": fresh_end["mean_teacher_regret"],
                    "mean_regret_reduction": float(
                        fresh_start["mean_teacher_regret"]
                    )
                    - float(fresh_end["mean_teacher_regret"]),
                    "top1_start": fresh_start["candidate_top1_accuracy"],
                    "top1_end": fresh_end["candidate_top1_accuracy"],
                },
                "training_loss": training["train_metrics"]["train_loss"],
                "dca_feedback_accepted": feedback["accepted"],
                "dca_feedback_failures": feedback["failures"],
                "round_gate_accepted": round_gate["accepted"],
                "round_gate_failures": round_gate["failures"],
                "promoted": False,
            }
        )
        for label, path in (
            ("fixed", fixed_path),
            ("fresh_start", fresh_start_path),
            ("fresh_end", fresh_end_path),
            ("dca_feedback", feedback_path),
            ("round_gate", round_gate_path),
            ("training", training_path),
        ):
            input_hashes[f"round_{index}_{label}"] = sha256_file(path)
    payload = {
        "schema_version": 1,
        "kind": "candidate_coevolution_min3_result_summary",
        "created_at": utc_now(),
        "baseline": {key: baseline.get(key) for key in KEYS},
        "rounds": rows,
        "round_count": 3,
        "rounds_promoted": 0,
        "active_probe_preserved": all(
            float(row["fixed"]["active_probe_rate"]) > 0.0 for row in rows
        ),
        "defense_learned": any(
            float(row["fixed"]["safe_success"]) > 0.0
            or float(row["fixed"]["attack_mitigation"]) > 0.0
            for row in rows
        ),
        "memory_learned": any(
            float(row["fixed"]["memory_operation_rate"]) > 0.0
            or float(row["fixed"]["memory_use_rate"]) > 0.0
            for row in rows
        ),
        "conclusion": (
            "The pilot preserved active probing and compiler validity but did not "
            "learn successful mitigation, probe yield, or memory use. No checkpoint "
            "is eligible for promotion."
        ),
        "input_hashes": input_hashes,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
