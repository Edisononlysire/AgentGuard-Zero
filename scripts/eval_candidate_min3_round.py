#!/usr/bin/env python3
"""Evaluate fixed-set safety, fresh curriculum gain, retention, and rollback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.gates import evaluate_round_gate
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def _metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("metrics", payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-start", type=Path, required=True)
    parser.add_argument("--fixed-end", type=Path, required=True)
    parser.add_argument("--fresh-start", type=Path, required=True)
    parser.add_argument("--fresh-end", type=Path, required=True)
    parser.add_argument("--retention", type=Path)
    parser.add_argument("--retention-start", type=Path)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    fixed_start = _metrics(args.fixed_start)
    fixed_end = _metrics(args.fixed_end)
    fresh_start = _metrics(args.fresh_start)
    fresh_end = _metrics(args.fresh_end)
    fixed_verdict = evaluate_round_gate(
        fixed_start, fixed_end, round_index=args.round_index
    )
    fresh_regret_reduction = float(fresh_start["mean_teacher_regret"]) - float(
        fresh_end["mean_teacher_regret"]
    )
    fresh_top1_gain = float(fresh_end["candidate_top1_accuracy"]) - float(
        fresh_start["candidate_top1_accuracy"]
    )
    fresh_family_gain = float(fresh_end["action_family_accuracy"]) - float(
        fresh_start["action_family_accuracy"]
    )
    fresh_improved = fresh_regret_reduction >= 0.001
    probe_preserved = float(fixed_end.get("active_probe_rate", 0.0)) > 0.0
    failures = list(fixed_verdict.failures)
    if not fresh_improved:
        failures.append("fresh_curriculum_not_improved")
    if not probe_preserved:
        failures.append("active_probe_skill_zero")
    retention_metrics = _metrics(args.retention) if args.retention else None
    retention_start_metrics = (
        _metrics(args.retention_start) if args.retention_start else None
    )
    retention_regret_delta = None
    if retention_metrics is not None and retention_start_metrics is not None:
        retention_regret_delta = float(retention_metrics["mean_teacher_regret"]) - float(
            retention_start_metrics["mean_teacher_regret"]
        )
        if retention_regret_delta > 0.01:
            failures.append("previous_round_regret_regressed")
    payload = {
        "schema_version": 1,
        "kind": "candidate_min3_round_gate",
        "created_at": utc_now(),
        "round_index": args.round_index,
        "accepted": not failures,
        "rollback_required": bool(failures),
        "failures": failures,
        "fixed_verdict": fixed_verdict.to_dict(),
        "fresh_improvement": {
            "mean_teacher_regret_reduction": fresh_regret_reduction,
            "candidate_top1_gain": fresh_top1_gain,
            "action_family_gain": fresh_family_gain,
            "improved": fresh_improved,
        },
        "active_probe_preserved": probe_preserved,
        "retention": {
            "start": retention_start_metrics,
            "end": retention_metrics,
            "mean_teacher_regret_delta": retention_regret_delta,
        }
        if retention_metrics is not None
        else None,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (
                ("fixed_start", args.fixed_start),
                ("fixed_end", args.fixed_end),
                ("fresh_start", args.fresh_start),
                ("fresh_end", args.fresh_end),
            )
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
