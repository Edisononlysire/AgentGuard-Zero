#!/usr/bin/env python3
"""Cap repeated exact targets so structured SFT cannot learn one modal packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.bootstrap_data import audit_bootstrap_records
from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION
from agentguard_zero.training.coevolution import sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_with_target_cap(
    rows: list[dict[str, Any]], *, target_cap: int
) -> list[dict[str, Any]]:
    if target_cap < 1:
        raise ValueError("target_cap must be positive")
    target_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["record_id"])):
        target = str(row["target"])
        if target_counts[target] >= target_cap:
            continue
        target_counts[target] += 1
        selected.append(row)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-cap", type=int, default=4)
    parser.add_argument("--min-records", type=int, default=2_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    parent_manifest_path = args.input_dir / "manifest.json"
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if (
        parent.get("accepted") is not True
        or parent.get("protocol_version") != RECOVERY_PROTOCOL_VERSION
        or parent.get("counterfactual_worlds_training_visible") is not False
    ):
        raise RuntimeError("parent Teacher dataset is not accepted/public-only")
    rows = pd.read_parquet(args.input_dir / "bootstrap_sft.parquet").to_dict(
        orient="records"
    )
    audits = read_jsonl(args.input_dir / "teacher_selection_audit.jsonl")
    audit_by_id = {str(row["record_id"]): row for row in audits}
    if len(audit_by_id) != len(audits):
        raise RuntimeError("duplicate Teacher audit record IDs")

    selected = select_with_target_cap(rows, target_cap=args.target_cap)
    # PyArrow may materialize the nested messages column as ndarray objects.
    # Reconstruct the canonical two-message view from the unchanged prompt and
    # target before running the strict public-data audit.
    selected = [
        {
            **row,
            "messages": [
                {"role": "user", "content": str(row["prompt"])},
                {"role": "assistant", "content": str(row["target"])},
            ],
        }
        for row in selected
    ]
    selected_audits = [audit_by_id[str(row["record_id"])] for row in selected]
    if len(selected) < args.min_records:
        raise RuntimeError("sequence-balanced dataset is smaller than requested")
    target_counts = Counter(str(row["target"]) for row in selected)
    category_counts = Counter(str(row["action_category"]) for row in selected)
    manifest = audit_bootstrap_records(selected, selected_audits)
    manifest.update(
        {
            "schema_version": 1,
            "kind": "recovery_sequence_balanced_teacher_dataset",
            "protocol_version": RECOVERY_PROTOCOL_VERSION,
            "source_scenarios_sha256": parent["source_scenarios_sha256"],
            "source_scenario_count": parent["source_scenario_count"],
            "counterfactual_worlds_training_visible": False,
            "human_action_labels": 0,
            "parent_manifest": str(parent_manifest_path.resolve()),
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
            "selection_policy": {
                "name": "deterministic_exact_target_frequency_cap",
                "sort_key": "record_id",
                "target_cap": args.target_cap,
                "model_performance_used": False,
            },
            "record_count": len(selected),
            "exact_target_count": len(target_counts),
            "maximum_exact_target_frequency": max(target_counts.values()),
            "action_category_counts": dict(sorted(category_counts.items())),
            "target_frequency_gate": {
                "maximum": args.target_cap,
                "observed": max(target_counts.values()),
                "accepted": max(target_counts.values()) <= args.target_cap,
            },
        }
    )
    rank_gate = {
        "accepted": bool(
            manifest.get("teacher_core_rank_correlation_state_count", 0) >= 200
            and float(manifest.get("teacher_core_rank_correlation_mean", 0.0))
            > 0.5
        )
    }
    manifest["teacher_core_rank_correlation_gate"] = rank_gate
    manifest["accepted"] = bool(
        manifest.get("accepted")
        and rank_gate["accepted"]
        and manifest["target_frequency_gate"]["accepted"]
        and len(selected) >= args.min_records
    )
    manifest["status"] = "accepted" if manifest["accepted"] else "rejected"

    args.output_dir.mkdir(parents=True)
    train = args.output_dir / "bootstrap_sft.parquet"
    audit = args.output_dir / "teacher_selection_audit.jsonl"
    out_manifest = args.output_dir / "manifest.json"
    pd.DataFrame(selected).to_parquet(train, index=False)
    with audit.open("w", encoding="utf-8") as handle:
        for row in selected_audits:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    out_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sums = {
        path.name: sha256_file(path) for path in (train, audit, out_manifest)
    }
    (args.output_dir / "SHA256SUMS.json").write_text(
        json.dumps(sums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
