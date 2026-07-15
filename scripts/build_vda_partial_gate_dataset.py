#!/usr/bin/env python3
"""Freeze a small balanced VDA gate dataset from valid DCA partial records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.env.checker import full_check
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    load_checkpoint_manifest,
    scenario_fingerprint,
    sha256_file,
    utc_now,
)
from agentguard_zero.training.vda_dataset import scenario_to_training_row


TASK_IDS = ("T1", "T2", "T3", "T4")


def collect_balanced_records(
    paths: list[Path], *, rows_per_task: int, dca_manifest_sha: str
) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
            metadata = json.loads(first)
            if metadata.get("kind") != "meta":
                raise ValueError(f"partial file has no metadata header: {path}")
            if metadata.get("config", {}).get("checkpoint_manifest_sha256") != dca_manifest_sha:
                raise ValueError(f"partial file DCA lineage mismatch: {path}")
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record = item.get("record") if item.get("kind") == "record" else None
                if not isinstance(record, dict):
                    continue
                scenario = record.get("scenario")
                if not record.get("parse_ok") or not isinstance(scenario, dict):
                    continue
                if not (record.get("checks", {}) or {}).get("all_ok"):
                    continue
                if not full_check(scenario).get("all_ok"):
                    continue
                fingerprint = scenario_fingerprint(scenario)
                if fingerprint != record.get("scenario_fingerprint") or fingerprint in seen:
                    continue
                task_id = str(record.get("task_focus", "")).split()[0]
                if task_id not in TASK_IDS or len(by_task[task_id]) >= rows_per_task:
                    continue
                seen.add(fingerprint)
                by_task[task_id].append(record)
    shortages = {
        task_id: len(by_task.get(task_id, []))
        for task_id in TASK_IDS
        if len(by_task.get(task_id, [])) < rows_per_task
    }
    if shortages:
        raise ValueError(f"insufficient valid partial records for gate: {shortages}")
    return [record for task_id in TASK_IDS for record in by_task[task_id]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partials", nargs="+", required=True)
    parser.add_argument("--dca-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows-per-task", type=int, default=8)
    args = parser.parse_args()
    if args.rows_per_task <= 0:
        raise SystemExit("--rows-per-task must be positive")

    partials = [Path(value).resolve() for value in args.partials]
    manifest_path = Path(args.dca_manifest).resolve()
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = load_checkpoint_manifest(
        manifest_path,
        role="dca",
        backbone=str(raw_manifest.get("backbone", "")),
        round_index=int(raw_manifest.get("round", -1)),
    )
    manifest_sha = sha256_file(manifest_path)
    records = collect_balanced_records(
        partials,
        rows_per_task=args.rows_per_task,
        dca_manifest_sha=manifest_sha,
    )

    import pandas as pd

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "train.parquet"
    rows = [scenario_to_training_row(record["scenario"], split="gate") for record in records]
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    gate_manifest = {
        "schema_version": 1,
        "kind": "vda_partial_gate_pool",
        "created_at": utc_now(),
        "backbone": manifest["backbone"],
        "source_dca_round": manifest["round"],
        "source_dca_manifest": str(manifest_path),
        "source_dca_manifest_sha256": manifest_sha,
        "partial_paths": [str(path) for path in partials],
        "rows_per_task": args.rows_per_task,
        "row_count": len(rows),
        "task_counts": {task_id: args.rows_per_task for task_id in TASK_IDS},
        "scenario_fingerprints": [
            record["scenario_fingerprint"] for record in records
        ],
        "parquet": str(parquet_path),
        "parquet_sha256": sha256_file(parquet_path),
        "official_training_artifact": False,
    }
    manifest_output = output_dir / "manifest.json"
    atomic_write_json(manifest_output, gate_manifest)
    print(json.dumps(gate_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
