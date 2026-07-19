#!/usr/bin/env python3
"""Create deterministic train/dev candidate-set manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            str(row.get("semantic_scenario_fingerprint", row["record_id"])), []
        ).append(row)
    ranked_groups = sorted(
        grouped.items(),
        key=lambda item: hashlib.sha256(f"{args.seed}:{item[0]}".encode()).hexdigest(),
    )
    target_dev = max(1, round(len(rows) * args.dev_ratio))
    dev: list[dict] = []
    train: list[dict] = []
    for _, values in ranked_groups:
        (dev if len(dev) < target_dev else train).extend(values)
    if not train and len(ranked_groups) > 1:
        moved_fingerprint, moved = ranked_groups[-1]
        del moved_fingerprint
        dev_ids = {row["record_id"] for row in moved}
        dev = [row for row in dev if row["record_id"] not in dev_ids]
        train.extend(moved)
    args.output_dir.mkdir(parents=True)
    for name, values in (("train", train), ("dev", dev)):
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, values)
        manifest = {
            "schema_version": 1,
            "kind": "candidate_listwise_teacher_dataset",
            "created_at": utc_now(),
            "accepted": bool(values),
            "split": name,
            "record_count": len(values),
            "candidate_sets_sha256": sha256_file(path),
            "parent_candidate_sets_sha256": sha256_file(args.input),
            "seed": args.seed,
        }
        atomic_write_json(args.output_dir / f"{name}_manifest.json", manifest)
    summary = {
        "train_records": len(train),
        "dev_records": len(dev),
        "parent_sha256": sha256_file(args.input),
        "semantic_train_dev_overlap": len(
            {row["semantic_scenario_fingerprint"] for row in train}
            & {row["semantic_scenario_fingerprint"] for row in dev}
        ),
    }
    if summary["semantic_train_dev_overlap"]:
        raise RuntimeError("semantic scenario overlap across train/dev")
    atomic_write_json(args.output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
