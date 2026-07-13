#!/usr/bin/env python3
"""Filter a fresh DCA_{r+1} pool and create isolated VDA round splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentguard_zero.env.checker import full_check
from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    feedback_fingerprints,
    load_checkpoint_manifest,
    parse_utc,
    read_json,
    scenario_fingerprint,
    sha256_file,
    utc_now,
)
from agentguard_zero.training.vda_dataset import scenario_to_training_row
from generate_level1_frontier import compute_cfc_metrics


TASK_IDS = ("T1", "T2", "T3", "T4")


def _safe_cfc_metrics(scenario: dict[str, Any]) -> dict[str, Any] | None:
    """Treat malformed generated scenarios as rejected candidates, not fatal input."""
    try:
        return compute_cfc_metrics(scenario)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _task_id(record: dict[str, Any], scenario: dict[str, Any]) -> str:
    value = str(record.get("task_focus") or scenario.get("metadata", {}).get("task_id", ""))
    for task_id in TASK_IDS:
        if value.upper().startswith(task_id):
            return task_id
    return "unknown"


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _split_stratified(
    items: list[dict[str, Any]], split_counts: dict[str, int], seed: int
) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[item["task_id"]].append(item)
    for values in by_task.values():
        values.sort(key=lambda item: float(item["cfc"].get("frontier_score", 0.0)), reverse=True)

    rng = random.Random(seed)
    split_items: dict[str, list[dict[str, Any]]] = {}
    remainder_offset = 0
    for split, count in split_counts.items():
        base, remainder = divmod(count, len(TASK_IDS))
        quota = {task_id: base for task_id in TASK_IDS}
        for offset in range(remainder):
            quota[TASK_IDS[(remainder_offset + offset) % len(TASK_IDS)]] += 1
        remainder_offset = (remainder_offset + remainder) % len(TASK_IDS)

        selected: list[dict[str, Any]] = []
        for task_id in TASK_IDS:
            required = quota[task_id]
            available = by_task.get(task_id, [])
            if len(available) < required:
                raise LineageError(
                    f"task {task_id} has {len(available)} eligible candidates, "
                    f"but split {split} requires {required}"
                )
            selected.extend(available[:required])
            del available[:required]
        rng.shuffle(selected)
        split_items[split] = selected
    return split_items


def _task_batched_order(
    items: list[dict[str, Any]], batch_size: int, seed: int
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        return items
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[item["task_id"]].append(item)
    rng = random.Random(seed)
    batches: list[list[dict[str, Any]]] = []
    leftovers: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        values = by_task.get(task_id, [])
        rng.shuffle(values)
        full = len(values) - (len(values) % batch_size)
        batches.extend(values[start : start + batch_size] for start in range(0, full, batch_size))
        leftovers.extend(values[full:])
    rng.shuffle(leftovers)
    batches.extend(
        leftovers[start : start + batch_size]
        for start in range(0, len(leftovers), batch_size)
    )
    rng.shuffle(batches)
    ordered = [item for batch in batches for item in batch]
    if len(ordered) != len(items) or {
        item["scenario_fingerprint"] for item in ordered
    } != {item["scenario_fingerprint"] for item in items}:
        raise LineageError("task-batched VDA ordering changed the selected training set")
    return ordered


def _training_row(
    item: dict[str, Any],
    *,
    split: str,
    dca_manifest_path: Path,
    dca_manifest_sha: str,
    target_round: int,
    candidate_pool_path: Path,
    candidate_pool_sha: str,
) -> dict[str, Any]:
    row = scenario_to_training_row(item["scenario"], split=split)
    extra = dict(row.get("extra_info", {}) or {})
    extra.update(
        {
            "task_id": item["task_id"],
            "task_focus": item.get("task_focus", ""),
            "scenario_fingerprint": item["scenario_fingerprint"],
            "source_role": "dca",
            "source_dca_round": int(target_round),
            "source_checkpoint_manifest": str(dca_manifest_path),
            "source_checkpoint_manifest_sha256": dca_manifest_sha,
            "source_candidate_pool": str(candidate_pool_path),
            "source_candidate_pool_sha256": candidate_pool_sha,
            "frontier_score": float(item["cfc"].get("frontier_score", 0.0)),
            "difficulty": float(item["cfc"].get("difficulty", 0.0)),
            "oracle_solvable": bool(item["cfc"].get("oracle_solvable", False)),
        }
    )
    row["extra_info"] = extra
    row["frontier_score"] = extra["frontier_score"]
    row["difficulty"] = extra["difficulty"]
    row["task_id"] = item["task_id"]
    row["cfc_metrics"] = json.dumps(item["cfc"], ensure_ascii=False, sort_keys=True)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--feedback-log", required=True)
    parser.add_argument("--dca-checkpoint-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--dev-size", type=int, required=True)
    parser.add_argument("--xplay-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--train-batch-size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = args.train_size + args.dev_size + args.xplay_size
    if min(args.train_size, args.dev_size, args.xplay_size) < 0 or requested <= 0:
        raise SystemExit("split sizes must be non-negative and total size must be positive")

    candidate_path = Path(args.candidate_pool).resolve()
    feedback_path = Path(args.feedback_log).resolve()
    dca_manifest_path = Path(args.dca_checkpoint_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    pool = read_json(candidate_path)
    raw_manifest = read_json(dca_manifest_path)
    target_round = int(raw_manifest.get("round", -1))
    backbone = str(raw_manifest.get("backbone", ""))
    dca_manifest = load_checkpoint_manifest(
        dca_manifest_path,
        role="dca",
        backbone=backbone,
        round_index=target_round,
    )
    dca_manifest_sha = sha256_file(dca_manifest_path)
    if pool.get("kind") != "dca_candidate_pool":
        raise LineageError("candidate input is not a dca_candidate_pool")
    if int(pool.get("source_dca_round", -1)) != target_round:
        raise LineageError("candidate pool DCA round does not match checkpoint")
    if pool.get("source_dca_checkpoint_manifest_sha256") != dca_manifest_sha:
        raise LineageError("candidate pool checkpoint hash does not match DCA manifest")
    if parse_utc(pool["created_at"]) < parse_utc(dca_manifest["created_at"]):
        raise LineageError("candidate pool predates DCA_{r+1} checkpoint")

    feedback = feedback_fingerprints(feedback_path)
    candidate_pool_sha = sha256_file(candidate_path)
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    excluded = Counter()
    for record in pool.get("candidates", []) or []:
        scenario = record.get("scenario")
        if not record.get("parse_ok", False) or not isinstance(scenario, dict):
            excluded["parse_or_type"] += 1
            continue
        fingerprint = scenario_fingerprint(scenario)
        if fingerprint != str(record.get("scenario_fingerprint", "")):
            excluded["fingerprint_mismatch"] += 1
            continue
        if fingerprint in feedback:
            excluded["dca_feedback_overlap"] += 1
            continue
        if fingerprint in seen or record.get("duplicate", False):
            excluded["duplicate"] += 1
            continue
        metadata = scenario.get("metadata", {}) or {}
        if int(metadata.get("source_dca_round", -1)) != target_round:
            excluded["wrong_dca_round"] += 1
            continue
        if metadata.get("source_checkpoint_manifest_sha256") != dca_manifest_sha:
            excluded["wrong_checkpoint"] += 1
            continue
        checks = full_check(scenario)
        if not checks.get("all_ok", False):
            excluded["hard_check"] += 1
            continue
        cfc = _safe_cfc_metrics(scenario)
        if cfc is None:
            excluded["cfc_exception"] += 1
            continue
        if not cfc.get("oracle_solvable", False) or float(cfc.get("frontier_score", 0.0)) <= 0.0:
            excluded["not_hard_but_solvable"] += 1
            continue
        seen.add(fingerprint)
        valid.append(
            {
                **record,
                "scenario_fingerprint": fingerprint,
                "task_id": _task_id(record, scenario),
                "cfc": cfc,
            }
        )

    if len(valid) < requested:
        raise LineageError(
            f"only {len(valid)} isolated hard-but-solvable candidates for {requested} requested; "
            f"excluded={dict(excluded)}"
        )

    split_items = _split_stratified(
        valid,
        {"train": args.train_size, "dev": args.dev_size, "xplay": args.xplay_size},
        args.seed,
    )
    split_items["train"] = _task_batched_order(
        split_items["train"], args.train_batch_size, args.seed + 17
    )
    selected = [item for split in ("train", "dev", "xplay") for item in split_items[split]]
    paths = {
        "train": output_dir / "vda_train" / "train.parquet",
        "dev": output_dir / "vda_dev" / "dev.parquet",
        "xplay": output_dir / "vda_xplay" / "xplay.parquet",
    }
    for split, items in split_items.items():
        rows = [
            _training_row(
                item,
                split=split,
                dca_manifest_path=dca_manifest_path,
                dca_manifest_sha=dca_manifest_sha,
                target_round=target_round,
                candidate_pool_path=candidate_path,
                candidate_pool_sha=candidate_pool_sha,
            )
            for item in items
        ]
        _write_parquet(paths[split], rows)

    frontier_path = output_dir / "vda_candidates" / "frontier.json"
    atomic_write_json(
        frontier_path,
        {
            "kind": "vda_frontier",
            "created_at": utc_now(),
            "source_candidate_pool": str(candidate_path),
            "selected": selected,
        },
    )
    split_fingerprints = {
        split: [item["scenario_fingerprint"] for item in items] for split, items in split_items.items()
    }
    all_split_fingerprints = [value for values in split_fingerprints.values() for value in values]
    if len(all_split_fingerprints) != len(set(all_split_fingerprints)):
        raise LineageError("VDA train/dev/xplay splits overlap")
    if feedback.intersection(all_split_fingerprints):
        raise LineageError("DCA feedback leaked into a VDA split")

    manifest_path = output_dir / "vda_pool_manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "vda_pool",
        "created_at": utc_now(),
        "seed": args.seed,
        "backbone": backbone,
        "target_round": target_round,
        "source_dca_checkpoint_manifest": str(dca_manifest_path),
        "source_dca_checkpoint_manifest_sha256": dca_manifest_sha,
        "source_dca_adapter_sha256": dca_manifest.get("adapter_sha256"),
        "candidate_pool": str(candidate_path),
        "candidate_pool_sha256": candidate_pool_sha,
        "feedback_log": str(feedback_path),
        "feedback_log_sha256": sha256_file(feedback_path),
        "feedback_fingerprint_count": len(feedback),
        "feedback_vda_overlap_count": 0,
        "candidate_count": len(pool.get("candidates", []) or []),
        "eligible_count": len(valid),
        "selected_count": len(selected),
        "excluded_counts": dict(excluded),
        "split_counts": {split: len(items) for split, items in split_items.items()},
        "split_task_counts": {
            split: dict(Counter(item["task_id"] for item in items)) for split, items in split_items.items()
        },
        "train_ordering": (
            f"task_batched_{args.train_batch_size}"
            if args.train_batch_size > 0
            else "row_shuffled"
        ),
        "split_fingerprints": split_fingerprints,
        "paths": {split: str(path) for split, path in paths.items()} | {"frontier": str(frontier_path)},
        "sha256": {split: sha256_file(path) for split, path in paths.items()}
        | {"frontier": sha256_file(frontier_path)},
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({**manifest, "split_fingerprints": "omitted"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
