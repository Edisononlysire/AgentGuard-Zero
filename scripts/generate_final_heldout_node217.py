#!/usr/bin/env python3
"""Generate and seal a disjoint DCA_3 final-heldout split on four GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
    canonical_json,
    load_checkpoint_manifest,
    parquet_lineage,
    read_json,
    read_jsonl,
    scenario_fingerprint,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from agentguard_zero.training.vda_dataset import scenario_to_training_row
from generate_level1_frontier import compute_cfc_metrics


TASK_IDS = ("T1", "T2", "T3", "T4")


def _task_id(record: dict[str, Any], scenario: dict[str, Any]) -> str:
    value = str(record.get("task_focus") or scenario.get("metadata", {}).get("task_id", ""))
    for task_id in TASK_IDS:
        if value.upper().startswith(task_id):
            return task_id
    return "unknown"


def _training_fingerprints(root: Path, backbone: str) -> set[str]:
    fingerprints: set[str] = set()
    for round_index in (1, 2, 3):
        round_dir = root / "data" / "co_evolution" / backbone / f"round_{round_index}"
        feedback_path = round_dir / "dca_feedback" / "feedback.jsonl"
        if feedback_path.exists():
            for row in read_jsonl(feedback_path):
                fingerprint = row.get("scenario_fingerprint")
                if fingerprint:
                    fingerprints.add(str(fingerprint))
        candidate_path = round_dir / "vda_candidates" / "candidates.json"
        if candidate_path.exists():
            for record in read_json(candidate_path).get("candidates", []) or []:
                fingerprint = record.get("scenario_fingerprint")
                if fingerprint:
                    fingerprints.add(str(fingerprint))
        for split, filename in (
            ("vda_train", "train.parquet"),
            ("vda_dev", "dev.parquet"),
            ("vda_xplay", "xplay.parquet"),
        ):
            split_path = round_dir / split / filename
            if split_path.exists():
                fingerprints.update(
                    row["scenario_fingerprint"] for row in parquet_lineage(split_path)
                )
    return fingerprints


def _other_heldout_fingerprints(root: Path, backbone: str) -> set[str]:
    fingerprints: set[str] = set()
    heldout_root = root / "data" / "final_heldout"
    for path in heldout_root.glob("*/final_heldout.parquet"):
        if path.parent.name == backbone:
            continue
        fingerprints.update(row["scenario_fingerprint"] for row in parquet_lineage(path))
    return fingerprints


def select_heldout(
    records: list[dict[str, Any]], excluded: set[str], per_task: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    for record in records:
        scenario = record.get("scenario")
        if not record.get("parse_ok", False) or not isinstance(scenario, dict):
            rejected["parse_or_type"] += 1
            continue
        fingerprint = scenario_fingerprint(scenario)
        if fingerprint != str(record.get("scenario_fingerprint", "")):
            rejected["fingerprint_mismatch"] += 1
            continue
        if fingerprint in excluded:
            rejected["training_overlap"] += 1
            continue
        if fingerprint in seen or record.get("duplicate", False):
            rejected["duplicate"] += 1
            continue
        checks = full_check(scenario)
        if not checks.get("all_ok", False):
            rejected["hard_check"] += 1
            continue
        cfc = compute_cfc_metrics(scenario)
        if not cfc.get("oracle_solvable", False) or float(cfc.get("frontier_score", 0.0)) <= 0:
            rejected["not_hard_but_solvable"] += 1
            continue
        task_id = _task_id(record, scenario)
        if task_id not in TASK_IDS:
            rejected["unknown_task"] += 1
            continue
        seen.add(fingerprint)
        by_task[task_id].append(
            {
                "task_id": task_id,
                "scenario_fingerprint": fingerprint,
                "frontier_score": float(cfc.get("frontier_score", 0.0)),
                "cfc": cfc,
                "scenario": scenario,
            }
        )

    selected: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        values = sorted(by_task[task_id], key=lambda item: item["frontier_score"], reverse=True)
        if len(values) < per_task:
            raise LineageError(
                f"final-heldout task {task_id} has {len(values)} eligible scenarios, "
                f"requires {per_task}; rejected={dict(rejected)}"
            )
        selected.extend(values[:per_task])
    selected.sort(key=lambda item: (item["task_id"], item["scenario_fingerprint"]))
    return selected, rejected


def _run_parallel(jobs: list[tuple[list[str], dict[str, str]]]) -> None:
    processes = [(command, subprocess.Popen(command, env=environment)) for command, environment in jobs]
    try:
        for command, process in processes:
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
    finally:
        for _command, process in processes:
            if process.poll() is None:
                process.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--backbone", choices=["qwen3.5-4b", "qwen3.5-9b"], required=True)
    parser.add_argument("--candidate-count", type=int, default=4000)
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260709 + 900000)
    parser.add_argument("--allocated-gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    gpu_ids = [value.strip() for value in args.allocated_gpus.split(",") if value.strip()]
    if len(gpu_ids) != 4:
        raise SystemExit(f"final-heldout generation requires exactly four GPUs, got {gpu_ids}")
    if args.candidate_count < args.per_task * len(TASK_IDS):
        raise SystemExit("candidate count is smaller than the requested heldout size")

    dca_manifest_path = root / "checkpoints" / args.backbone / "dca" / "round_3" / "manifest.json"
    dca_manifest = load_checkpoint_manifest(
        dca_manifest_path, role="dca", backbone=args.backbone, round_index=3
    )
    output_dir = root / "data" / "final_heldout" / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final_heldout.parquet"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if (
            existing.get("status") == "sealed"
            and existing.get("source_dca_checkpoint_manifest_sha256")
            == sha256_file(dca_manifest_path)
            and final_path.exists()
            and existing.get("final_heldout_sha256") == sha256_file(final_path)
        ):
            print(json.dumps(existing, indent=2), flush=True)
            return

    shard_paths = [output_dir / f"candidates.shard_{index}.json" for index in range(4)]
    candidate_path = output_dir / "candidates.json"
    jobs = []
    for shard_index, (gpu_id, shard_path) in enumerate(zip(gpu_ids, shard_paths)):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        triton_cache = Path(
            os.environ.get("AGZ_TRITON_CACHE_ROOT", "/tmp/agentguard_zero_triton")
        ) / f"agz_{args.backbone}_final_heldout" / f"shard_{shard_index}"
        triton_cache.mkdir(parents=True, exist_ok=True)
        environment["TRITON_CACHE_DIR"] = str(triton_cache)
        jobs.append(
            (
                [
                    sys.executable,
                    str(root / "scripts" / "generate_dca_scenarios.py"),
                    "--checkpoint-manifest",
                    str(dca_manifest_path),
                    "--output",
                    str(shard_path),
                    "--num-candidates",
                    str(args.candidate_count),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-shards",
                    "4",
                    "--shard-index",
                    str(shard_index),
                    "--seed",
                    str(args.seed),
                ],
                environment,
            )
        )
    _run_parallel(jobs)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "merge_dca_candidate_shards.py"),
            "--shards",
            *[str(path) for path in shard_paths],
            "--expected-count",
            str(args.candidate_count),
            "--output",
            str(candidate_path),
        ],
        check=True,
    )

    training_fingerprints = _training_fingerprints(root, args.backbone)
    cross_backbone_fingerprints = _other_heldout_fingerprints(root, args.backbone)
    exclusion_fingerprints = training_fingerprints | cross_backbone_fingerprints
    pool = read_json(candidate_path)
    if pool.get("kind") != "dca_candidate_pool":
        raise LineageError("final-heldout candidate input is not a merged DCA pool")
    if int(pool.get("source_dca_round", -1)) != 3:
        raise LineageError("final-heldout candidates were not generated by DCA_3")
    if pool.get("source_dca_checkpoint_manifest_sha256") != sha256_file(dca_manifest_path):
        raise LineageError("final-heldout candidate checkpoint hash does not match DCA_3")
    selected, rejected = select_heldout(
        pool.get("candidates", []) or [], exclusion_fingerprints, args.per_task
    )
    rows = []
    for item in selected:
        row = scenario_to_training_row(item["scenario"], split="final_heldout")
        extra = dict(row.get("extra_info", {}) or {})
        extra.update(
            {
                "task_id": item["task_id"],
                "scenario_fingerprint": item["scenario_fingerprint"],
                "source_dca_round": 3,
                "source_checkpoint_manifest_sha256": sha256_file(dca_manifest_path),
                "sealed_heldout": True,
            }
        )
        row["extra_info"] = extra
        row["task_id"] = item["task_id"]
        row["frontier_score"] = item["frontier_score"]
        row["cfc_metrics"] = json.dumps(item["cfc"], ensure_ascii=False, sort_keys=True)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(final_path, index=False)

    selected_fingerprints = [item["scenario_fingerprint"] for item in selected]
    if exclusion_fingerprints.intersection(selected_fingerprints):
        raise LineageError("final-heldout overlaps co-evolution or an earlier heldout source")
    manifest = {
        "schema_version": 1,
        "kind": "dca_final_heldout",
        "status": "sealed",
        "sealed_at": utc_now(),
        "backbone": args.backbone,
        "seed": args.seed,
        "source_dca_round": 3,
        "source_dca_checkpoint_manifest": str(dca_manifest_path),
        "source_dca_checkpoint_manifest_sha256": sha256_file(dca_manifest_path),
        "source_dca_adapter_sha256": dca_manifest.get("adapter_sha256"),
        "candidate_pool": str(candidate_path),
        "candidate_pool_sha256": sha256_file(candidate_path),
        "candidate_count": args.candidate_count,
        "selected_count": len(selected),
        "task_counts": dict(Counter(item["task_id"] for item in selected)),
        "rejected_counts": dict(rejected),
        "training_exclusion_count": len(training_fingerprints),
        "training_exclusion_sha256": sha256_bytes(
            canonical_json(sorted(training_fingerprints)).encode("utf-8")
        ),
        "cross_backbone_exclusion_count": len(cross_backbone_fingerprints),
        "cross_backbone_exclusion_sha256": sha256_bytes(
            canonical_json(sorted(cross_backbone_fingerprints)).encode("utf-8")
        ),
        "selected_fingerprints_sha256": sha256_bytes(
            canonical_json(selected_fingerprints).encode("utf-8")
        ),
        "final_heldout": str(final_path),
        "final_heldout_sha256": sha256_file(final_path),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
