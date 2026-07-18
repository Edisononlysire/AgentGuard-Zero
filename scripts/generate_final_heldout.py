#!/usr/bin/env python3
"""Generate and seal a disjoint DCA_3 TMCD-Test split on four GPUs.

Selection is deliberately model-independent: candidates pass only structural,
safety, task, solvability, leakage, and overlap gates before seeded sampling.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentguard_zero.env.checker import full_check
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.protocol import FAMILY_TASK_MAP, TASK_FAMILY_MAP
from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    canonical_json,
    load_checkpoint_manifest,
    read_json,
    read_jsonl,
    scenario_fingerprint,
    sha256_bytes,
    sha256_file,
    sha256_source_tree,
    utc_now,
)
from agentguard_zero.training.vda_dataset import scenario_to_training_row
from agentguard_zero.world.public_projector import forbidden_public_paths


TASK_IDS = ("T1", "T2", "T3", "T4")


def semantic_overlap_fingerprint(scenario: dict[str, Any]) -> str:
    """Hash scenario semantics while ignoring generator and split identities."""

    payload = copy.deepcopy(scenario)
    for key in (
        "scenario_id",
        "metadata",
        "split",
        "distribution",
        "pair_id",
        "prefix_hash",
    ):
        payload.pop(key, None)
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _as_scenario(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def _task_id(record: dict[str, Any], scenario: dict[str, Any]) -> str:
    focus = str(record.get("task_focus", "")).strip().upper()
    focus_task = next((task for task in TASK_IDS if focus.startswith(task)), "unknown")
    metadata_task = str((scenario.get("metadata", {}) or {}).get("task_id", "")).upper()
    family_task = FAMILY_TASK_MAP.get(str(scenario.get("scenario_family", "")), "unknown")
    if (
        focus_task in TASK_IDS
        and metadata_task == focus_task
        and family_task == focus_task
        and TASK_FAMILY_MAP[focus_task] == scenario.get("scenario_family")
    ):
        return focus_task
    return "unknown"


def _iter_parquet_scenarios(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    frame = pd.read_parquet(path, columns=["scenario"])
    for value in frame["scenario"].tolist():
        scenario = _as_scenario(value)
        if scenario is not None:
            yield scenario


def _iter_calibration_scenarios(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    if path.suffix == ".parquet":
        yield from _iter_parquet_scenarios(path)
        return
    if path.suffix == ".jsonl":
        for row in read_jsonl(path):
            scenario = _as_scenario(row.get("scenario") or row)
            if scenario is not None:
                yield scenario
        return
    if path.suffix == ".json":
        value = read_json(path)
        records = value if isinstance(value, list) else value.get("scenarios", [])
        for row in records or []:
            scenario = _as_scenario(row.get("scenario") if isinstance(row, dict) else row)
            if scenario is not None:
                yield scenario


def _calibration_paths(root: Path, artifact_scope: str, explicit: list[str]) -> list[Path]:
    paths = {Path(value).resolve() for value in explicit}
    candidates = (
        root / "data" / "v5c" / "calibration",
        root / "data" / "ecrg" / "calibration",
        root / "data" / artifact_scope / "v5c_calibration",
        root / "data" / artifact_scope / "ecrg_calibration",
    )
    for candidate in candidates:
        if candidate.is_file():
            paths.add(candidate.resolve())
        elif candidate.is_dir():
            for suffix in ("*.parquet", "*.jsonl", "*.json"):
                paths.update(path.resolve() for path in candidate.rglob(suffix))
    return sorted(paths)


def build_exclusion_ledger(
    root: Path,
    artifact_scope: str,
    backbone: str,
    calibration_paths: list[Path],
) -> dict[str, Any]:
    lineage: set[str] = set()
    semantic: set[str] = set()
    sources: dict[str, dict[str, Any]] = {}
    data_root = root / "data" / artifact_scope / backbone

    def record_source(label: str, path: Path, scenarios: Iterable[dict[str, Any]]) -> None:
        before_lineage = len(lineage)
        before_semantic = len(semantic)
        rows = 0
        for scenario in scenarios:
            rows += 1
            lineage.add(scenario_fingerprint(scenario))
            semantic.add(semantic_overlap_fingerprint(scenario))
        sources[label] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": rows,
            "new_lineage_fingerprints": len(lineage) - before_lineage,
            "new_semantic_fingerprints": len(semantic) - before_semantic,
        }

    for round_index in (1, 2, 3):
        round_dir = data_root / f"round_{round_index}"
        feedback_path = round_dir / "dca_feedback" / "feedback.jsonl"
        if feedback_path.exists():
            feedback_rows = read_jsonl(feedback_path)
            record_source(
                f"round_{round_index}.dca_feedback",
                feedback_path,
                (
                    scenario
                    for row in feedback_rows
                    if (scenario := _as_scenario(row.get("scenario"))) is not None
                ),
            )
        candidate_path = round_dir / "vda_candidates" / "candidates.json"
        if candidate_path.exists():
            candidate_rows = read_json(candidate_path).get("candidates", []) or []
            record_source(
                f"round_{round_index}.vda_candidates",
                candidate_path,
                (
                    scenario
                    for row in candidate_rows
                    if (scenario := _as_scenario(row.get("scenario"))) is not None
                ),
            )
        for split, filename in (
            ("vda_train", "train.parquet"),
            ("vda_dev", "dev.parquet"),
            ("vda_xplay", "xplay.parquet"),
        ):
            split_path = round_dir / split / filename
            if split_path.exists():
                record_source(
                    f"round_{round_index}.{split}",
                    split_path,
                    _iter_parquet_scenarios(split_path),
                )

    for index, path in enumerate(calibration_paths):
        record_source(
            f"v5c_calibration.{index}",
            path,
            _iter_calibration_scenarios(path),
        )

    heldout_root = root / "data" / "final_heldout" / artifact_scope
    for path in heldout_root.glob("*/final_heldout.parquet"):
        if path.parent.name != backbone:
            record_source(
                f"cross_backbone_heldout.{path.parent.name}",
                path,
                _iter_parquet_scenarios(path),
            )

    return {
        "schema_version": 1,
        "kind": "tmcd_test_exclusion_ledger",
        "created_at": utc_now(),
        "artifact_scope": artifact_scope,
        "backbone": backbone,
        "sources": sources,
        "v5c_calibration_present": bool(calibration_paths),
        "v5c_calibration_paths": [str(path) for path in calibration_paths],
        "future_v5c_calibration_must_exclude_sealed_tmcd_test": True,
        "lineage_fingerprint_count": len(lineage),
        "semantic_fingerprint_count": len(semantic),
        "lineage_fingerprints": sorted(lineage),
        "semantic_fingerprints": sorted(semantic),
    }


def select_heldout(
    records: list[dict[str, Any]],
    excluded_lineage: set[str],
    excluded_semantic: set[str],
    per_task: int,
    selection_seed: int,
    *,
    require_quota: bool = True,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, int]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_lineage: set[str] = set()
    seen_semantic: set[str] = set()
    rejected: Counter[str] = Counter()
    for record in records:
        scenario = record.get("scenario")
        if not record.get("parse_ok", False) or not isinstance(scenario, dict):
            rejected["parse_or_type"] += 1
            continue
        lineage_fingerprint = scenario_fingerprint(scenario)
        if lineage_fingerprint != str(record.get("scenario_fingerprint", "")):
            rejected["fingerprint_mismatch"] += 1
            continue
        semantic_fingerprint = semantic_overlap_fingerprint(scenario)
        if lineage_fingerprint in excluded_lineage or semantic_fingerprint in excluded_semantic:
            rejected["formal_data_overlap"] += 1
            continue
        if (
            lineage_fingerprint in seen_lineage
            or semantic_fingerprint in seen_semantic
            or record.get("duplicate", False)
        ):
            rejected["candidate_duplicate"] += 1
            continue
        checks = full_check(scenario)
        if not checks.get("all_ok", False):
            rejected["hard_check"] += 1
            continue
        if not bool((checks.get("solvable", {}) or {}).get("ok", False)):
            rejected["oracle_unsolvable"] += 1
            continue
        task_id = _task_id(record, scenario)
        if task_id not in TASK_IDS:
            rejected["task_mismatch"] += 1
            continue
        try:
            public_observation = instantiate_scenario(scenario).observe()
            leaked = forbidden_public_paths(public_observation)
        except Exception:
            rejected["public_projection_error"] += 1
            continue
        if leaked:
            rejected["hidden_state_leakage"] += 1
            continue
        seen_lineage.add(lineage_fingerprint)
        seen_semantic.add(semantic_fingerprint)
        by_task[task_id].append(
            {
                "task_id": task_id,
                "scenario_fingerprint": lineage_fingerprint,
                "semantic_fingerprint": semantic_fingerprint,
                "scenario": scenario,
            }
        )

    eligible_counts = {task_id: len(by_task[task_id]) for task_id in TASK_IDS}
    selected: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(TASK_IDS, start=1):
        values = sorted(by_task[task_id], key=lambda item: item["semantic_fingerprint"])
        if len(values) < per_task:
            if require_quota:
                raise LineageError(
                    f"TMCD-Test task {task_id} has {len(values)} eligible scenarios, "
                    f"requires {per_task}; eligible={eligible_counts}; rejected={dict(rejected)}"
                )
            continue
        rng = random.Random(selection_seed + task_index * 1_000_003)
        selected.extend(rng.sample(values, per_task))
    selected.sort(key=lambda item: (item["task_id"], item["semantic_fingerprint"]))
    return selected, rejected, eligible_counts


def _run_parallel(jobs: list[tuple[list[str], dict[str, str]]]) -> None:
    processes = [(command, subprocess.Popen(command, env=environment)) for command, environment in jobs]
    failures = []
    for command, process in processes:
        return_code = process.wait()
        if return_code:
            failures.append((command, return_code))
    if failures:
        command, return_code = failures[0]
        raise subprocess.CalledProcessError(return_code, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--artifact-scope", default="tmcd_v242")
    parser.add_argument("--backbone", choices=["qwen3.5-4b", "qwen3.5-9b"], required=True)
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--candidate-count", type=int, default=4800)
    parser.add_argument("--per-task", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--generation-seed", type=int, default=21160709)
    parser.add_argument("--selection-seed", type=int, default=21160710)
    parser.add_argument("--v5c-calibration", action="append", default=[])
    parser.add_argument("--topup-size", type=int, default=500)
    parser.add_argument("--max-topup-rounds", type=int, default=10)
    parser.add_argument(
        "--additional-candidate-pool",
        action="append",
        default=[],
        help="Task-targeted DCA_3 pool generated after an initial quota shortfall.",
    )
    parser.add_argument(
        "--allocated-gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    gpu_ids = [value.strip() for value in args.allocated_gpus.split(",") if value.strip()]
    if len(gpu_ids) != 4:
        raise SystemExit(f"TMCD-Test generation requires exactly four GPUs, got {gpu_ids}")
    if args.candidate_count < args.per_task * len(TASK_IDS):
        raise SystemExit("candidate count is smaller than the requested TMCD-Test size")
    if args.candidate_count % len(TASK_IDS):
        raise SystemExit("candidate count must be divisible by four for exact task targeting")

    dca_manifest_path = (
        Path(args.checkpoint_manifest).resolve()
        if args.checkpoint_manifest
        else root
        / "checkpoints"
        / args.artifact_scope
        / args.backbone
        / "dca"
        / "round_3"
        / "manifest.json"
    )
    dca_manifest = load_checkpoint_manifest(
        dca_manifest_path, role="dca", backbone=args.backbone, round_index=3
    )
    output_dir = root / "data" / "final_heldout" / args.artifact_scope / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final_heldout.parquet"
    manifest_path = output_dir / "manifest.json"
    exclusion_path = output_dir / "excluded_fingerprints.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if (
            existing.get("status") == "sealed"
            and existing.get("source_dca_checkpoint_manifest_sha256") == sha256_file(dca_manifest_path)
            and existing.get("initial_candidate_count") == args.candidate_count
            and existing.get("per_task") == args.per_task
            and existing.get("generation_seed") == args.generation_seed
            and existing.get("selection_seed") == args.selection_seed
            and existing.get("generation_max_input_tokens") == args.max_input_tokens
            and existing.get("generation_max_new_tokens") == args.max_new_tokens
            and final_path.exists()
            and existing.get("final_heldout_sha256") == sha256_file(final_path)
        ):
            print(json.dumps(existing, indent=2), flush=True)
            return
        raise LineageError(f"unrecognized or incompatible existing TMCD-Test output: {output_dir}")

    calibration_paths = _calibration_paths(root, args.artifact_scope, args.v5c_calibration)
    exclusion_ledger = build_exclusion_ledger(
        root, args.artifact_scope, args.backbone, calibration_paths
    )
    atomic_write_json(exclusion_path, exclusion_ledger)
    excluded_lineage = set(exclusion_ledger["lineage_fingerprints"])
    excluded_semantic = set(exclusion_ledger["semantic_fingerprints"])

    shard_paths = [output_dir / f"candidates.shard_{index}.json" for index in range(4)]
    candidate_path = output_dir / "candidates.json"
    jobs = []
    for shard_index, (gpu_id, shard_path) in enumerate(zip(gpu_ids, shard_paths)):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        triton_cache = Path(
            os.environ.get("AGZ_TRITON_CACHE_ROOT", "/tmp/agentguard_zero_triton")
        ) / f"agz_{args.backbone}_tmcd_test" / f"shard_{shard_index}"
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
                    "--max-input-tokens",
                    str(args.max_input_tokens),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--num-shards",
                    "4",
                    "--shard-index",
                    str(shard_index),
                    "--seed",
                    str(args.generation_seed),
                    "--experiment-variant",
                    "full",
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

    pool = read_json(candidate_path)
    if pool.get("kind") != "dca_candidate_pool":
        raise LineageError("TMCD-Test candidate input is not a merged DCA pool")
    if int(pool.get("source_dca_round", -1)) != 3:
        raise LineageError("TMCD-Test candidates were not generated by DCA_3")
    if pool.get("source_dca_checkpoint_manifest_sha256") != sha256_file(dca_manifest_path):
        raise LineageError("TMCD-Test candidate checkpoint hash does not match DCA_3")
    candidate_records = list(pool.get("candidates", []) or [])
    additional_pool_entries = []
    for value in args.additional_candidate_pool:
        path = Path(value).resolve()
        additional = read_json(path)
        if int(additional.get("source_dca_round", -1)) != 3:
            raise LineageError(f"additional candidate pool is not from DCA_3: {path}")
        if additional.get("source_dca_checkpoint_manifest_sha256") != sha256_file(
            dca_manifest_path
        ):
            raise LineageError(f"additional candidate pool checkpoint mismatch: {path}")
        rows = list(additional.get("candidates", []) or [])
        candidate_records.extend(rows)
        additional_pool_entries.append(
            {"path": str(path), "sha256": sha256_file(path), "count": len(rows)}
        )
    automatic_topup_entries = []
    for completed_topup_rounds in range(args.max_topup_rounds + 1):
        selected, rejected, eligible_counts = select_heldout(
            candidate_records,
            excluded_lineage,
            excluded_semantic,
            args.per_task,
            args.selection_seed,
            require_quota=False,
        )
        deficient_tasks = [
            task_id for task_id in TASK_IDS if eligible_counts[task_id] < args.per_task
        ]
        if not deficient_tasks:
            break
        if completed_topup_rounds == args.max_topup_rounds:
            raise LineageError(
                f"TMCD-Test quota remained short after {args.max_topup_rounds} top-up rounds"
            )
        topup_round = completed_topup_rounds + 1
        topup_jobs = []
        topup_specs = []
        for task_id in deficient_tasks:
            task_index = TASK_IDS.index(task_id)
            topup_seed = (
                args.generation_seed
                + topup_round * 100_000_007
                + (task_index + 1) * 1_000_003
            )
            topup_path = output_dir / f"candidates.topup_{task_id}_{topup_round:02d}.json"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[task_index]
            triton_cache = Path(
                os.environ.get("AGZ_TRITON_CACHE_ROOT", "/tmp/agentguard_zero_triton")
            ) / f"agz_{args.backbone}_tmcd_test" / f"topup_{task_id}_{topup_round:02d}"
            triton_cache.mkdir(parents=True, exist_ok=True)
            environment["TRITON_CACHE_DIR"] = str(triton_cache)
            command = [
                sys.executable,
                str(root / "scripts" / "generate_dca_scenarios.py"),
                "--checkpoint-manifest",
                str(dca_manifest_path),
                "--output",
                str(topup_path),
                "--num-candidates",
                str(args.topup_size),
                "--batch-size",
                str(args.batch_size),
                "--max-input-tokens",
                str(args.max_input_tokens),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--num-shards",
                "1",
                "--shard-index",
                "0",
                "--seed",
                str(topup_seed),
                "--experiment-variant",
                "full",
                "--task-id",
                task_id,
            ]
            topup_jobs.append((command, environment))
            topup_specs.append((task_id, topup_seed, topup_path))
        _run_parallel(topup_jobs)
        for task_id, topup_seed, topup_path in topup_specs:
            topup = read_json(topup_path)
            if int(topup.get("source_dca_round", -1)) != 3:
                raise LineageError(f"automatic top-up is not from DCA_3: {topup_path}")
            if topup.get("source_dca_checkpoint_manifest_sha256") != sha256_file(
                dca_manifest_path
            ):
                raise LineageError(f"automatic top-up checkpoint mismatch: {topup_path}")
            if topup.get("task_id") != task_id:
                raise LineageError(f"automatic top-up task mismatch: {topup_path}")
            rows = list(topup.get("candidates", []) or [])
            candidate_records.extend(rows)
            automatic_topup_entries.append(
                {
                    "task_id": task_id,
                    "seed": topup_seed,
                    "path": str(topup_path),
                    "sha256": sha256_file(topup_path),
                    "count": len(rows),
                }
            )
    selected, rejected, eligible_counts = select_heldout(
        candidate_records,
        excluded_lineage,
        excluded_semantic,
        args.per_task,
        args.selection_seed,
    )

    rows = []
    for item in selected:
        row = scenario_to_training_row(item["scenario"], split="final_heldout")
        extra = dict(row.get("extra_info", {}) or {})
        extra.update(
            {
                "task_id": item["task_id"],
                "scenario_fingerprint": item["scenario_fingerprint"],
                "semantic_overlap_fingerprint": item["semantic_fingerprint"],
                "source_dca_round": 3,
                "source_checkpoint_manifest_sha256": sha256_file(dca_manifest_path),
                "sealed_heldout": True,
                "selection_policy": "hard_filter_then_seeded_random",
            }
        )
        row["extra_info"] = extra
        row["task_id"] = item["task_id"]
        row["scenario_fingerprint"] = item["scenario_fingerprint"]
        row["semantic_overlap_fingerprint"] = item["semantic_fingerprint"]
        rows.append(row)
    pd.DataFrame(rows).to_parquet(final_path, index=False)

    selected_lineage = [item["scenario_fingerprint"] for item in selected]
    selected_semantic = [item["semantic_fingerprint"] for item in selected]
    if excluded_lineage.intersection(selected_lineage) or excluded_semantic.intersection(selected_semantic):
        raise LineageError("sealed TMCD-Test overlaps formal data or V5-C calibration")
    if len(set(selected_lineage)) != len(selected_lineage) or len(set(selected_semantic)) != len(selected_semantic):
        raise LineageError("sealed TMCD-Test contains duplicate scenarios")

    audit = {
        "schema_version": 1,
        "kind": "tmcd_test_selection_audit",
        "created_at": utc_now(),
        "model_performance_filtering": False,
        "frontier_score_filtering": False,
        "oracle_used_for_boolean_solvability_only": True,
        "candidate_count": len(candidate_records),
        "eligible_counts": eligible_counts,
        "rejected_counts": dict(rejected),
        "selected_task_counts": dict(Counter(item["task_id"] for item in selected)),
        "selected_formal_overlap_count": 0,
        "selected_duplicate_count": 0,
        "rejected_formal_overlap_count": int(rejected.get("formal_data_overlap", 0)),
        "rejected_candidate_duplicate_count": int(rejected.get("candidate_duplicate", 0)),
    }
    audit_path = output_dir / "selection_audit.json"
    atomic_write_json(audit_path, audit)
    manifest = {
        "schema_version": 2,
        "kind": "dca_tmcd_test",
        "status": "sealed",
        "sealed_at": utc_now(),
        "artifact_scope": args.artifact_scope,
        "backbone": args.backbone,
        "generation_seed": args.generation_seed,
        "selection_seed": args.selection_seed,
        "generation_max_input_tokens": args.max_input_tokens,
        "generation_max_new_tokens": args.max_new_tokens,
        "selection_policy": "hard_filter_then_seeded_random",
        "model_performance_filtering": False,
        "frontier_score_filtering": False,
        "source_dca_round": 3,
        "source_dca_checkpoint_manifest": str(dca_manifest_path),
        "source_dca_checkpoint_manifest_sha256": sha256_file(dca_manifest_path),
        "source_dca_adapter_sha256": dca_manifest.get("adapter_sha256"),
        "candidate_pool": str(candidate_path),
        "candidate_pool_sha256": sha256_file(candidate_path),
        "candidate_shards": {
            str(index): {"path": str(path), "sha256": sha256_file(path)}
            for index, path in enumerate(shard_paths)
        },
        "initial_candidate_count": args.candidate_count,
        "candidate_count": len(candidate_records),
        "candidate_count_per_task": args.candidate_count // len(TASK_IDS),
        "additional_candidate_pools": additional_pool_entries,
        "automatic_task_topups": automatic_topup_entries,
        "topup_size": args.topup_size,
        "per_task": args.per_task,
        "selected_count": len(selected),
        "eligible_counts": eligible_counts,
        "task_counts": dict(Counter(item["task_id"] for item in selected)),
        "rejected_counts": dict(rejected),
        "exclusion_ledger": str(exclusion_path),
        "exclusion_ledger_sha256": sha256_file(exclusion_path),
        "exclusion_lineage_count": len(excluded_lineage),
        "exclusion_semantic_count": len(excluded_semantic),
        "v5c_calibration_present": bool(calibration_paths),
        "v5c_calibration_paths": [str(path) for path in calibration_paths],
        "future_v5c_calibration_must_exclude_sealed_tmcd_test": True,
        "selected_lineage_fingerprints_sha256": sha256_bytes(
            canonical_json(selected_lineage).encode("utf-8")
        ),
        "selected_semantic_fingerprints_sha256": sha256_bytes(
            canonical_json(selected_semantic).encode("utf-8")
        ),
        "selection_audit": str(audit_path),
        "selection_audit_sha256": sha256_file(audit_path),
        "source_tree_sha256": sha256_source_tree(root / "agentguard_zero"),
        "generator_script_sha256": sha256_file(Path(__file__).resolve()),
        "final_heldout": str(final_path),
        "final_heldout_sha256": sha256_file(final_path),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
