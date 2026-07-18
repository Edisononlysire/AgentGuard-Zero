#!/usr/bin/env python3
"""Top up a fresh DCA pool until every VDA split task quota is feasible."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.protocol import TMCD_RELEASE_REVISION
from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
)
from scripts.build_vda_round_pool import TASK_IDS, audit_candidate_task_quotas


POOL_SIGNATURE_KEYS = (
    "backbone",
    "experiment_variant",
    "generation_prompt_version",
    "candidate_normalization_version",
    "tmcd_release_revision",
    "max_attempts",
    "source_dca_round",
    "source_dca_checkpoint_manifest_sha256",
)


def _run_parallel(jobs: list[tuple[list[str], dict[str, str]]]) -> None:
    processes: list[tuple[list[str], subprocess.Popen[Any]]] = []
    try:
        for command, environment in jobs:
            print(json.dumps({"parallel_command": command}, ensure_ascii=False), flush=True)
            processes.append((command, subprocess.Popen(command, env=environment)))
        failures = []
        for command, process in processes:
            return_code = process.wait()
            if return_code:
                failures.append((command, return_code))
        if failures:
            command, return_code = failures[0]
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        for _command, process in processes:
            if process.poll() is None:
                process.terminate()
        for _command, process in processes:
            if process.poll() is None:
                process.wait()


def planned_topup_count(
    *,
    deficit: int,
    eligible: int,
    attempted: int,
    minimum: int,
    safety_factor: float,
) -> int:
    """Estimate a deterministic top-up size from the observed hard-pass rate."""

    if deficit <= 0:
        return 0
    if minimum <= 0 or safety_factor < 1.0:
        raise ValueError("minimum must be positive and safety_factor must be at least one")
    observed_rate = min(1.0, eligible / max(1, attempted))
    conservative_rate = max(0.25, observed_rate)
    return max(minimum, math.ceil(deficit * safety_factor / conservative_rate))


def _validate_pool_signature(
    pool: dict[str, Any],
    *,
    expected: dict[str, Any],
    path: Path,
) -> None:
    if pool.get("kind") != "dca_candidate_pool":
        raise LineageError(f"quota candidate source is not a merged DCA pool: {path}")
    if int(pool.get("num_candidates_generated", -1)) != len(
        pool.get("candidates", []) or []
    ):
        raise LineageError(f"quota candidate source count mismatch: {path}")
    for key in POOL_SIGNATURE_KEYS:
        if pool.get(key) != expected.get(key):
            raise LineageError(
                f"quota candidate source signature mismatch for {key}: {path}"
            )


def merge_candidate_pools(
    *,
    initial_path: Path,
    topup_entries: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Merge immutable initial/top-up pools while preserving complete provenance."""

    initial = read_json(initial_path)
    expected = {key: initial.get(key) for key in POOL_SIGNATURE_KEYS}
    _validate_pool_signature(initial, expected=expected, path=initial_path)
    if expected["tmcd_release_revision"] != TMCD_RELEASE_REVISION:
        raise LineageError("candidate quota merge uses a stale TMCD release")

    sources = [
        {
            "kind": "initial",
            "path": str(initial_path),
            "sha256": sha256_file(initial_path),
            "count": len(initial.get("candidates", []) or []),
            "seed": initial.get("seed"),
            "task_id": initial.get("task_id"),
        }
    ]
    pools: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (initial, sources[0])
    ]
    for entry in topup_entries:
        path = Path(str(entry["path"])).resolve()
        pool = read_json(path)
        _validate_pool_signature(pool, expected=expected, path=path)
        task_id = str(entry.get("task_id", ""))
        if task_id not in TASK_IDS or pool.get("task_id") != task_id:
            raise LineageError(f"quota top-up task mismatch: {path}")
        if int(pool.get("seed", -1)) != int(entry.get("seed", -2)):
            raise LineageError(f"quota top-up seed mismatch: {path}")
        source = {
            "kind": "quota_topup",
            "path": str(path),
            "sha256": sha256_file(path),
            "count": len(pool.get("candidates", []) or []),
            "seed": int(entry["seed"]),
            "task_id": task_id,
            "topup_round": int(entry["topup_round"]),
        }
        sources.append(source)
        pools.append((pool, source))

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for pool, source in pools:
        for record in pool.get("candidates", []) or []:
            merged = copy.deepcopy(record)
            fingerprint = str(merged.get("scenario_fingerprint", ""))
            duplicate = (
                bool(merged.get("duplicate", False))
                or not fingerprint
                or fingerprint in seen
            )
            seen.add(fingerprint)
            duplicate_count += int(duplicate)
            merged["duplicate"] = duplicate
            merged["source_candidate_index"] = merged.get("candidate_index")
            merged["combined_candidate_index"] = len(records)
            merged["candidate_pool_source_sha256"] = source["sha256"]
            merged["candidate_pool_source_kind"] = source["kind"]
            records.append(merged)

    result = {
        **{key: initial.get(key) for key in initial if key != "candidates"},
        "schema_version": 2,
        "kind": "dca_candidate_pool",
        "created_at": utc_now(),
        "seed": initial.get("seed"),
        "task_id": None,
        "initial_candidate_pool": str(initial_path),
        "initial_candidate_pool_sha256": sha256_file(initial_path),
        "initial_candidate_count": len(initial.get("candidates", []) or []),
        "num_candidates_requested": sum(source["count"] for source in sources),
        "num_candidates_generated": len(records),
        "num_parse_ok": sum(bool(item.get("parse_ok", False)) for item in records),
        "num_all_checks_ok": sum(
            bool((item.get("checks", {}) or {}).get("all_ok", False))
            for item in records
        ),
        "num_duplicates": duplicate_count,
        "quota_topup_count": len(records)
        - len(initial.get("candidates", []) or []),
        "candidate_sources": sources,
        "candidates": records,
    }
    atomic_write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-candidate-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--feedback-log", type=Path, required=True)
    parser.add_argument("--dca-checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--dev-size", type=int, required=True)
    parser.add_argument("--xplay-size", type=int, required=True)
    parser.add_argument("--allocated-gpus", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-topup-size", type=int, default=250)
    parser.add_argument("--max-topup-rounds", type=int, default=3)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--partial-fsync-every-batches", type=int, default=16)
    parser.add_argument("--experiment-variant", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids = [value.strip() for value in args.allocated_gpus.split(",") if value.strip()]
    if len(gpu_ids) != 4:
        raise SystemExit("candidate quota completion requires exactly four GPUs")
    if args.min_topup_size <= 0 or args.max_topup_rounds < 0:
        raise SystemExit("invalid candidate quota top-up limits")
    if args.safety_factor < 1.0:
        raise SystemExit("candidate quota safety factor must be at least one")

    initial_path = args.initial_candidate_pool.resolve()
    output_path = args.output.resolve()
    audit_path = args.audit_output.resolve()
    feedback_path = args.feedback_log.resolve()
    dca_manifest_path = args.dca_checkpoint_manifest.resolve()
    split_counts = {
        "train": args.train_size,
        "dev": args.dev_size,
        "xplay": args.xplay_size,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topup_entries: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for topup_round in range(args.max_topup_rounds + 1):
        merge_candidate_pools(
            initial_path=initial_path,
            topup_entries=topup_entries,
            output_path=output_path,
        )
        audit = audit_candidate_task_quotas(
            candidate_path=output_path,
            feedback_path=feedback_path,
            dca_manifest_path=dca_manifest_path,
            split_counts=split_counts,
        )
        attempts.append(
            {
                "topup_round": topup_round,
                "candidate_pool": str(output_path),
                **audit,
            }
        )
        shortages = audit["shortages"]
        if not shortages:
            break
        if topup_round == args.max_topup_rounds:
            raise LineageError(
                "VDA candidate task quotas remain short after "
                f"{args.max_topup_rounds} top-up rounds: {shortages}"
            )

        jobs = []
        round_entries = []
        for task_id in TASK_IDS:
            if task_id not in shortages:
                continue
            task_index = TASK_IDS.index(task_id)
            deficit = int(shortages[task_id]["deficit"])
            count = planned_topup_count(
                deficit=deficit,
                eligible=int(audit["eligible_task_counts"].get(task_id, 0)),
                attempted=int(audit["attempted_task_counts"].get(task_id, 0)),
                minimum=args.min_topup_size,
                safety_factor=args.safety_factor,
            )
            next_round = topup_round + 1
            topup_seed = (
                args.seed
                + next_round * 100_000_007
                + (task_index + 1) * 1_000_003
            )
            topup_path = output_path.with_name(
                f"candidates.topup_{task_id}_{next_round:02d}.json"
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[task_index]
            triton_cache = Path(
                os.environ.get("AGZ_TRITON_CACHE_ROOT", "/tmp/agentguard_zero_triton")
            ) / "vda_candidate_quota" / f"topup_{task_id}_{next_round:02d}"
            triton_cache.mkdir(parents=True, exist_ok=True)
            environment["TRITON_CACHE_DIR"] = str(triton_cache)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "generate_dca_scenarios.py"),
                "--checkpoint-manifest",
                str(dca_manifest_path),
                "--output",
                str(topup_path),
                "--num-candidates",
                str(count),
                "--batch-size",
                str(args.batch_size),
                "--num-shards",
                "1",
                "--shard-index",
                "0",
                "--seed",
                str(topup_seed),
                "--max-input-tokens",
                str(args.max_input_tokens),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--attn-implementation",
                args.attn_implementation,
                "--partial-fsync-every-batches",
                str(args.partial_fsync_every_batches),
                "--max-attempts",
                str(args.max_attempts),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--top-k",
                str(args.top_k),
                "--experiment-variant",
                args.experiment_variant,
                "--task-id",
                task_id,
            ]
            entry = {
                "task_id": task_id,
                "seed": topup_seed,
                "path": str(topup_path),
                "requested_count": count,
                "topup_round": next_round,
            }
            if topup_path.exists():
                existing = read_json(topup_path)
                reusable = (
                    existing.get("task_id") == task_id
                    and int(existing.get("seed", -1)) == topup_seed
                    and int(existing.get("num_candidates_requested", -1)) == count
                    and existing.get("source_dca_checkpoint_manifest_sha256")
                    == audit["source_dca_checkpoint_manifest_sha256"]
                )
                if not reusable:
                    raise LineageError(
                        f"existing quota top-up has a different signature: {topup_path}"
                    )
            else:
                jobs.append((command, environment))
            round_entries.append(entry)
        _run_parallel(jobs)
        for entry in round_entries:
            path = Path(entry["path"])
            entry["sha256"] = sha256_file(path)
            entry["generated_count"] = len(read_json(path).get("candidates", []) or [])
            topup_entries.append(entry)
    else:  # pragma: no cover - the explicit failure above is the only loop exit
        raise AssertionError("candidate quota loop exited unexpectedly")

    final_audit = attempts[-1]
    manifest = {
        "schema_version": 1,
        "kind": "vda_candidate_quota_audit",
        "created_at": utc_now(),
        "status": "complete",
        "model_performance_filtering": False,
        "vda_rollout_filtering": False,
        "hard_filters_unchanged": True,
        "initial_candidate_pool": str(initial_path),
        "initial_candidate_pool_sha256": sha256_file(initial_path),
        "initial_candidate_count": len(read_json(initial_path).get("candidates", []) or []),
        "final_candidate_pool": str(output_path),
        "final_candidate_pool_sha256": sha256_file(output_path),
        "final_candidate_count": int(final_audit["candidate_count"]),
        "source_dca_checkpoint_manifest": str(dca_manifest_path),
        "source_dca_checkpoint_manifest_sha256": sha256_file(dca_manifest_path),
        "split_counts": split_counts,
        "topup_config": {
            "seed": args.seed,
            "min_topup_size": args.min_topup_size,
            "max_topup_rounds": args.max_topup_rounds,
            "safety_factor": args.safety_factor,
        },
        "topups": topup_entries,
        "attempts": attempts,
        "final_eligible_task_counts": final_audit["eligible_task_counts"],
        "required_task_counts": final_audit["required_task_counts"],
        "final_shortages": final_audit["shortages"],
        "excluded_counts": final_audit["excluded_counts"],
    }
    atomic_write_json(audit_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
