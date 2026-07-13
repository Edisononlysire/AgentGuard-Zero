#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one TMCD system as four resumable GPU shards.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "tmcd_eval"))
    parser.add_argument("--model-path", default="")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--model-backend", default="hf")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", default="all")
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--trajectory-batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",") if item.strip()]
    if len(gpu_ids) != 4:
        raise SystemExit(f"expected exactly four visible GPUs, got {gpu_ids}")
    run_dir = Path(args.output_dir).resolve() / args.run_name
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    model_cache_key = Path(args.model_path).name if args.model_path else args.system
    for shard_index, gpu_id in enumerate(gpu_ids):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "eval_tmcd_systems.py"),
            "--data", args.data,
            "--system", args.system,
            "--run_name", args.run_name,
            "--output_dir", args.output_dir,
            "--model_backend", args.model_backend,
            "--candidate_count", str(args.candidate_count),
            "--limit", str(args.limit),
            "--split", args.split,
            "--max_turns", str(args.max_turns),
            "--trajectory_batch_size", str(args.trajectory_batch_size),
            "--max_input_tokens", str(args.max_input_tokens),
            "--max_new_tokens", str(args.max_new_tokens),
            "--seed", str(args.seed),
            "--num_shards", "4",
            "--shard_index", str(shard_index),
        ]
        if args.model_path:
            command.extend(["--model_path", args.model_path])
        if args.adapter_path:
            command.extend(["--adapter_path", args.adapter_path])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        cache_dir = (
            Path(environment.get("AGZ_TRITON_CACHE_ROOT", "/tmp/agentguard_zero_triton"))
            / "tmcd_eval"
            / model_cache_key
            / f"rank_{shard_index}"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        environment["TRITON_CACHE_DIR"] = str(cache_dir)
        handle = (log_dir / f"shard_{shard_index}.log").open("a", encoding="utf-8")
        handles.append(handle)
        processes.append(subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT))
    failures = []
    for shard_index, process in enumerate(processes):
        code = process.wait()
        if code:
            failures.append((shard_index, code))
    for handle in handles:
        handle.close()
    if failures:
        raise RuntimeError(f"TMCD evaluation shards failed: {failures}")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "merge_tmcd_eval_shards.py"),
        "--run-dir", str(run_dir),
    ]
    if args.limit > 0:
        command.extend(["--expected-count", str(args.limit)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
