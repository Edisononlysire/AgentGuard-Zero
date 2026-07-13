#!/usr/bin/env python3
"""Run three serial DCA-first rounds for one backbone on four GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--backbone", choices=["qwen3.5-4b", "qwen3.5-9b"], required=True)
    parser.add_argument(
        "--experiment-variant",
        choices=["full", "append_only_memory"],
        default="full",
    )
    parser.add_argument("--model-path", default="")
    parser.add_argument(
        "--artifact-scope",
        choices=["formal", "pilot", "tmcd_v2", "tmcd_v2_pilot"],
        default="tmcd_v2",
    )
    parser.add_argument("--allocated-gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--end-round", type=int, default=3)
    parser.add_argument("--dca-feedback-candidates", type=int, default=4000)
    parser.add_argument("--dca-rollout-n", type=int, default=2)
    parser.add_argument("--dca-batch-size", type=int, default=40)
    parser.add_argument("--dca-steps", type=int, default=50)
    parser.add_argument("--vda-candidates", type=int, default=10000)
    parser.add_argument("--vda-train-size", type=int, default=2400)
    parser.add_argument("--vda-dev-size", type=int, default=400)
    parser.add_argument("--vda-xplay-size", type=int, default=800)
    parser.add_argument("--vda-batch-size", type=int, default=32)
    parser.add_argument("--vda-steps", type=int, default=75)
    parser.add_argument("--vda-rollout-n", type=int, default=1)
    parser.add_argument("--vda-max-turns", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.start_round < args.end_round <= 3:
        raise SystemExit("round bounds must satisfy 0 <= start-round < end-round <= 3")

    root = Path(args.root).resolve()
    runner = root / "scripts" / "run_dca_first_round.py"
    common = [
        "--root",
        str(root),
        "--backbone",
        args.backbone,
        "--experiment-variant",
        args.experiment_variant,
        "--artifact-scope",
        args.artifact_scope,
        "--allocated-gpus",
        args.allocated_gpus,
        "--seed",
        str(args.seed),
        "--dca-feedback-candidates",
        str(args.dca_feedback_candidates),
        "--dca-rollout-n",
        str(args.dca_rollout_n),
        "--dca-batch-size",
        str(args.dca_batch_size),
        "--dca-steps",
        str(args.dca_steps),
        "--vda-candidates",
        str(args.vda_candidates),
        "--vda-train-size",
        str(args.vda_train_size),
        "--vda-dev-size",
        str(args.vda_dev_size),
        "--vda-xplay-size",
        str(args.vda_xplay_size),
        "--vda-batch-size",
        str(args.vda_batch_size),
        "--vda-steps",
        str(args.vda_steps),
        "--vda-rollout-n",
        str(args.vda_rollout_n),
        "--vda-max-turns",
        str(args.vda_max_turns),
        "--candidate-batch-size",
        str(args.candidate_batch_size),
    ]
    if args.model_path:
        common.extend(["--model-path", args.model_path])

    for source_round in range(args.start_round, args.end_round):
        command = [sys.executable, str(runner), "--source-round", str(source_round), *common]
        print(f"[AgentGuard-Zero] starting source round {source_round}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
