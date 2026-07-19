#!/usr/bin/env python3
"""Run one isolated DCA-first candidate-ranker co-evolution round."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import RoundLayout, atomic_write_json, sha256_file, utc_now


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True, env=env)


def evaluate(
    *,
    python: str,
    root: Path,
    model_path: Path,
    ranker_manifest: Path,
    output_dir: Path,
    group_offset: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[Any]] = []
    shards = []
    for index in range(4):
        shard = output_dir / f"shard_{index}.json"
        shards.append(shard)
        command = [
            python,
            "-s",
            str(root / "scripts/eval_candidate_policy.py"),
            "run",
            "--model-path",
            str(model_path),
            "--ranker-manifest",
            str(ranker_manifest),
            "--output",
            str(shard),
            "--scenario-count",
            "32",
            "--group-offset",
            str(group_offset),
            "--shard-index",
            str(index),
            "--shard-count",
            "4",
            "--device",
            f"cuda:{index}",
        ]
        processes.append(subprocess.Popen(command))
    failures = [process.wait() for process in processes]
    if any(failures):
        raise subprocess.CalledProcessError(next(code for code in failures if code), "candidate evaluation")
    merged = output_dir / "metrics.json"
    run(
        [
            python,
            "-s",
            str(root / "scripts/eval_candidate_policy.py"),
            "merge",
            "--inputs",
            *map(str, shards),
            "--output",
            str(merged),
        ]
    )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-round", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--source-ranker-manifest", type=Path, required=True)
    parser.add_argument("--canonical-replay", type=Path, required=True)
    parser.add_argument("--past-candidate-sets", type=Path, nargs="*")
    parser.add_argument("--dagger-candidate-sets", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    root = args.root.resolve()
    target_round = args.source_round + 1
    layout = RoundLayout(
        root=root,
        backbone="qwen3.5-4b",
        source_round=args.source_round,
        artifact_scope="candidate_min3",
        experiment_variant="full",
    )
    source_vda_manifest = layout.checkpoint_dir("vda", args.source_round) / "manifest.json"
    if not source_vda_manifest.exists():
        command = [
            args.python,
            "-s",
            str(root / "scripts/write_candidate_lineage_manifest.py"),
            "--ranker-manifest",
            str(args.source_ranker_manifest),
            "--output",
            str(source_vda_manifest),
            "--model-path",
            str(args.model_path),
            "--round-index",
            str(args.source_round),
            "--seed",
            str(args.seed),
        ]
        if args.source_round > 0:
            raise RuntimeError(
                f"source VDA lineage is missing for round {args.source_round}: {source_vda_manifest}"
            )
        run(command)

    evaluation_root = root / "evaluations/candidate_min3" / f"round_{target_round}"
    start_metrics = evaluate(
        python=args.python,
        root=root,
        model_path=args.model_path,
        ranker_manifest=args.source_ranker_manifest,
        output_dir=evaluation_root / "start",
        group_offset=20000,
    )

    environment = os.environ.copy()
    environment.update(
        {
            "AGZ_ROOT": str(root),
            "AGZ_VDA_POLICY_BACKEND": "candidate_ranker",
            "AGZ_VDA_RANKER_MANIFEST": str(args.source_ranker_manifest.resolve()),
            "AGZ_RESOURCE_ROOT": environment.get(
                "AGZ_RESOURCE_ROOT",
                "/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero",
            ),
        }
    )
    run(
        [
            args.python,
            "-s",
            str(root / "scripts/run_dca_first_round.py"),
            "--root",
            str(root),
            "--backbone",
            "qwen3.5-4b",
            "--artifact-scope",
            "candidate_min3",
            "--source-round",
            str(args.source_round),
            "--model-path",
            str(args.model_path),
            "--seed",
            str(args.seed),
            "--allocated-gpus",
            "0,1,2,3",
            "--dca-feedback-candidates",
            "64",
            "--dca-rollout-n",
            "2",
            "--dca-batch-size",
            "8",
            "--dca-steps",
            "4",
            "--vda-candidates",
            "256",
            "--vda-train-size",
            "128",
            "--vda-dev-size",
            "32",
            "--vda-xplay-size",
            "32",
            "--vda-batch-size",
            "32",
            "--vda-steps",
            "4",
            "--vda-selection-policy",
            "pilot_balanced_50_40_10",
            "--candidate-batch-size",
            "4",
            "--candidate-quota-min-topup-size",
            "32",
            "--candidate-quota-max-topup-rounds",
            "1",
            "--stop-after-stage",
            "build_isolated_vda_pool",
        ],
        env=environment,
    )

    feedback_log = layout.data_dir / "dca_feedback/feedback.jsonl"
    feedback_gate = layout.data_dir / "dca_feedback/candidate_gate.json"
    run(
        [
            args.python,
            "-s",
            str(root / "scripts/summarize_candidate_dca_feedback.py"),
            "--feedback-log",
            str(feedback_log),
            "--output",
            str(feedback_gate),
        ]
    )

    fresh_dir = layout.data_dir / "candidate_fresh"
    run(
        [
            args.python,
            "-s",
            str(root / "scripts/build_candidate_dataset.py"),
            "--output-dir",
            str(fresh_dir),
            "--scenario-source",
            str(layout.data_dir / "vda_train/train.parquet"),
            "--max-records",
            "96",
        ]
    )
    mix_dir = layout.data_dir / "candidate_mix"
    mix_command = [
        args.python,
        "-s",
        str(root / "scripts/mix_candidate_replay.py"),
        "--fresh",
        str(fresh_dir / "candidate_sets.jsonl"),
        "--canonical",
        str(args.canonical_replay),
        "--output-dir",
        str(mix_dir),
        "--total-records",
        "128",
        "--seed",
        str(args.seed + target_round),
    ]
    if args.past_candidate_sets:
        mix_command.extend(["--past", *map(str, args.past_candidate_sets)])
    if args.dagger_candidate_sets:
        mix_command.extend(["--dagger", str(args.dagger_candidate_sets)])
    run(mix_command)

    source_ranker = json.loads(args.source_ranker_manifest.read_text(encoding="utf-8"))
    target_ranker_dir = root / "outputs/candidate_min3" / f"ranker_round_{target_round}"
    run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            str(root / "scripts/train_candidate_ranker.py"),
            "--model-path",
            str(args.model_path),
            "--train-jsonl",
            str(mix_dir / "candidate_sets.jsonl"),
            "--data-manifest",
            str(mix_dir / "manifest.json"),
            "--output-dir",
            str(target_ranker_dir),
            "--init-adapter",
            str(source_ranker["adapter_path"]),
            "--init-heads",
            str(source_ranker.get("heads_path", source_ranker["score_head_path"])),
            "--objective",
            "joint",
            "--learning-rate",
            "5e-6",
            "--epochs",
            "1",
            "--seed",
            str(args.seed + target_round),
        ],
        env=environment,
    )
    target_ranker_manifest = target_ranker_dir / "manifest.json"
    end_metrics = evaluate(
        python=args.python,
        root=root,
        model_path=args.model_path,
        ranker_manifest=target_ranker_manifest,
        output_dir=evaluation_root / "end",
        group_offset=20000,
    )
    round_gate = evaluation_root / "round_gate.json"
    run(
        [
            args.python,
            "-s",
            str(root / "scripts/eval_candidate_gates.py"),
            "round",
            "--start",
            str(start_metrics),
            "--end",
            str(end_metrics),
            "--round-index",
            str(target_round),
            "--output",
            str(round_gate),
        ]
    )
    target_vda_manifest = layout.checkpoint_dir("vda", target_round) / "manifest.json"
    run(
        [
            args.python,
            "-s",
            str(root / "scripts/write_candidate_lineage_manifest.py"),
            "--ranker-manifest",
            str(target_ranker_manifest),
            "--output",
            str(target_vda_manifest),
            "--model-path",
            str(args.model_path),
            "--round-index",
            str(target_round),
            "--parent-manifest",
            str(source_vda_manifest),
            "--training-data-manifest",
            str(mix_dir / "manifest.json"),
            "--seed",
            str(args.seed + target_round),
        ]
    )
    summary = {
        "schema_version": 1,
        "kind": "candidate_coevolution_round",
        "created_at": utc_now(),
        "source_round": args.source_round,
        "target_round": target_round,
        "source_ranker_manifest_sha256": sha256_file(args.source_ranker_manifest),
        "target_ranker_manifest": str(target_ranker_manifest),
        "target_ranker_manifest_sha256": sha256_file(target_ranker_manifest),
        "feedback_gate": str(feedback_gate),
        "round_gate": str(round_gate),
        "start_metrics": str(start_metrics),
        "end_metrics": str(end_metrics),
    }
    atomic_write_json(evaluation_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
