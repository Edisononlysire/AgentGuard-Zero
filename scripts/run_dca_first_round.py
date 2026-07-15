#!/usr/bin/env python3
"""Run one resumable DCA-first alternating co-evolution round."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import (
    LineageError,
    RoundLayout,
    atomic_write_json,
    feedback_fingerprints,
    latest_global_step,
    load_checkpoint_manifest,
    mark_stage,
    read_jsonl,
    scenario_fingerprint,
    sha256_file,
    sha256_tree,
    stage_complete,
    utc_now,
    validate_round_lineage,
    write_base_manifest,
    write_frozen_manifest,
    write_trained_manifest,
)
from agentguard_zero.training.dca_dataset import DCA_PROMPT_VERSION, write_dca_prompt_dataset
from agentguard_zero.protocol import TMCD_RELEASE_REVISION
from agentguard_zero.variants import TRAINING_VARIANTS, experiment_variant
from scripts.generate_dca_scenarios import DCA_CANDIDATE_NORMALIZATION_VERSION


BACKBONE_ENV = {
    "qwen3.5-4b": "AGZ_QWEN35_4B_PATH",
    "qwen3.5-9b": "AGZ_QWEN35_9B_PATH",
}


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True, env=env)


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


def _maybe_latest(checkpoint_root: Path) -> Path | None:
    try:
        return latest_global_step(checkpoint_root)
    except LineageError:
        return None


def _step(path: Path | None) -> int:
    if path is None:
        return 0
    return int(path.name.split("global_step_", 1)[1])


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _reconcile_feedback_log(
    *,
    feedback_log_path: Path,
    checkpoint_root: Path,
    parent_step: int,
    batch_size: int,
    rollout_n: int,
) -> None:
    latest = _maybe_latest(checkpoint_root)
    if latest is None:
        return
    completed_round_steps = _step(latest) - parent_step
    if completed_round_steps < 0:
        raise LineageError(
            f"checkpoint step {_step(latest)} precedes parent step {parent_step}"
        )
    expected_rows = completed_round_steps * batch_size * rollout_n
    rows = read_jsonl(feedback_log_path) if feedback_log_path.exists() else []
    if len(rows) < expected_rows:
        raise LineageError(
            f"DCA feedback log has {len(rows)} rows but checkpoint {latest.name} "
            f"requires {expected_rows}"
        )
    if len(rows) == expected_rows:
        return

    orphaned_path = feedback_log_path.with_name(
        f"feedback.orphaned_after_{latest.name}.{utc_now().replace(':', '').replace('-', '')}.jsonl"
    )
    _atomic_write_jsonl(orphaned_path, rows[expected_rows:])
    _atomic_write_jsonl(feedback_log_path, rows[:expected_rows])
    print(
        json.dumps(
            {
                "event": "dca_feedback_log_reconciled",
                "checkpoint": str(latest),
                "kept_rows": expected_rows,
                "orphaned_rows": len(rows) - expected_rows,
                "orphaned_path": str(orphaned_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _resume_link(data_dir: Path, role: str, parent_checkpoint: Path) -> Path:
    step = _step(parent_checkpoint)
    target = data_dir / "_resume" / role / f"global_step_{step}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("actor", "critic"):
        source = parent_checkpoint / name
        link = target / name
        if not source.exists():
            continue
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise LineageError(f"resume link already exists with different target: {link}")
        link.symlink_to(source.resolve(), target_is_directory=True)
    if not (target / "actor").exists():
        raise LineageError(f"parent actor checkpoint is missing: {parent_checkpoint}")
    return target


def _resume_plan(
    *,
    role: str,
    source_round: int,
    parent_manifest: dict[str, Any],
    target_checkpoint_root: Path,
    data_dir: Path,
    round_steps: int,
) -> tuple[str, str, int, bool]:
    target_latest = _maybe_latest(target_checkpoint_root)
    if target_latest is not None:
        target_step = _step(target_latest)
        parent_step = (
            _step(Path(parent_manifest["checkpoint_path"])) if source_round > 0 else 0
        )
        required_step = parent_step + round_steps
        if target_step > required_step:
            raise LineageError(
                f"{role} target checkpoint step {target_step} exceeds required step {required_step}"
            )
        return "auto", "null", required_step, target_step >= required_step

    if source_round == 0:
        return "disable", "null", round_steps, False
    parent_checkpoint = Path(parent_manifest["checkpoint_path"])
    parent_step = _step(parent_checkpoint)
    resume_path = _resume_link(data_dir, role, parent_checkpoint)
    return "resume_path", str(resume_path), parent_step + round_steps, False


def _checkpoint_manifest_path(layout: RoundLayout, role: str, round_index: int) -> Path:
    return layout.checkpoint_dir(role, round_index) / "manifest.json"


def _load_parent(layout: RoundLayout, role: str, model_path: str, seed: int) -> tuple[Path, dict[str, Any]]:
    path = _checkpoint_manifest_path(layout, role, layout.source_round)
    path.parent.mkdir(parents=True, exist_ok=True)
    if layout.source_round == 0:
        manifest = write_base_manifest(
            path,
            role=role,
            backbone=layout.backbone,
            model_path=model_path,
            seed=seed,
        )
    else:
        manifest = load_checkpoint_manifest(
            path,
            role=role,
            backbone=layout.backbone,
            round_index=layout.source_round,
        )
    return path, manifest


def _mark_failed(state_path: Path, stage: str, exc: Exception) -> None:
    mark_stage(
        state_path,
        stage,
        "failed",
        error_type=type(exc).__name__,
        error=str(exc),
    )


def _prune_parent_recovery_checkpoint(
    manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    checkpoint_path = Path(str(manifest.get("checkpoint_path", ""))).resolve()
    adapter_path = Path(str(manifest.get("adapter_path", ""))).resolve()
    expected_adapter_sha = str(manifest.get("adapter_sha256", ""))
    if not checkpoint_path.is_dir():
        return {"status": "already_pruned", "checkpoint_path": str(checkpoint_path)}
    if checkpoint_path in adapter_path.parents:
        raise LineageError(
            f"refusing to prune checkpoint containing its only adapter copy: {checkpoint_path}"
        )
    if not adapter_path.is_dir() or sha256_tree(adapter_path) != expected_adapter_sha:
        raise LineageError(f"stable adapter verification failed before pruning: {adapter_path}")
    trainer_root = checkpoint_path.parent
    removed = []
    for step in sorted(trainer_root.glob("global_step_*")):
        if step.is_dir():
            removed.append({"path": str(step), "bytes": sum(p.stat().st_size for p in step.rglob("*") if p.is_file())})
            shutil.rmtree(step)
    tracker = trainer_root / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        tracker.unlink()
    report = {
        "schema_version": 1,
        "kind": "pruned_parent_recovery_checkpoint",
        "pruned_at": utc_now(),
        "adapter_path": str(adapter_path),
        "adapter_sha256": expected_adapter_sha,
        "removed_steps": removed,
        "removed_bytes": sum(item["bytes"] for item in removed),
    }
    atomic_write_json(manifest_path.resolve().parent / "child_pruned_recovery.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--backbone", choices=sorted(BACKBONE_ENV), required=True)
    parser.add_argument(
        "--experiment-variant",
        choices=TRAINING_VARIANTS,
        default="full",
    )
    parser.add_argument(
        "--artifact-scope",
        choices=["formal", "pilot", "tmcd_v2", "tmcd_v2_pilot"],
        default="tmcd_v2",
    )
    parser.add_argument("--model-path", default="")
    parser.add_argument("--source-round", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument(
        "--allocated-gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    )
    parser.add_argument("--dca-feedback-candidates", type=int, default=4000)
    parser.add_argument("--dca-rollout-n", type=int, default=2)
    parser.add_argument("--dca-batch-size", type=int, default=40)
    parser.add_argument("--dca-steps", type=int, default=50)
    parser.add_argument("--vda-candidates", type=int, default=10000)
    parser.add_argument("--vda-train-size", type=int, default=2400)
    parser.add_argument("--vda-dev-size", type=int, default=400)
    parser.add_argument("--vda-xplay-size", type=int, default=800)
    parser.add_argument("--vda-batch-size", type=int, default=32)
    parser.add_argument("--vda-rollout-n", type=int, default=1)
    parser.add_argument("--vda-max-turns", type=int, default=16)
    parser.add_argument("--vda-steps", type=int, default=75)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--feedback-port", type=int, default=0)
    parser.add_argument(
        "--stop-after-stage",
        choices=["build_isolated_vda_pool"],
        default=None,
        help="Stop after producing and validating the isolated VDA pool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant = experiment_variant(args.experiment_variant)
    if args.source_round < 0:
        raise SystemExit("--source-round must be non-negative")
    gpu_ids = [value.strip() for value in args.allocated_gpus.split(",") if value.strip()]
    if len(gpu_ids) != 4:
        raise SystemExit(f"one backbone must receive exactly four GPUs, got {args.allocated_gpus}")
    if args.dca_feedback_candidates % args.dca_rollout_n:
        raise SystemExit("DCA feedback candidate count must be divisible by rollout_n")
    prompt_rows = args.dca_feedback_candidates // args.dca_rollout_n
    if prompt_rows % args.dca_batch_size:
        raise SystemExit("DCA prompt rows must be divisible by DCA batch size")
    if args.vda_train_size % args.vda_batch_size:
        raise SystemExit("VDA train size must be divisible by VDA batch size")
    if args.dca_steps * args.dca_batch_size * args.dca_rollout_n != args.dca_feedback_candidates:
        raise SystemExit(
            "DCA max-step budget must consume exactly the isolated feedback pool: "
            "dca_steps * dca_batch_size * rollout_n == dca_feedback_candidates"
        )
    if args.vda_steps * args.vda_batch_size != args.vda_train_size:
        raise SystemExit(
            "VDA max-step budget must consume exactly the train split once: "
            "vda_steps * vda_batch_size == vda_train_size"
        )
    if args.vda_max_turns <= 0:
        raise SystemExit("VDA max turns must be positive")

    root = Path(args.root).resolve()
    model_path = args.model_path or os.environ.get(BACKBONE_ENV[args.backbone], "")
    if not model_path or not Path(model_path).exists():
        raise SystemExit(f"model path is missing for {args.backbone}: {model_path}")
    layout = RoundLayout(
        root=root,
        backbone=args.backbone,
        source_round=args.source_round,
        artifact_scope=args.artifact_scope,
        experiment_variant=args.experiment_variant,
    )
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    state_path = layout.state_path
    target_round = layout.target_round
    execution_host = socket.gethostname()

    dca_parent_path, dca_parent = _load_parent(layout, "dca", model_path, args.seed)
    vda_parent_path, vda_parent = _load_parent(layout, "vda", model_path, args.seed)
    dca_target_dir = layout.checkpoint_dir("dca")
    vda_target_dir = layout.checkpoint_dir("vda")
    dca_target_dir.mkdir(parents=True, exist_ok=True)
    vda_target_dir.mkdir(parents=True, exist_ok=True)

    feedback_dir = layout.data_dir / "dca_feedback"
    prompt_path = feedback_dir / "prompts.parquet"
    prompt_manifest_path = feedback_dir / "prompt_manifest.json"
    feedback_log_path = feedback_dir / "feedback.jsonl"
    feedback_manifest_path = feedback_dir / "manifest.json"
    candidate_path = layout.data_dir / "vda_candidates" / "candidates.json"
    candidate_shards = [
        candidate_path.with_name(f"candidates.shard_{index}.json")
        for index in range(len(gpu_ids))
    ]
    pool_manifest_path = layout.data_dir / "vda_pool_manifest.json"
    split_paths = {
        "train": layout.data_dir / "vda_train" / "train.parquet",
        "dev": layout.data_dir / "vda_dev" / "dev.parquet",
        "xplay": layout.data_dir / "vda_xplay" / "xplay.parquet",
    }

    stage = "prepare_dca_feedback_prompts"
    try:
        expected_prompt_signature = {
            "tmcd_release_revision": TMCD_RELEASE_REVISION,
            "prompt_version": DCA_PROMPT_VERSION,
            "num_rows": prompt_rows,
            "seed": args.seed,
            "backbone": args.backbone,
            "source_round": args.source_round,
            "rollout_n": args.dca_rollout_n,
            "experiment_variant": args.experiment_variant,
        }
        if stage_complete(state_path, stage):
            existing = (
                json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
                if prompt_manifest_path.exists()
                else {}
            )
            if any(existing.get(key) != value for key, value in expected_prompt_signature.items()):
                if stage_complete(state_path, "update_dca"):
                    raise LineageError(
                        "completed DCA stage predates the active TMCD release revision; "
                        "start a clean artifact scope"
                    )
                mark_stage(
                    state_path,
                    stage,
                    "stale",
                    reason="prompt signature changed",
                    expected=expected_prompt_signature,
                )
        if not stage_complete(state_path, stage):
            mark_stage(state_path, stage, "in_progress")
            dataset = write_dca_prompt_dataset(
                prompt_path,
                num_rows=prompt_rows,
                seed=args.seed,
                backbone=args.backbone,
                source_round=args.source_round,
                experiment_variant=args.experiment_variant,
            )
            prompt_manifest = {
                "schema_version": 1,
                "kind": "dca_prompt_pool",
                "created_at": utc_now(),
                "tmcd_release_revision": TMCD_RELEASE_REVISION,
                **dataset,
                "sha256": sha256_file(prompt_path),
                "dca_feedback_candidates_requested": args.dca_feedback_candidates,
                "rollout_n": args.dca_rollout_n,
                "artifact_scope": args.artifact_scope,
                "source_dca_manifest": str(dca_parent_path),
                "source_dca_manifest_sha256": sha256_file(dca_parent_path),
                "source_vda_manifest": str(vda_parent_path),
                "source_vda_manifest_sha256": sha256_file(vda_parent_path),
            }
            atomic_write_json(prompt_manifest_path, prompt_manifest)
            mark_stage(state_path, stage, "completed", manifest=str(prompt_manifest_path))
    except Exception as exc:
        _mark_failed(state_path, stage, exc)
        raise

    stage = "update_dca"
    dca_target_manifest_path = _checkpoint_manifest_path(layout, "dca", target_round)
    try:
        if not stage_complete(state_path, stage):
            mark_stage(state_path, stage, "in_progress")
            if not variant.train_dca:
                feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
                feedback_log_path.write_text("", encoding="utf-8")
                feedback_manifest = {
                    "schema_version": 1,
                    "kind": "dca_feedback",
                    "tmcd_release_revision": TMCD_RELEASE_REVISION,
                    "created_at": utc_now(),
                    "backbone": args.backbone,
                    "source_round": args.source_round,
                    "target_dca_round": target_round,
                    "seed": args.seed,
                    "prompt_manifest": str(prompt_manifest_path),
                    "prompt_manifest_sha256": sha256_file(prompt_manifest_path),
                    "prompt_parquet": str(prompt_path),
                    "prompt_parquet_sha256": sha256_file(prompt_path),
                    "feedback_log": str(feedback_log_path),
                    "feedback_log_sha256": sha256_file(feedback_log_path),
                    "feedback_rows": 0,
                    "valid_vda_evaluated_rows": 0,
                    "feedback_unique_fingerprints": 0,
                    "source_dca_manifest": str(dca_parent_path),
                    "source_dca_manifest_sha256": sha256_file(dca_parent_path),
                    "source_vda_manifest": str(vda_parent_path),
                    "source_vda_manifest_sha256": sha256_file(vda_parent_path),
                    "parameter_update": False,
                    "ablation": "no_dca_training",
                }
                atomic_write_json(feedback_manifest_path, feedback_manifest)
                write_frozen_manifest(
                    dca_target_manifest_path,
                    role="dca",
                    backbone=args.backbone,
                    round_index=target_round,
                    model_path=model_path,
                    seed=args.seed,
                    parent_manifest_path=str(dca_parent_path),
                    training_data_manifest_path=str(feedback_manifest_path),
                    training_config={
                        "protocol": "dca_first_alternating",
                        "tmcd_protocol_version": "tmcd-v2",
                        "tmcd_release_revision": TMCD_RELEASE_REVISION,
                        "experiment_variant": args.experiment_variant,
                        "artifact_scope": args.artifact_scope,
                        "execution_host": execution_host,
                        "allocated_gpus": gpu_ids,
                        "world_size": len(gpu_ids),
                        "parameter_update": False,
                        "seed": args.seed,
                    },
                )
                mark_stage(
                    state_path,
                    stage,
                    "completed",
                    checkpoint_manifest=str(dca_target_manifest_path),
                    feedback_manifest=str(feedback_manifest_path),
                    parameter_update=False,
                )

        if variant.train_dca and not stage_complete(state_path, stage):
            mark_stage(state_path, stage, "in_progress")
            dca_round_steps = args.dca_steps or math.ceil(prompt_rows / args.dca_batch_size)
            dca_checkpoint_root = dca_target_dir / "trainer"
            resume_mode, resume_path, target_steps, training_done = _resume_plan(
                role="dca",
                source_round=args.source_round,
                parent_manifest=dca_parent,
                target_checkpoint_root=dca_checkpoint_root,
                data_dir=layout.data_dir,
                round_steps=dca_round_steps,
            )
            if not training_done:
                latest_dca_checkpoint = _maybe_latest(dca_checkpoint_root)
                if latest_dca_checkpoint is None and feedback_log_path.exists():
                    stale = feedback_log_path.with_name(
                        f"feedback.stale.{utc_now().replace(':', '').replace('-', '')}.jsonl"
                    )
                    feedback_log_path.replace(stale)
                elif latest_dca_checkpoint is not None:
                    dca_parent_step = (
                        _step(Path(dca_parent["checkpoint_path"]))
                        if args.source_round > 0
                        else 0
                    )
                    _reconcile_feedback_log(
                        feedback_log_path=feedback_log_path,
                        checkpoint_root=dca_checkpoint_root,
                        parent_step=dca_parent_step,
                        batch_size=args.dca_batch_size,
                        rollout_n=args.dca_rollout_n,
                    )
                environment = os.environ.copy()
                environment.update(
                    {
                        "AGZ_ROOT": str(root),
                        "AGZ_MODEL_PATH": model_path,
                        "AGZ_VDA_MODEL_PATH": model_path,
                        "AGZ_VDA_ADAPTER_PATH": str(vda_parent.get("adapter_path") or ""),
                        "AGZ_TRAIN_FILE": str(prompt_path),
                        "AGZ_VAL_FILE": str(prompt_path),
                        "AGZ_DCA_FEEDBACK_LOG": str(feedback_log_path),
                        "AGZ_RUN_NAME": (
                            f"agz_{args.artifact_scope}_{args.experiment_variant}_{args.backbone}_dca_r{target_round}"
                        ),
                        "AGZ_CHECKPOINT_DIR": str(dca_checkpoint_root),
                        "AGZ_MAX_STEPS": str(target_steps),
                        "AGZ_RESUME_MODE": resume_mode,
                        "AGZ_RESUME_FROM_PATH": resume_path,
                        "AGZ_ALLOCATED_GPU_IDS": ",".join(gpu_ids),
                        "AGZ_BACKBONE": args.backbone,
                        "AGZ_EXPERIMENT_VARIANT": args.experiment_variant,
                        "AGZ_BATCH_SIZE": str(args.dca_batch_size),
                        "AGZ_PPO_MINI_BATCH_SIZE": os.environ.get(
                            "AGZ_DCA_PPO_MINI_BATCH_SIZE",
                            str(args.dca_batch_size),
                        ),
                        "AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU": os.environ.get(
                            "AGZ_DCA_PPO_MICRO_BATCH_SIZE_PER_GPU", "4"
                        ),
                        "AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": os.environ.get(
                            "AGZ_DCA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "4"
                        ),
                        "AGZ_ROLLOUT_N": str(args.dca_rollout_n),
                        "AGZ_MAX_PROMPT_LENGTH": os.environ.get(
                            "AGZ_DCA_MAX_PROMPT_LENGTH", "2048"
                        ),
                        "AGZ_MAX_RESPONSE_LENGTH": os.environ.get(
                            "AGZ_DCA_MAX_RESPONSE_LENGTH", "1024"
                        ),
                        "AGZ_SEED": str(args.seed),
                        "AGZ_SAVE_FREQ": str(target_steps),
                        "AGZ_VDA_FEEDBACK_PORT_A": str(
                            args.feedback_port
                            or (31501 if args.backbone == "qwen3.5-4b" else 32501)
                        ),
                    }
                )
                _run(["/usr/bin/bash", str(root / "scripts" / "train_dca_qwen35_lora.sh")], env=environment)
                # Reward logging may batch fsync calls for shared-filesystem
                # throughput. Make the completed feedback pool durable before
                # reading it or hashing its lineage manifest.
                with feedback_log_path.open("rb") as handle:
                    os.fsync(handle.fileno())

            feedback_rows = read_jsonl(feedback_log_path)
            if len(feedback_rows) < args.dca_feedback_candidates:
                raise LineageError(
                    f"DCA feedback log has {len(feedback_rows)} rows, expected at least "
                    f"{args.dca_feedback_candidates}"
                )
            valid_feedback_rows = [
                row
                for row in feedback_rows
                if bool(row.get("parse_ok", False))
                and bool((row.get("vda_evaluation", {}) or {}).get("oracle_solvable", False))
                and "current_vda_safe_success" in (row.get("vda_evaluation", {}) or {})
            ]
            if not valid_feedback_rows:
                raise LineageError(
                    "DCA update produced no valid hard-but-solvable scenario evaluated by current VDA"
                )
            feedback_manifest = {
                "schema_version": 1,
                "kind": "dca_feedback",
                "tmcd_release_revision": TMCD_RELEASE_REVISION,
                "created_at": utc_now(),
                "backbone": args.backbone,
                "source_round": args.source_round,
                "target_dca_round": target_round,
                "seed": args.seed,
                "prompt_manifest": str(prompt_manifest_path),
                "prompt_manifest_sha256": sha256_file(prompt_manifest_path),
                "prompt_parquet": str(prompt_path),
                "prompt_parquet_sha256": sha256_file(prompt_path),
                "feedback_log": str(feedback_log_path),
                "feedback_log_sha256": sha256_file(feedback_log_path),
                "feedback_rows": len(feedback_rows),
                "valid_vda_evaluated_rows": len(valid_feedback_rows),
                "feedback_unique_fingerprints": len(feedback_fingerprints(feedback_log_path)),
                "source_dca_manifest": str(dca_parent_path),
                "source_dca_manifest_sha256": sha256_file(dca_parent_path),
                "source_vda_manifest": str(vda_parent_path),
                "source_vda_manifest_sha256": sha256_file(vda_parent_path),
            }
            atomic_write_json(feedback_manifest_path, feedback_manifest)
            write_trained_manifest(
                dca_target_manifest_path,
                role="dca",
                backbone=args.backbone,
                round_index=target_round,
                model_path=model_path,
                seed=args.seed,
                parent_manifest_path=str(dca_parent_path),
                training_data_manifest_path=str(feedback_manifest_path),
                checkpoint_root=str(dca_checkpoint_root),
                training_config={
                    "protocol": "dca_first_alternating",
                    "tmcd_protocol_version": "tmcd-v2",
                    "tmcd_release_revision": TMCD_RELEASE_REVISION,
                    "experiment_variant": args.experiment_variant,
                    "artifact_scope": args.artifact_scope,
                    "execution_host": execution_host,
                    "allocated_gpus": gpu_ids,
                    "world_size": len(gpu_ids),
                    "feedback_candidates": args.dca_feedback_candidates,
                    "feedback_services": len(gpu_ids),
                    "rollout_n": args.dca_rollout_n,
                    "batch_size": args.dca_batch_size,
                    "ppo_mini_batch_size": int(
                        os.environ.get(
                            "AGZ_DCA_PPO_MINI_BATCH_SIZE",
                            str(args.dca_batch_size),
                        )
                    ),
                    "round_steps": dca_round_steps,
                    "ppo_micro_batch_size_per_gpu": int(
                        os.environ.get("AGZ_DCA_PPO_MICRO_BATCH_SIZE_PER_GPU", "4")
                    ),
                    "log_prob_micro_batch_size_per_gpu": int(
                        os.environ.get("AGZ_DCA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "4")
                    ),
                    "save_freq": dca_round_steps,
                    "max_actor_ckpt_to_keep": int(
                        os.environ.get("AGZ_MAX_ACTOR_CKPT_TO_KEEP", "1")
                    ),
                    "reward_fsync_every_batches": int(
                        os.environ.get("AGZ_DCA_REWARD_FSYNC_EVERY_BATCHES", "1")
                    ),
                    "max_prompt_length": int(os.environ.get("AGZ_DCA_MAX_PROMPT_LENGTH", 2048)),
                    "max_response_length": int(os.environ.get("AGZ_DCA_MAX_RESPONSE_LENGTH", 1024)),
                    "rollout_temperature": float(os.environ.get("AGZ_ROLLOUT_TEMPERATURE", 0.7)),
                    "rollout_top_p": float(os.environ.get("AGZ_ROLLOUT_TOP_P", 1.0)),
                    "rollout_top_k": int(os.environ.get("AGZ_ROLLOUT_TOP_K", 0)),
                    "vda_feedback_max_turns": int(
                        os.environ.get("AGZ_VDA_FEEDBACK_MAX_TURNS", 5)
                    ),
                    "vda_feedback_max_input_tokens": int(
                        os.environ.get("AGZ_VDA_FEEDBACK_MAX_INPUT_TOKENS", 2048)
                    ),
                    "vda_feedback_continuation_prompt_mode": os.environ.get(
                        "AGZ_VDA_FEEDBACK_CONTINUATION_PROMPT_MODE", "legacy"
                    ),
                    "vda_feedback_history_window": int(
                        os.environ.get("AGZ_VDA_FEEDBACK_HISTORY_WINDOW", "0")
                    ),
                    "vda_feedback_invalid_action_patience": int(
                        os.environ.get("AGZ_VDA_FEEDBACK_INVALID_ACTION_PATIENCE", 0)
                    ),
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "learning_rate": 2e-5,
                    "activation_offload": False,
                    "gradient_checkpointing": (
                        os.environ.get("AGZ_ENABLE_GRADIENT_CHECKPOINTING", "true")
                        .strip()
                        .lower()
                        in {"1", "true", "yes", "on"}
                    ),
                    "seed": args.seed,
                },
            )
            mark_stage(
                state_path,
                stage,
                "completed",
                checkpoint_manifest=str(dca_target_manifest_path),
                feedback_manifest=str(feedback_manifest_path),
            )
        if not stage_complete(state_path, stage):
            raise LineageError(f"DCA update stage did not complete: {state_path}")
        dca_target = load_checkpoint_manifest(
            dca_target_manifest_path,
            role="dca",
            backbone=args.backbone,
            round_index=target_round,
        )
        if str(
            (dca_target.get("training_config", {}) or {}).get(
                "tmcd_release_revision", ""
            )
        ) != TMCD_RELEASE_REVISION:
            raise LineageError("DCA checkpoint release revision mismatch")
    except Exception as exc:
        _mark_failed(state_path, stage, exc)
        raise

    stage = "generate_fresh_vda_candidates"
    candidate_max_attempts = int(
        os.environ.get("AGZ_DCA_CANDIDATE_MAX_ATTEMPTS", "3")
    )
    try:
        if candidate_max_attempts <= 0:
            raise LineageError("candidate generation retry budget must be positive")
        if stage_complete(state_path, stage):
            existing_pool = (
                json.loads(candidate_path.read_text(encoding="utf-8"))
                if candidate_path.exists()
                else {}
            )
            expected_signature = {
                "tmcd_release_revision": TMCD_RELEASE_REVISION,
                "num_candidates_requested": args.vda_candidates,
                "experiment_variant": args.experiment_variant,
                "generation_prompt_version": DCA_PROMPT_VERSION,
                "candidate_normalization_version": DCA_CANDIDATE_NORMALIZATION_VERSION,
                "max_attempts": candidate_max_attempts,
                "source_dca_round": target_round,
                "source_dca_checkpoint_manifest_sha256": sha256_file(
                    dca_target_manifest_path
                ),
            }
            if any(
                existing_pool.get(key) != value
                for key, value in expected_signature.items()
            ):
                if stage_complete(state_path, "update_vda"):
                    raise LineageError(
                        "candidate generation signature changed after VDA training completed; "
                        "use a new artifact scope instead of mutating a completed round"
                    )
                existing_vda_steps = list(
                    (vda_target_dir / "trainer").glob("global_step_*")
                )
                if existing_vda_steps:
                    raise LineageError(
                        "candidate generation signature changed after VDA recovery checkpoints "
                        f"were created: {existing_vda_steps}"
                    )
                mark_stage(
                    state_path,
                    stage,
                    "stale",
                    reason="candidate generation signature changed",
                    expected=expected_signature,
                )
                mark_stage(
                    state_path,
                    "build_isolated_vda_pool",
                    "stale",
                    reason="upstream candidate generation signature changed",
                )
        if not stage_complete(state_path, stage):
            mark_stage(state_path, stage, "in_progress")
            jobs = []
            for shard_index, (gpu_id, shard_path) in enumerate(
                zip(gpu_ids, candidate_shards)
            ):
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
                triton_cache = Path(
                    os.environ.get("AGZ_TRITON_CACHE_ROOT", "/tmp/agentguard_zero_triton")
                ) / "dca_candidates" / args.backbone / f"shard_{shard_index}"
                triton_cache.mkdir(parents=True, exist_ok=True)
                environment["TRITON_CACHE_DIR"] = str(triton_cache)
                jobs.append(
                    (
                        [
                            sys.executable,
                            str(root / "scripts" / "generate_dca_scenarios.py"),
                            "--checkpoint-manifest",
                            str(dca_target_manifest_path),
                            "--output",
                            str(shard_path),
                            "--num-candidates",
                            str(args.vda_candidates),
                            "--batch-size",
                            str(args.candidate_batch_size),
                            "--num-shards",
                            str(len(gpu_ids)),
                            "--shard-index",
                            str(shard_index),
                            "--seed",
                            str(args.seed + target_round * 1000),
                            "--max-input-tokens",
                            os.environ.get("AGZ_DCA_MAX_PROMPT_LENGTH", "2048"),
                            "--max-new-tokens",
                            os.environ.get("AGZ_DCA_MAX_RESPONSE_LENGTH", "1024"),
                            "--attn-implementation",
                            os.environ.get(
                                "AGZ_DCA_CANDIDATE_ATTN_IMPLEMENTATION", "sdpa"
                            ),
                            "--partial-fsync-every-batches",
                            os.environ.get(
                                "AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES", "16"
                            ),
                            "--max-attempts",
                            str(candidate_max_attempts),
                            "--temperature",
                            os.environ.get("AGZ_ROLLOUT_TEMPERATURE", "0.7"),
                            "--top-p",
                            os.environ.get("AGZ_ROLLOUT_TOP_P", "1.0"),
                            "--top-k",
                            os.environ.get("AGZ_ROLLOUT_TOP_K", "0"),
                            "--experiment-variant",
                            args.experiment_variant,
                        ],
                        environment,
                    )
                )
            _run_parallel(jobs)
            _run(
                [
                    sys.executable,
                    str(root / "scripts" / "merge_dca_candidate_shards.py"),
                    "--shards",
                    *[str(path) for path in candidate_shards],
                    "--expected-count",
                    str(args.vda_candidates),
                    "--output",
                    str(candidate_path),
                ]
            )
            mark_stage(
                state_path,
                stage,
                "completed",
                candidate_pool=str(candidate_path),
                candidate_pool_sha256=sha256_file(candidate_path),
                candidate_shards={
                    str(index): {"path": str(path), "sha256": sha256_file(path)}
                    for index, path in enumerate(candidate_shards)
                },
            )
    except Exception as exc:
        _mark_failed(state_path, stage, exc)
        raise

    stage = "build_isolated_vda_pool"
    try:
        if stage_complete(state_path, stage):
            try:
                validate_round_lineage(
                    dca_manifest_path=str(dca_target_manifest_path),
                    feedback_log_path=str(feedback_log_path),
                    pool_manifest_path=str(pool_manifest_path),
                    split_paths=[str(path) for path in split_paths.values()],
                    backbone=args.backbone,
                    target_round=target_round,
                )
            except (LineageError, FileNotFoundError, json.JSONDecodeError) as exc:
                if stage_complete(state_path, "update_vda"):
                    raise LineageError(
                        "isolated VDA pool lineage changed after VDA training completed"
                    ) from exc
                mark_stage(
                    state_path,
                    stage,
                    "stale",
                    reason="stored VDA pool lineage no longer validates",
                    error=str(exc),
                )
        if not stage_complete(state_path, stage):
            mark_stage(state_path, stage, "in_progress")
            _run(
                [
                    sys.executable,
                    str(root / "scripts" / "build_vda_round_pool.py"),
                    "--candidate-pool",
                    str(candidate_path),
                    "--feedback-log",
                    str(feedback_log_path),
                    "--dca-checkpoint-manifest",
                    str(dca_target_manifest_path),
                    "--output-dir",
                    str(layout.data_dir),
                    "--train-size",
                    str(args.vda_train_size),
                    "--dev-size",
                    str(args.vda_dev_size),
                    "--xplay-size",
                    str(args.vda_xplay_size),
                    "--seed",
                    str(args.seed),
                    "--train-batch-size",
                    str(args.vda_batch_size),
                ]
            )
            lineage = validate_round_lineage(
                dca_manifest_path=str(dca_target_manifest_path),
                feedback_log_path=str(feedback_log_path),
                pool_manifest_path=str(pool_manifest_path),
                split_paths=[str(path) for path in split_paths.values()],
                backbone=args.backbone,
                target_round=target_round,
            )
            mark_stage(state_path, stage, "completed", lineage=lineage)
    except Exception as exc:
        _mark_failed(state_path, stage, exc)
        raise

    _run(
        [
            sys.executable,
            str(root / "scripts" / "audit_tmcd_v2_release.py"),
            "--pool-manifest",
            str(pool_manifest_path),
            "--train-size",
            str(args.vda_train_size),
            "--dev-size",
            str(args.vda_dev_size),
            "--xplay-size",
            str(args.vda_xplay_size),
        ]
    )

    if args.stop_after_stage == stage:
        lineage = validate_round_lineage(
            dca_manifest_path=str(dca_target_manifest_path),
            feedback_log_path=str(feedback_log_path),
            pool_manifest_path=str(pool_manifest_path),
            split_paths=[str(path) for path in split_paths.values()],
            backbone=args.backbone,
            target_round=target_round,
        )
        print(
            json.dumps(
                {
                    "status": "stopped_after_requested_stage",
                    "stage": stage,
                    "backbone": args.backbone,
                    "source_round": args.source_round,
                    "target_round": target_round,
                    "lineage": lineage,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    stage = "update_vda"
    vda_target_manifest_path = _checkpoint_manifest_path(layout, "vda", target_round)
    try:
        if not stage_complete(state_path, stage):
            mark_stage(state_path, stage, "in_progress")
            vda_round_steps = args.vda_steps or math.ceil(args.vda_train_size / args.vda_batch_size)
            vda_checkpoint_root = vda_target_dir / "trainer"
            resume_mode, resume_path, target_steps, training_done = _resume_plan(
                role="vda",
                source_round=args.source_round,
                parent_manifest=vda_parent,
                target_checkpoint_root=vda_checkpoint_root,
                data_dir=layout.data_dir,
                round_steps=vda_round_steps,
            )
            vda_action_tokens = int(os.environ.get("AGZ_VDA_ACTION_TOKENS", "384"))
            vda_observation_tokens = int(
                os.environ.get("AGZ_VDA_OBSERVATION_TOKENS", "512")
            )
            computed_trajectory_tokens = args.vda_max_turns * (
                vda_action_tokens + vda_observation_tokens
            )
            vda_trajectory_tokens = int(
                os.environ.get(
                    "AGZ_VDA_TRAJECTORY_TOKENS", str(computed_trajectory_tokens)
                )
            )
            vda_model_tokens = int(
                os.environ.get(
                    "AGZ_VDA_MODEL_TOKENS",
                    str(2048 + vda_trajectory_tokens + vda_action_tokens),
                )
            )
            if not training_done:
                environment = os.environ.copy()
                environment.update(
                    {
                        "AGZ_ROOT": str(root),
                        "AGZ_MODEL_PATH": model_path,
                        "AGZ_TRAIN_FILE": str(split_paths["train"]),
                        "AGZ_VAL_FILE": str(split_paths["dev"] if args.vda_dev_size else split_paths["train"]),
                        "AGZ_RUN_NAME": (
                            f"agz_{args.artifact_scope}_{args.experiment_variant}_{args.backbone}_vda_r{target_round}"
                        ),
                        "AGZ_CHECKPOINT_DIR": str(vda_checkpoint_root),
                        "AGZ_MAX_STEPS": str(target_steps),
                        "AGZ_RESUME_MODE": resume_mode,
                        "AGZ_RESUME_FROM_PATH": resume_path,
                        "AGZ_CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
                        "AGZ_BACKBONE": args.backbone,
                        "AGZ_EXPERIMENT_VARIANT": args.experiment_variant,
                        "AGZ_N_GPUS_PER_NODE": str(len(gpu_ids)),
                        "AGZ_BATCH_SIZE": str(args.vda_batch_size),
                        "AGZ_PPO_MINI_BATCH_SIZE": os.environ.get(
                            "AGZ_VDA_PPO_MINI_BATCH_SIZE",
                            str(args.vda_batch_size),
                        ),
                        "AGZ_ROLLOUT_N": str(args.vda_rollout_n),
                        "AGZ_ADV_ESTIMATOR": "reinforce_plus_plus",
                        "AGZ_TOOL_SERVER_MODE": "level1",
                        "AGZ_BUILD_SMOKE_DATASET": "0",
                        "AGZ_AGENT_MAX_TURNS": str(args.vda_max_turns),
                        "AGZ_MAX_PROMPT_LENGTH": "2048",
                        "AGZ_MAX_RESPONSE_LENGTH": str(vda_trajectory_tokens),
                        "AGZ_MAX_MODEL_LENGTH": str(vda_model_tokens),
                        "AGZ_MAX_ACTION_LENGTH": str(vda_action_tokens),
                        "AGZ_MAX_OBS_LENGTH": str(vda_observation_tokens),
                        "AGZ_GPU_MEMORY_UTILIZATION": os.environ.get(
                            "AGZ_GPU_MEMORY_UTILIZATION", "0.35"
                        ),
                        "AGZ_MAX_NUM_SEQS": (
                            os.environ.get(
                                "AGZ_MAX_NUM_SEQS",
                                os.environ.get(
                                    "AGZ_VDA_MAX_NUM_SEQS",
                                    "8" if args.backbone == "qwen3.5-4b" else "4",
                                ),
                            )
                        ),
                        "AGZ_AGENT_NUM_WORKERS": os.environ.get(
                            "AGZ_AGENT_NUM_WORKERS", "1"
                        ),
                        "AGZ_ROLLOUT_BACKEND": os.environ.get(
                            "AGZ_ROLLOUT_BACKEND", "hf"
                        ),
                        "AGZ_LORA_RANK": "16",
                        "AGZ_LORA_ALPHA": "32",
                        "AGZ_LORA_TARGET_MODULES": "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]",
                        "AGZ_ACTOR_LR": "2e-5",
                        "AGZ_ACTOR_CPU_OFFLOAD": "false",
                        "AGZ_ACTOR_PARAM_OFFLOAD": "false",
                        "AGZ_ACTOR_OPTIMIZER_OFFLOAD": "false",
                        "AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU": (
                            os.environ.get(
                                "AGZ_VDA_PPO_MICRO_BATCH_SIZE_PER_GPU", "2"
                            )
                        ),
                        "AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": (
                            os.environ.get(
                                "AGZ_VDA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "2"
                            )
                        ),
                        "AGZ_REF_PARAM_OFFLOAD": (
                            "true" if args.backbone == "qwen3.5-9b" else "false"
                        ),
                        "AGZ_SEED": str(args.seed),
                        "AGZ_VAL_BEFORE_TRAIN": "False",
                        "AGZ_DATA_SHUFFLE": "false",
                        "AGZ_SAVE_FREQ": os.environ.get(
                            "AGZ_VDA_SAVE_FREQ", str(target_steps)
                        ),
                        "AGZ_TEST_FREQ": "0",
                    }
                )
                _run(
                    ["/usr/bin/bash", str(root / "scripts" / "train_vda_qwen35_lora.sh")],
                    env=environment,
                )
            write_trained_manifest(
                vda_target_manifest_path,
                role="vda",
                backbone=args.backbone,
                round_index=target_round,
                model_path=model_path,
                seed=args.seed,
                parent_manifest_path=str(vda_parent_path),
                training_data_manifest_path=str(pool_manifest_path),
                checkpoint_root=str(vda_checkpoint_root),
                training_config={
                    "protocol": "dca_first_alternating",
                    "tmcd_protocol_version": "tmcd-v2",
                    "tmcd_release_revision": TMCD_RELEASE_REVISION,
                    "experiment_variant": args.experiment_variant,
                    "artifact_scope": args.artifact_scope,
                    "execution_host": execution_host,
                    "allocated_gpus": gpu_ids,
                    "world_size": len(gpu_ids),
                    "candidate_pool_size": args.vda_candidates,
                    "candidate_generation_batch_size": args.candidate_batch_size,
                    "candidate_generation_prompt_version": DCA_PROMPT_VERSION,
                    "candidate_normalization_version": DCA_CANDIDATE_NORMALIZATION_VERSION,
                    "candidate_generation_max_attempts": candidate_max_attempts,
                    "candidate_attention_implementation": os.environ.get(
                        "AGZ_DCA_CANDIDATE_ATTN_IMPLEMENTATION", "sdpa"
                    ),
                    "candidate_partial_fsync_every_batches": int(
                        os.environ.get(
                            "AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES", "16"
                        )
                    ),
                    "train_size": args.vda_train_size,
                    "dev_size": args.vda_dev_size,
                    "xplay_size": args.vda_xplay_size,
                    "rollout_n": args.vda_rollout_n,
                    "advantage_estimator": "reinforce_plus_plus",
                    "batch_size": args.vda_batch_size,
                    "round_steps": vda_round_steps,
                    "max_turns": args.vda_max_turns,
                    "max_prompt_length": 2048,
                    "max_trajectory_response_length": vda_trajectory_tokens,
                    "max_model_length": vda_model_tokens,
                    "max_action_length": vda_action_tokens,
                    "max_observation_length": vda_observation_tokens,
                    "ppo_mini_batch_size": int(
                        os.environ.get(
                            "AGZ_VDA_PPO_MINI_BATCH_SIZE",
                            str(args.vda_batch_size),
                        )
                    ),
                    "ppo_micro_batch_size_per_gpu": int(
                        os.environ.get(
                            "AGZ_VDA_PPO_MICRO_BATCH_SIZE_PER_GPU", "2"
                        )
                    ),
                    "log_prob_micro_batch_size_per_gpu": int(
                        os.environ.get(
                            "AGZ_VDA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "2"
                        )
                    ),
                    "rollout_backend": os.environ.get("AGZ_ROLLOUT_BACKEND", "hf"),
                    "agent_num_workers": int(
                        os.environ.get("AGZ_AGENT_NUM_WORKERS", "1")
                    ),
                    "gpu_memory_utilization": float(
                        os.environ.get("AGZ_GPU_MEMORY_UTILIZATION", "0.35")
                    ),
                    "max_num_seqs": int(
                        os.environ.get(
                            "AGZ_MAX_NUM_SEQS",
                            os.environ.get(
                                "AGZ_VDA_MAX_NUM_SEQS",
                                "8" if args.backbone == "qwen3.5-4b" else "4",
                            ),
                        )
                    ),
                    "save_freq": int(
                        os.environ.get("AGZ_VDA_SAVE_FREQ", str(vda_round_steps))
                    ),
                    "max_actor_ckpt_to_keep": int(
                        os.environ.get("AGZ_MAX_ACTOR_CKPT_TO_KEEP", "1")
                    ),
                    "gradient_checkpointing": (
                        os.environ.get("AGZ_ENABLE_GRADIENT_CHECKPOINTING", "true")
                        .strip()
                        .lower()
                        in {"1", "true", "yes", "on"}
                    ),
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "learning_rate": 2e-5,
                    "seed": args.seed,
                },
            )
            vda_target = load_checkpoint_manifest(
                vda_target_manifest_path,
                role="vda",
                backbone=args.backbone,
                round_index=target_round,
            )
            if str(
                (vda_target.get("training_config", {}) or {}).get(
                    "tmcd_release_revision", ""
                )
            ) != TMCD_RELEASE_REVISION:
                raise LineageError("VDA checkpoint release revision mismatch")
            if dca_target.get("adapter_sha256") == vda_target.get("adapter_sha256"):
                raise LineageError("DCA and VDA adapters have the same hash")
            lineage = validate_round_lineage(
                dca_manifest_path=str(dca_target_manifest_path),
                feedback_log_path=str(feedback_log_path),
                pool_manifest_path=str(pool_manifest_path),
                split_paths=[str(path) for path in split_paths.values()],
                backbone=args.backbone,
                target_round=target_round,
            )
            mark_stage(
                state_path,
                stage,
                "completed",
                checkpoint_manifest=str(vda_target_manifest_path),
                lineage=lineage,
            )
        vda_target = load_checkpoint_manifest(
            vda_target_manifest_path,
            role="vda",
            backbone=args.backbone,
            round_index=target_round,
        )
        if str(
            (vda_target.get("training_config", {}) or {}).get(
                "tmcd_release_revision", ""
            )
        ) != TMCD_RELEASE_REVISION:
            raise LineageError("VDA checkpoint release revision mismatch")
    except Exception as exc:
        _mark_failed(state_path, stage, exc)
        raise

    stage = "validate_adapter_reloads"
    reload_reports = {
        "dca": layout.data_dir / "adapter_reload_dca.json",
        "vda": layout.data_dir / "adapter_reload_vda.json",
    }
    try:
        if not stage_complete(state_path, stage):
            mark_stage(state_path, stage, "in_progress")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
            for role, manifest_path in (
                ("dca", dca_target_manifest_path),
                ("vda", vda_target_manifest_path),
            ):
                _run(
                    [
                        sys.executable,
                        str(root / "scripts" / "validate_adapter_reload.py"),
                        "--checkpoint-manifest",
                        str(manifest_path),
                        "--output",
                        str(reload_reports[role]),
                    ],
                    env=environment,
                )
            mark_stage(
                state_path,
                stage,
                "completed",
                reports={role: str(path) for role, path in reload_reports.items()},
            )
    except Exception as exc:
        _mark_failed(state_path, stage, exc)
        raise

    stage = "prune_parent_recovery_checkpoints"
    try:
        if not stage_complete(state_path, stage):
            mark_stage(state_path, stage, "in_progress")
            reports: dict[str, Any] = {}
            if args.artifact_scope in {"formal", "tmcd_v2"} and args.source_round > 0:
                reports = {
                    "dca": _prune_parent_recovery_checkpoint(dca_parent, dca_parent_path),
                    "vda": _prune_parent_recovery_checkpoint(vda_parent, vda_parent_path),
                }
            mark_stage(state_path, stage, "completed", reports=reports)
    except Exception as exc:
        _mark_failed(state_path, stage, exc)
        raise

    report = {
        "schema_version": 1,
        "kind": "dca_first_round_report",
        "completed_at": utc_now(),
        "backbone": args.backbone,
        "artifact_scope": args.artifact_scope,
        "experiment_variant": args.experiment_variant,
        "source_round": args.source_round,
        "target_round": target_round,
        "seed": args.seed,
        "execution_host": execution_host,
        "allocated_gpus": gpu_ids,
        "protocol": (
            [
                f"DCA_{args.source_round} generates feedback candidates",
                f"VDA_{args.source_round} produces rollout feedback",
                f"DCA_{args.source_round} updates to DCA_{target_round}",
                f"DCA_{target_round} generates a fresh disjoint pool",
                f"VDA_{args.source_round} updates to VDA_{target_round}",
            ]
            if variant.train_dca
            else [
                f"DCA_{args.source_round} remains frozen as DCA_{target_round}",
                f"frozen DCA_{target_round} generates a fresh pool",
                f"VDA_{args.source_round} updates to VDA_{target_round}",
            ]
        ),
        "dca_parameter_update": bool(variant.train_dca),
        "dca_parent_manifest": str(dca_parent_path),
        "dca_target_manifest": str(dca_target_manifest_path),
        "vda_parent_manifest": str(vda_parent_path),
        "vda_target_manifest": str(vda_target_manifest_path),
        "dca_adapter_sha256": dca_target.get("adapter_sha256"),
        "vda_adapter_sha256": vda_target.get("adapter_sha256"),
        "adapter_reload_reports": {role: str(path) for role, path in reload_reports.items()},
        "feedback_manifest": str(feedback_manifest_path),
        "vda_pool_manifest": str(pool_manifest_path),
        "state": str(state_path),
    }
    report_path = layout.data_dir / "round_report.json"
    atomic_write_json(report_path, report)
    mark_stage(state_path, "round_complete", "completed", report=str(report_path))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
