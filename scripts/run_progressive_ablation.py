#!/usr/bin/env python3
"""Train one progressive VDA-only control across sealed formal R1-R3 data."""

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

from agentguard_zero.protocol import TMCD_RELEASE_REVISION
from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    latest_global_step,
    load_checkpoint_manifest,
    sha256_file,
    sha256_tree,
    utc_now,
    write_base_manifest,
    write_trained_manifest,
)
from agentguard_zero.variants import PROGRESSIVE_VARIANTS, experiment_variant


def _step(path: Path) -> int:
    return int(path.name.split("global_step_", 1)[1])


def _maybe_latest(path: Path) -> Path | None:
    try:
        return latest_global_step(path)
    except LineageError:
        return None


def _resume_link(data_dir: Path, parent_checkpoint: Path) -> Path:
    target = data_dir / "_resume" / "vda" / parent_checkpoint.name
    target.mkdir(parents=True, exist_ok=True)
    for name in ("actor", "critic"):
        source = parent_checkpoint / name
        link = target / name
        if not source.exists():
            continue
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise LineageError(f"resume link already exists with another target: {link}")
        link.symlink_to(source.resolve(), target_is_directory=True)
    if not (target / "actor").exists():
        raise LineageError(f"parent actor checkpoint is missing: {parent_checkpoint}")
    return target


def _dca_identity(root: Path, scope: str, backbone: str) -> dict[str, Any]:
    values = {}
    dca_root = root / "checkpoints" / scope / backbone / "dca"
    for round_index in (0, 1, 2, 3):
        manifest = dca_root / f"round_{round_index}" / "manifest.json"
        if not manifest.exists():
            raise LineageError(f"formal DCA manifest is missing: {manifest}")
        item: dict[str, Any] = {"manifest_sha256": sha256_file(manifest)}
        value = json.loads(manifest.read_text(encoding="utf-8"))
        adapter = value.get("adapter_path")
        if adapter:
            adapter_path = Path(adapter).resolve()
            item["adapter_path"] = str(adapter_path)
            item["adapter_sha256"] = sha256_tree(adapter_path)
        values[str(round_index)] = item
    return values


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--formal-scope", default="tmcd_v242")
    parser.add_argument("--backbone", choices=["qwen3.5-4b"], required=True)
    parser.add_argument("--variant", choices=PROGRESSIVE_VARIANTS, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--allocated-gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    model_path = str(Path(args.model_path).resolve())
    gpu_ids = [value.strip() for value in args.allocated_gpus.split(",") if value.strip()]
    if len(gpu_ids) != 4:
        raise SystemExit(f"progressive training requires four GPUs, got {gpu_ids}")
    variant = experiment_variant(args.variant)
    if variant.train_dca or variant.frontier_filtering:
        raise SystemExit(f"progressive variant must be VDA-only and static: {args.variant}")

    data_root = (
        root
        / "data"
        / args.formal_scope
        / "ablations"
        / "progressive"
        / args.variant
        / args.backbone
    )
    checkpoint_root = (
        root
        / "checkpoints"
        / args.formal_scope
        / "ablations"
        / "progressive"
        / args.variant
        / args.backbone
        / "vda"
    )
    output_root = root / "outputs" / args.formal_scope / "progressive" / args.variant
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    dca_before = _dca_identity(root, args.formal_scope, args.backbone)
    atomic_write_json(output_root / "formal_dca_identity_before.json", dca_before)

    base_manifest_path = checkpoint_root / "round_0" / "manifest.json"
    base_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_base_manifest(
        base_manifest_path,
        role="vda",
        backbone=args.backbone,
        model_path=model_path,
        seed=args.seed,
    )

    round_reports = []
    parent_manifest_path = base_manifest_path
    for round_index in (1, 2, 3):
        round_data = data_root / f"round_{round_index}"
        _run(
            [
                sys.executable,
                str(root / "scripts" / "prepare_progressive_ablation_data.py"),
                "--root",
                str(root),
                "--formal-scope",
                args.formal_scope,
                "--backbone",
                args.backbone,
                "--variant",
                args.variant,
                "--round",
                str(round_index),
                "--output-dir",
                str(round_data),
            ]
        )
        data_manifest_path = round_data / "manifest.json"
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        train_path = round_data / "vda_train" / "train.parquet"
        dev_path = round_data / "vda_dev" / "dev.parquet"

        target_dir = checkpoint_root / f"round_{round_index}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_manifest_path = target_dir / "manifest.json"
        target_trainer = target_dir / "trainer"
        required_step = round_index * 25
        run_name = f"agz_{args.formal_scope}_progressive_{args.variant}_{args.backbone}_vda_r{round_index}"
        log_path = root / "logs" / f"{run_name}.log"

        if target_manifest_path.exists():
            trained = load_checkpoint_manifest(
                target_manifest_path,
                role="vda",
                backbone=args.backbone,
                round_index=round_index,
            )
            if trained.get("training_data_manifest_sha256") != sha256_file(data_manifest_path):
                raise LineageError(f"round {round_index} training data lineage changed")
            if trained.get("adapter_sha256") != sha256_tree(trained["adapter_path"]):
                raise LineageError(f"round {round_index} adapter hash changed")
            if _step(Path(trained["checkpoint_path"])) != required_step:
                raise LineageError(f"round {round_index} checkpoint step mismatch")
            round_reports.append(
                {
                    "round": round_index,
                    "status": "already_trained",
                    "manifest": str(target_manifest_path),
                    "manifest_sha256": sha256_file(target_manifest_path),
                }
            )
            parent_manifest_path = target_manifest_path
            continue

        latest = _maybe_latest(target_trainer)
        if latest is not None:
            if _step(latest) > required_step:
                raise LineageError(f"round {round_index} checkpoint exceeds step {required_step}")
            resume_mode = "auto"
            resume_path = "null"
            training_complete = _step(latest) == required_step
        elif round_index == 1:
            resume_mode = "disable"
            resume_path = "null"
            training_complete = False
        else:
            parent = load_checkpoint_manifest(
                parent_manifest_path,
                role="vda",
                backbone=args.backbone,
                round_index=round_index - 1,
            )
            parent_checkpoint = Path(parent["checkpoint_path"])
            if _step(parent_checkpoint) != required_step - 25:
                raise LineageError(f"round {round_index} parent checkpoint step mismatch")
            resume_mode = "resume_path"
            resume_path = str(_resume_link(round_data, parent_checkpoint))
            training_complete = False

        if not training_complete:
            environment = os.environ.copy()
            environment.update(
                {
                    "AGZ_ROOT": str(root),
                    "AGZ_MODEL_PATH": model_path,
                    "AGZ_TRAIN_FILE": str(train_path),
                    "AGZ_VAL_FILE": str(dev_path),
                    "AGZ_RUN_NAME": run_name,
                    "AGZ_CHECKPOINT_DIR": str(target_trainer),
                    "AGZ_MAX_STEPS": str(required_step),
                    "AGZ_RESUME_MODE": resume_mode,
                    "AGZ_RESUME_FROM_PATH": resume_path,
                    "AGZ_CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
                    "AGZ_BACKBONE": args.backbone,
                    "AGZ_EXPERIMENT_VARIANT": args.variant,
                    "AGZ_N_GPUS_PER_NODE": "4",
                    "AGZ_BATCH_SIZE": "32",
                    "AGZ_GEN_BATCH_SIZE": "96",
                    "AGZ_ROLLOUT_N": "1",
                    "AGZ_ROLLOUT_TEMPERATURE": "0.7",
                    "AGZ_ROLLOUT_TOP_P": "1.0",
                    "AGZ_ROLLOUT_TOP_K": "0",
                    "AGZ_PPO_MINI_BATCH_SIZE": "32",
                    "AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
                    "AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": "2",
                    "AGZ_MAX_PROMPT_LENGTH": "2048",
                    "AGZ_MAX_RESPONSE_LENGTH": "11264",
                    "AGZ_MAX_ACTION_LENGTH": "320",
                    "AGZ_MAX_OBS_LENGTH": "1280",
                    "AGZ_AGENT_MAX_TURNS": "16",
                    "AGZ_MAX_MODEL_LENGTH": "15360",
                    "AGZ_GPU_MEMORY_UTILIZATION": "0.50",
                    "AGZ_MAX_NUM_SEQS": "32",
                    "AGZ_AGENT_NUM_WORKERS": "4",
                    "AGZ_ROLLOUT_SERVER_MAX_PARALLEL_TRAJECTORIES": "24",
                    "AGZ_ROLLOUT_SERVER_MAX_STATES": "512",
                    "AGZ_ROLLOUT_BACKEND": "hf",
                    "AGZ_LORA_RANK": "16",
                    "AGZ_LORA_ALPHA": "32",
                    "AGZ_LORA_TARGET_MODULES": "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]",
                    "AGZ_ACTOR_LR": "2e-5",
                    "AGZ_ACTOR_CPU_OFFLOAD": "false",
                    "AGZ_ACTOR_PARAM_OFFLOAD": "false",
                    "AGZ_ACTOR_OPTIMIZER_OFFLOAD": "false",
                    "AGZ_REF_PARAM_OFFLOAD": "false",
                    "AGZ_RESHARD_AFTER_FORWARD": "true",
                    "AGZ_ENABLE_GRADIENT_CHECKPOINTING": "true",
                    "AGZ_REQUIRE_TRAJECTORY_REWARD": "1",
                    "AGZ_TOOL_SERVER_MODE": "level1",
                    "AGZ_BUILD_SMOKE_DATASET": "0",
                    "AGZ_SEED": str(args.seed),
                    "AGZ_VAL_BEFORE_TRAIN": "False",
                    "AGZ_DATA_SHUFFLE": "false",
                    "AGZ_SAVE_FREQ": "25",
                    "AGZ_MAX_ACTOR_CKPT_TO_KEEP": "1",
                    "AGZ_TEST_FREQ": "0",
                    "AGZ_PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            )
            _run(
                ["/usr/bin/bash", str(root / "scripts" / "train_vda_qwen35_lora.sh")],
                env=environment,
            )

        latest = latest_global_step(target_trainer)
        if _step(latest) != required_step:
            raise LineageError(
                f"round {round_index} ended at {_step(latest)}, expected {required_step}"
            )
        validation_path = output_root / f"round_{round_index}_training_validation.json"
        _run(
            [
                sys.executable,
                str(root / "scripts" / "validate_vda_training_log.py"),
                "--log",
                str(log_path),
                "--output",
                str(validation_path),
                "--expected-step",
                str(required_step),
                "--action-budget",
                "320",
                "--observation-budget",
                "1280",
            ]
        )
        manifest = write_trained_manifest(
            target_manifest_path,
            role="vda",
            backbone=args.backbone,
            round_index=round_index,
            model_path=model_path,
            seed=args.seed,
            parent_manifest_path=str(parent_manifest_path),
            training_data_manifest_path=str(data_manifest_path),
            checkpoint_root=str(target_trainer),
            training_config={
                "protocol": "progressive_static_vda_only_three_rounds",
                "tmcd_release_revision": TMCD_RELEASE_REVISION,
                "formal_scope": args.formal_scope,
                "experiment_variant": args.variant,
                "parameter_update": True,
                "dca_generation": False,
                "dca_update": False,
                "frontier_filtering": False,
                "source_formal_round": round_index,
                "source_formal_pool_manifest_sha256": data_manifest[
                    "source_pool_manifest_sha256"
                ],
                "train_size": 2400,
                "dev_size": 400,
                "batch_size": 32,
                "generation_batch_size": 96,
                "global_steps_this_round": 25,
                "global_steps_total": required_step,
                "ppo_updates_this_round": 75,
                "ppo_updates_total": required_step * 3,
                "rollout_n": 1,
                "max_turns": 16,
                "max_prompt_length": 2048,
                "max_trajectory_response_length": 11264,
                "max_model_length": 15360,
                "max_action_length": 320,
                "max_observation_length": 1280,
                "ppo_mini_batch_size": 32,
                "ppo_micro_batch_size_per_gpu": 1,
                "log_prob_micro_batch_size_per_gpu": 2,
                "lora_rank": 16,
                "lora_alpha": 32,
                "learning_rate": 2e-5,
                "seed": args.seed,
                "allocated_gpus": gpu_ids,
                "execution_host": os.uname().nodename,
            },
        )
        round_reports.append(
            {
                "round": round_index,
                "status": "trained",
                "data_manifest": str(data_manifest_path),
                "data_manifest_sha256": sha256_file(data_manifest_path),
                "checkpoint_manifest": str(target_manifest_path),
                "checkpoint_manifest_sha256": sha256_file(target_manifest_path),
                "adapter_sha256": manifest["adapter_sha256"],
                "training_validation": str(validation_path),
                "training_validation_sha256": sha256_file(validation_path),
            }
        )
        parent_manifest_path = target_manifest_path

    dca_after = _dca_identity(root, args.formal_scope, args.backbone)
    atomic_write_json(output_root / "formal_dca_identity_after.json", dca_after)
    if dca_before != dca_after:
        raise LineageError("formal DCA manifests or adapters changed during VDA-only ablation")
    final_manifest = checkpoint_root / "round_3" / "manifest.json"
    report = {
        "schema_version": 1,
        "kind": "progressive_ablation_training_report",
        "status": "complete",
        "completed_at": utc_now(),
        "formal_scope": args.formal_scope,
        "backbone": args.backbone,
        "variant": args.variant,
        "dca_generation": False,
        "dca_update": False,
        "formal_dca_unchanged": True,
        "rounds": round_reports,
        "total_train_scenarios": 7200,
        "total_global_steps": 75,
        "total_ppo_updates": 225,
        "final_checkpoint_manifest": str(final_manifest),
        "final_checkpoint_manifest_sha256": sha256_file(final_manifest),
    }
    atomic_write_json(output_root / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
