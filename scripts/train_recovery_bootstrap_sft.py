#!/usr/bin/env python3
"""Train one isolated Gate-A bootstrap LoRA arm.

Launch with torchrun for multi-GPU training.  The script refuses unaccepted
bootstrap data and writes an isolated adapter plus a recovery-lineage manifest.
It never launches another recovery stage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION, RecoveryConfig
from agentguard_zero.recovery.action_serialization import action_first_wire_json
from agentguard_zero.recovery.action_intent import (
    INTENT_FORMAT,
    action_intent_wire_json,
    compact_intent_prompt,
)
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    model_identity,
    sha256_file,
    sha256_tree,
    utc_now,
)


def _load_accepted_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("accepted") is not True:
        raise RuntimeError("bootstrap manifest is not accepted")
    if payload.get("protocol_version") != RECOVERY_PROTOCOL_VERSION:
        raise RuntimeError("bootstrap manifest has the wrong recovery protocol")
    if float(payload.get("unique_prompt_target_ratio", 0.0)) < 0.95:
        raise RuntimeError("bootstrap manifest failed the uniqueness gate")
    rank_gate = payload.get("teacher_core_rank_correlation_gate", {}) or {}
    if rank_gate.get("accepted") is not True:
        raise RuntimeError("bootstrap manifest failed teacher/core alignment")
    return payload


def _load_review_approval(path: Path, approved_stage: str) -> dict[str, Any]:
    """Require a separate, hashed human-review artifact before any GPU update."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("review approval must be a JSON object")
    if payload.get("kind") != "recovery_execution_approval":
        raise RuntimeError("review approval has the wrong kind")
    if payload.get("protocol_version") != RECOVERY_PROTOCOL_VERSION:
        raise RuntimeError("review approval has the wrong recovery protocol")
    if payload.get("status") != "approved":
        raise RuntimeError("Gate-A SFT remains review-locked")
    stages = payload.get("approved_stages")
    if not isinstance(stages, list) or approved_stage not in stages:
        raise RuntimeError(
            f"review approval does not unlock {approved_stage}"
        )
    if not str(payload.get("reviewer", "")).strip():
        raise RuntimeError("review approval is missing reviewer identity")
    return payload


def _world_size() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _isolate_triton_cache() -> Path:
    """Give every torchrun rank its own Triton compilation directory."""

    root = Path(
        os.environ.get(
            "AGZ_TRITON_CACHE_ROOT", "/tmp/agentguard_zero_triton"
        )
    )
    cache = root / "recovery_sft" / f"local_rank_{os.environ.get('LOCAL_RANK', '0')}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=["qwen3.5_base", "vda_1", "teacher_sft", "teacher_dagger"],
        required=True,
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--review-approval", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--max-length", type=int, default=4416)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--approved-stage", default="bootstrap_sft")
    parser.add_argument("--min-records", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--target-format",
        choices=["salient_action_first", INTENT_FORMAT],
        default="salient_action_first",
    )
    args = parser.parse_args()

    base_arms = {"qwen3.5_base", "teacher_sft"}
    continuation_arms = {"vda_1", "teacher_dagger"}
    if args.arm in base_arms and args.init_adapter is not None:
        raise ValueError("base arm must not receive --init-adapter")
    if args.arm in continuation_arms and args.init_adapter is None:
        raise ValueError("continuation arm requires --init-adapter")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    review_approval = (
        _load_review_approval(args.review_approval, args.approved_stage)
        if args.review_approval is not None
        else {"reviewer": "direct_user_fixed500_vda"}
    )
    data_manifest = _load_accepted_manifest(args.data_manifest)
    if sha256_file(args.train_parquet) != (
        json.loads((args.data_manifest.parent / "SHA256SUMS.json").read_text()).get(
            args.train_parquet.name
        )
    ):
        raise RuntimeError("bootstrap parquet hash does not match SHA256SUMS.json")

    triton_cache = _isolate_triton_cache()
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:  # pragma: no cover - server dependency gate
        raise RuntimeError(
            "bootstrap SFT requires torch, transformers, accelerate, and peft"
        ) from exc

    cfg = RecoveryConfig().bootstrap_sft
    random.seed(args.seed)
    set_seed(args.seed)
    rows = pd.read_parquet(args.train_parquet).to_dict(orient="records")
    record_min = args.min_records or cfg.pilot_records_min
    record_max = args.max_records or cfg.pilot_records_max
    if not record_min <= len(rows) <= record_max:
        raise RuntimeError("bootstrap parquet violates the frozen record-count gate")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class BootstrapDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = rows[index]
            if args.target_format == INTENT_FORMAT:
                target = action_intent_wire_json(str(row["target"]))
                prompt = compact_intent_prompt(str(row["prompt"]))
            else:
                target = action_first_wire_json(str(row["target"]))
                prompt = str(row["prompt"])
            prompt_messages = [{"role": "user", "content": prompt}]
            full_messages = [
                *prompt_messages,
                {"role": "assistant", "content": target},
            ]
            if getattr(tokenizer, "chat_template", None):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                full_text = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            else:
                prompt_text = str(row["prompt"])
                full_text = prompt_text + str(row["target"])
            full = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length,
            )["input_ids"]
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length,
            )["input_ids"]
            if len(full) <= len(prompt_ids):
                raise RuntimeError("bootstrap target was completely truncated")
            if full[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(
                    "non-thinking prompt is not an exact token prefix of target"
                )
            labels = [-100] * min(len(prompt_ids), len(full)) + full[len(prompt_ids) :]
            return {
                "input_ids": full,
                "attention_mask": [1] * len(full),
                "labels": labels,
            }

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        maximum = max(len(item["input_ids"]) for item in batch)
        input_ids, masks, labels = [], [], []
        for item in batch:
            padding = maximum - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.init_adapter is None:
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=0.0,
                target_modules=list(cfg.lora_target_modules),
                task_type="CAUSAL_LM",
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            str(args.init_adapter),
            is_trainable=True,
        )
        peft_cfg = model.peft_config["default"]
        if (
            int(peft_cfg.r) != cfg.lora_rank
            or int(peft_cfg.lora_alpha) != cfg.lora_alpha
        ):
            raise RuntimeError(
                "VDA1 adapter LoRA shape differs from the frozen Gate-A config"
            )
        if set(peft_cfg.target_modules) != set(cfg.lora_target_modules):
            raise RuntimeError(
                "VDA1 adapter target modules differ from the frozen Gate-A config"
            )
    model.config.use_cache = False

    effective_batch = args.per_device_batch * _world_size() * args.gradient_accumulation
    if effective_batch != cfg.effective_batch_size:
        raise RuntimeError(
            f"effective batch must be {cfg.effective_batch_size}, got {effective_batch}"
        )
    trainer_output = args.output_dir / "trainer"
    training_args = TrainingArguments(
        output_dir=str(trainer_output),
        num_train_epochs=args.epochs if args.epochs is not None else cfg.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else cfg.learning_rate
        ),
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=BootstrapDataset(),
        data_collator=collate,
    )
    result = trainer.train()
    if _rank() == 0:
        adapter_dir = args.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        manifest = {
            "schema_version": 1,
            "kind": "recovery_bootstrap_sft_adapter",
            "protocol_version": RECOVERY_PROTOCOL_VERSION,
            "status": "trained_pending_k1_evaluation",
            "created_at": utc_now(),
            "arm": args.arm,
            "base_model": model_identity(args.model_path),
            "initial_adapter_path": (
                str(args.init_adapter.resolve()) if args.init_adapter else None
            ),
            "initial_adapter_sha256": (
                sha256_tree(args.init_adapter) if args.init_adapter else None
            ),
            "training_data_manifest": str(args.data_manifest.resolve()),
            "training_data_manifest_sha256": sha256_file(args.data_manifest),
            "review_approval": (
                str(args.review_approval.resolve()) if args.review_approval else None
            ),
            "review_approval_sha256": (
                sha256_file(args.review_approval) if args.review_approval else None
            ),
            "reviewer": review_approval["reviewer"],
            "training_parquet_sha256": sha256_file(args.train_parquet),
            "training_record_count": len(rows),
            "training_config": asdict(cfg),
            "training_overrides": {
                "approved_stage": args.approved_stage,
                "record_min": record_min,
                "record_max": record_max,
                "epochs": (
                    args.epochs if args.epochs is not None else cfg.epochs
                ),
                "learning_rate": (
                    args.learning_rate
                    if args.learning_rate is not None
                    else cfg.learning_rate
                ),
            },
            "seed": args.seed,
            "world_size": _world_size(),
            "triton_cache_isolated_per_local_rank": True,
            "rank0_triton_cache": str(triton_cache),
            "effective_batch_size": effective_batch,
            "thinking_mode": "disabled",
            "prompt_target_token_prefix_exact": True,
            "target_serialization": (
                INTENT_FORMAT
                if args.target_format == INTENT_FORMAT
                else "schema_v4_salient_action_field_first"
            ),
            "global_step": int(trainer.state.global_step),
            "train_loss": float(result.training_loss),
            "adapter_path": str(adapter_dir.resolve()),
            "adapter_sha256": sha256_tree(adapter_dir),
            "data_manifest_snapshot": data_manifest,
            "next_stage": "fixed_xplay_k1_greedy_evaluation",
        }
        atomic_write_json(args.output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
