#!/usr/bin/env python3
"""Conservative offline DPO branch initialized from the shared Teacher SFT."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION
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


def load_approval(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("approved_stages") if isinstance(payload, dict) else None
    if (
        payload.get("kind") != "recovery_execution_approval"
        or payload.get("protocol_version") != RECOVERY_PROTOCOL_VERSION
        or payload.get("status") != "approved"
        or not isinstance(stages, list)
        or "offline_preference_optimization" not in stages
    ):
        raise RuntimeError("offline preference optimization remains review-locked")
    return payload


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def isolate_triton_cache() -> Path:
    root = Path(
        os.environ.get(
            "AGZ_TRITON_CACHE_ROOT", "/tmp/agentguard_zero_triton"
        )
    )
    cache = root / "recovery_dpo" / f"local_rank_{os.environ.get('LOCAL_RANK', '0')}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--review-approval", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-length", type=int, default=4416)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument(
        "--target-format",
        choices=["salient_action_first", INTENT_FORMAT],
        default="salient_action_first",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    approval = (
        load_approval(args.review_approval)
        if args.review_approval is not None
        else {"reviewer": "direct_user_fixed500_vda"}
    )
    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if data_manifest.get("accepted") is not True:
        raise RuntimeError("preference data is not accepted")
    hashes = json.loads(
        (args.data_manifest.parent / "SHA256SUMS.json").read_text(encoding="utf-8")
    )
    if sha256_file(args.preferences) != hashes.get(args.preferences.name):
        raise RuntimeError("preference parquet hash mismatch")
    if not args.init_adapter.is_dir():
        raise RuntimeError("initial Teacher SFT adapter is missing")

    triton_cache = isolate_triton_cache()
    try:
        import torch
        import torch.nn.functional as functional
        from peft import PeftModel
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("DPO requires torch, transformers, accelerate, and peft") from exc

    random.seed(args.seed)
    set_seed(args.seed)
    rows = pd.read_parquet(args.preferences).to_dict(orient="records")
    if len(rows) != int(data_manifest.get("pair_count", -1)):
        raise RuntimeError("preference count disagrees with manifest")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(prompt: str, response: str) -> dict[str, list[int]]:
        if args.target_format == INTENT_FORMAT:
            response = action_intent_wire_json(response)
            prompt = compact_intent_prompt(prompt)
        else:
            response = action_first_wire_json(response)
        prompt_messages = [{"role": "user", "content": prompt}]
        full_messages = [
            *prompt_messages,
            {"role": "assistant", "content": response},
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
            prompt_text = prompt
            full_text = prompt + response
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
            raise RuntimeError("preference response was completely truncated")
        if full[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                "non-thinking prompt is not an exact token prefix of preference"
            )
        labels = [-100] * min(len(prompt_ids), len(full)) + full[len(prompt_ids) :]
        return {
            "input_ids": full,
            "attention_mask": [1] * len(full),
            "labels": labels,
        }

    class PreferenceDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = rows[index]
            chosen = encode(str(row["prompt"]), str(row["chosen"]))
            rejected = encode(str(row["prompt"]), str(row["rejected"]))
            return {
                f"chosen_{key}": value for key, value in chosen.items()
            } | {f"rejected_{key}": value for key, value in rejected.items()}

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for prefix in ("chosen", "rejected"):
            maximum = max(len(item[f"{prefix}_input_ids"]) for item in batch)
            ids, masks, labels = [], [], []
            for item in batch:
                padding = maximum - len(item[f"{prefix}_input_ids"])
                ids.append(
                    item[f"{prefix}_input_ids"]
                    + [tokenizer.pad_token_id] * padding
                )
                masks.append(item[f"{prefix}_attention_mask"] + [0] * padding)
                labels.append(item[f"{prefix}_labels"] + [-100] * padding)
            output[f"{prefix}_input_ids"] = torch.tensor(ids, dtype=torch.long)
            output[f"{prefix}_attention_mask"] = torch.tensor(
                masks, dtype=torch.long
            )
            output[f"{prefix}_labels"] = torch.tensor(labels, dtype=torch.long)
        return output

    def load_adapter(*, trainable: bool):
        base = AutoModelForCausalLM.from_pretrained(
            str(args.model_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        loaded = PeftModel.from_pretrained(
            base, str(args.init_adapter), is_trainable=trainable
        )
        loaded.config.use_cache = False
        return loaded

    model = load_adapter(trainable=True)
    reference = load_adapter(trainable=False)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    reference.to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    def response_logp(active_model: Any, prefix: str, inputs: dict[str, Any]):
        input_ids = inputs[f"{prefix}_input_ids"]
        attention_mask = inputs[f"{prefix}_attention_mask"]
        labels = inputs[f"{prefix}_labels"]
        logits = active_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits[:, :-1, :]
        shifted = labels[:, 1:]
        mask = shifted.ne(-100)
        safe_labels = shifted.masked_fill(~mask, 0)
        token_logp = functional.log_softmax(logits.float(), dim=-1).gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        # Length normalization follows the multi-turn preference result that
        # unequal trajectory lengths otherwise bias the Bradley-Terry score.
        return (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    class DPOTrainer(Trainer):
        def compute_loss(
            self,
            active_model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ):
            policy_chosen = response_logp(active_model, "chosen", inputs)
            policy_rejected = response_logp(active_model, "rejected", inputs)
            with torch.inference_mode():
                ref_chosen = response_logp(reference, "chosen", inputs)
                ref_rejected = response_logp(reference, "rejected", inputs)
            logits = args.beta * (
                (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            )
            loss = -functional.logsigmoid(logits).mean()
            if return_outputs:
                return loss, {"preference_margin": logits.detach().mean()}
            return loss

    effective_batch = args.per_device_batch * world_size() * args.gradient_accumulation
    if effective_batch != 64:
        raise RuntimeError(f"DPO effective batch must be 64, got {effective_batch}")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,
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
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=PreferenceDataset(),
        data_collator=collate,
    )
    result = trainer.train()
    if rank() == 0:
        adapter = args.output_dir / "adapter"
        adapter.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        manifest = {
            "schema_version": 1,
            "kind": "recovery_offline_dpo_adapter",
            "protocol_version": RECOVERY_PROTOCOL_VERSION,
            "status": "trained_pending_k1_evaluation",
            "created_at": utc_now(),
            "variant": "teacher_sft_plus_offline_dpo",
            "base_model": model_identity(args.model_path),
            "initial_adapter_path": str(args.init_adapter.resolve()),
            "initial_adapter_sha256": sha256_tree(args.init_adapter),
            "preferences_sha256": sha256_file(args.preferences),
            "preference_manifest_sha256": sha256_file(args.data_manifest),
            "pair_count": len(rows),
            "review_approval_sha256": (
                sha256_file(args.review_approval) if args.review_approval else None
            ),
            "reviewer": approval.get("reviewer"),
            "seed": args.seed,
            "world_size": world_size(),
            "triton_cache_isolated_per_local_rank": True,
            "rank0_triton_cache": str(triton_cache),
            "effective_batch_size": effective_batch,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "length_normalized_log_probability": True,
            "thinking_mode": "disabled",
            "prompt_target_token_prefix_exact": True,
            "target_serialization": (
                INTENT_FORMAT
                if args.target_format == INTENT_FORMAT
                else "schema_v4_salient_action_field_first"
            ),
            "global_step": int(trainer.state.global_step),
            "train_loss": float(result.training_loss),
            "adapter_path": str(adapter.resolve()),
            "adapter_sha256": sha256_tree(adapter),
            "next_stage": "fixed_xplay_k1_greedy_evaluation",
        }
        atomic_write_json(args.output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
