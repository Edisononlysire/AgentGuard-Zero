#!/usr/bin/env python3
"""Train the direct candidate scorer with family, listwise, or hard-negative loss."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.generator import ACTION_FAMILIES
from agentguard_zero.candidate.model import (
    FORMAT_VERSION,
    candidate_pair_text,
    load_ranker_components,
    score_all_encoded,
    score_encoded,
)
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    model_identity,
    sha256_file,
    sha256_tree,
    utc_now,
)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--init-score-head", type=Path)
    parser.add_argument("--init-heads", type=Path)
    parser.add_argument(
        "--objective", choices=["family", "listwise", "joint", "preference"], required=True
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if manifest.get("accepted") is not True:
        raise RuntimeError("candidate dataset manifest is not accepted")
    if sha256_file(args.train_jsonl) != manifest.get("candidate_sets_sha256"):
        raise RuntimeError("candidate dataset hash mismatch")
    if args.init_score_head and args.init_heads:
        raise ValueError("choose either legacy score head or multi-head initialization")
    if bool(args.init_adapter) != bool(args.init_score_head or args.init_heads):
        raise ValueError("adapter and ranker heads must be initialized together")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cache_root = Path(
        os.environ.get(
            "AGZ_TRITON_CACHE_ROOT",
            f"/tmp/agentguard_zero_triton_{os.environ.get('USER', 'user')}",
        )
    )
    rank_cache = cache_root / f"candidate_ranker_rank_{local_rank}"
    rank_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(rank_cache)

    try:
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import Dataset
        from transformers import Trainer, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover - GPU dependency gate
        raise RuntimeError("candidate training requires torch and transformers") from exc

    random.seed(args.seed)
    set_seed(args.seed)
    records = _read_records(args.train_jsonl)
    if len(records) != int(manifest.get("record_count", -1)):
        raise RuntimeError("candidate record count disagrees with manifest")
    tokenizer, backbone, heads = load_ranker_components(
        model_path=args.model_path,
        adapter_path=args.init_adapter,
        heads_path=args.init_heads,
        score_head_path=args.init_score_head,
        trainable=True,
    )
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    family_index = {name: index for index, name in enumerate(ACTION_FAMILIES)}

    class CandidateDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return records[index]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        group_lengths: list[int] = []
        teacher_probabilities: list[float] = []
        candidate_families: list[int] = []
        beliefs: list[list[float]] = []
        uncertainties: list[float] = []
        probe_values: list[float] = []
        business_risks: list[float] = []
        safety_risks: list[float] = []
        target_indices: list[int] = []
        target_families: list[int] = []
        negative_indices: list[list[int]] = []
        probe_chain_states: list[bool] = []
        probe_grounded_masks: list[list[bool]] = []
        for row in batch:
            candidates = [CandidateOption.from_record(item) for item in row["candidates"]]
            observation = dict(row["public_observation"])
            texts.extend(candidate_pair_text(observation, item) for item in candidates)
            group_lengths.append(len(candidates))
            probabilities = list(map(float, row["teacher_probabilities"]))
            if len(probabilities) != len(candidates):
                raise RuntimeError("teacher probability count mismatch")
            teacher_probabilities.extend(probabilities)
            candidate_families.extend(family_index[item.action_family] for item in candidates)
            auxiliary = list(row.get("auxiliary_targets") or [])
            if len(auxiliary) != len(candidates):
                raise RuntimeError("auxiliary target count mismatch")
            beliefs.extend([list(map(float, item["belief"])) for item in auxiliary])
            uncertainties.extend(float(item["uncertainty"]) for item in auxiliary)
            probe_values.extend(float(item["probe_value"]) for item in auxiliary)
            business_risks.extend(float(item["business_risk"]) for item in auxiliary)
            safety_risks.extend(float(item["safety_risk"]) for item in auxiliary)
            ids = [item.candidate_id for item in candidates]
            target_indices.append(ids.index(str(row["target_candidate_id"])))
            target_families.append(family_index[str(row["target_family"])])
            negative_indices.append(
                [
                    ids.index(str(item))
                    for item in row.get("hard_negative_candidate_ids", [])
                    if str(item) in ids
                ]
            )
            probe_chain = row.get("probe_chain_target") or {}
            probe_chain_states.append(
                bool(probe_chain.get("is_probe_followup_state", False))
            )
            probe_evidence_id = str(probe_chain.get("probe_evidence_id") or "")
            probe_grounded_masks.append(
                [
                    bool(probe_evidence_id and probe_evidence_id in item.referenced_ids)
                    for item in candidates
                ]
            )
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {
            **encoded,
            "group_lengths": torch.tensor(group_lengths, dtype=torch.long),
            "teacher_probabilities": torch.tensor(teacher_probabilities, dtype=torch.float32),
            "candidate_families": torch.tensor(candidate_families, dtype=torch.long),
            "belief_targets": torch.tensor(beliefs, dtype=torch.float32),
            "uncertainty_targets": torch.tensor(uncertainties, dtype=torch.float32),
            "probe_value_targets": torch.tensor(probe_values, dtype=torch.float32),
            "business_risk_targets": torch.tensor(business_risks, dtype=torch.float32),
            "safety_risk_targets": torch.tensor(safety_risks, dtype=torch.float32),
            "target_indices": torch.tensor(target_indices, dtype=torch.long),
            "target_families": torch.tensor(target_families, dtype=torch.long),
            "negative_indices": negative_indices,
            "probe_chain_states": probe_chain_states,
            "probe_grounded_masks": probe_grounded_masks,
        }

    class RankerModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.heads = heads

        def forward(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            return score_all_encoded(
                self.backbone,
                self.heads,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
            kwargs = dict(gradient_checkpointing_kwargs or {})
            self.backbone.gradient_checkpointing_enable(**kwargs)

        def gradient_checkpointing_disable(self) -> None:
            self.backbone.gradient_checkpointing_disable()

    class RankerTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            group_lengths = inputs.pop("group_lengths")
            teacher_probabilities = inputs.pop("teacher_probabilities")
            candidate_families = inputs.pop("candidate_families")
            belief_targets = inputs.pop("belief_targets")
            uncertainty_targets = inputs.pop("uncertainty_targets")
            probe_value_targets = inputs.pop("probe_value_targets")
            business_risk_targets = inputs.pop("business_risk_targets")
            safety_risk_targets = inputs.pop("safety_risk_targets")
            target_indices = inputs.pop("target_indices")
            target_families = inputs.pop("target_families")
            negative_indices = inputs.pop("negative_indices")
            probe_chain_states = inputs.pop("probe_chain_states")
            probe_grounded_masks = inputs.pop("probe_grounded_masks")
            predictions = model(**inputs)
            scores = predictions["utility"]
            family_losses = []
            rank_losses = []
            hard_losses = []
            probe_chain_losses = []
            offset = 0
            for group_index, raw_length in enumerate(group_lengths.tolist()):
                length = int(raw_length)
                group_scores = scores[offset : offset + length]
                group_probs = teacher_probabilities[offset : offset + length]
                group_families = candidate_families[offset : offset + length]
                rank_losses.append(
                    -(group_probs * functional.log_softmax(group_scores, dim=0)).sum()
                )
                family_logits = []
                for index in range(len(ACTION_FAMILIES)):
                    mask = group_families.eq(index)
                    family_logits.append(
                        torch.logsumexp(group_scores[mask], dim=0)
                        if mask.any()
                        else group_scores.new_tensor(-1.0e4)
                    )
                family_losses.append(
                    functional.cross_entropy(
                        torch.stack(family_logits).unsqueeze(0),
                        target_families[group_index].unsqueeze(0),
                    )
                )
                positive = group_scores[int(target_indices[group_index])]
                if probe_chain_states[group_index]:
                    observe_index = next(
                        (
                            index
                            for index in range(length)
                            if int(group_families[index])
                            == family_index["observe"]
                        ),
                        None,
                    )
                    if observe_index is not None:
                        probe_chain_losses.append(
                            functional.relu(
                                args.margin
                                - positive
                                + group_scores[int(observe_index)]
                            )
                        )
                    grounded = probe_grounded_masks[group_index]
                    if grounded[int(target_indices[group_index])]:
                        ungrounded_indices = [
                            index for index, value in enumerate(grounded) if not value
                        ]
                        if ungrounded_indices:
                            strongest_ungrounded = torch.stack(
                                [group_scores[index] for index in ungrounded_indices]
                            ).max()
                            probe_chain_losses.append(
                                functional.relu(
                                    args.margin - positive + strongest_ungrounded
                                )
                            )
                for negative in negative_indices[group_index]:
                    hard_losses.append(
                        functional.relu(args.margin - positive + group_scores[int(negative)])
                    )
                offset += length
            decision_family_loss = torch.stack(family_losses).mean()
            candidate_family_loss = functional.cross_entropy(
                predictions["family"], candidate_families
            )
            family_loss = 0.5 * (decision_family_loss + candidate_family_loss)
            rank_loss = torch.stack(rank_losses).mean()
            hard_loss = (
                torch.stack(hard_losses).mean() if hard_losses else scores.sum() * 0.0
            )
            probe_chain_loss = (
                torch.stack(probe_chain_losses).mean()
                if probe_chain_losses
                else scores.sum() * 0.0
            )
            belief_loss = functional.kl_div(
                functional.log_softmax(predictions["belief"], dim=-1),
                belief_targets,
                reduction="batchmean",
            )
            uncertainty_loss = functional.mse_loss(
                torch.sigmoid(predictions["uncertainty"]), uncertainty_targets
            )
            probe_mask = candidate_families.eq(family_index["active_probe"])
            probe_loss = (
                functional.mse_loss(
                    predictions["probe_value"][probe_mask], probe_value_targets[probe_mask]
                )
                if probe_mask.any()
                else scores.sum() * 0.0
            )
            risk_loss = functional.mse_loss(
                torch.sigmoid(predictions["business_risk"]), business_risk_targets
            ) + functional.mse_loss(
                torch.sigmoid(predictions["safety_risk"]), safety_risk_targets
            )
            weights = {
                "family": (1.0, 0.1, 0.0),
                "listwise": (0.3, 1.0, 0.0),
                "joint": (0.3, 1.0, 0.3),
                "preference": (0.1, 0.1, 1.0),
            }[args.objective]
            loss = (
                weights[0] * family_loss
                + weights[1] * rank_loss
                + weights[2] * hard_loss
                + 0.5 * probe_loss
                + 0.5 * probe_chain_loss
                + 0.2 * belief_loss
                + 0.2 * uncertainty_loss
                + 0.2 * risk_loss
            )
            outputs = {
                "scores": scores.detach(),
                "family_loss": family_loss.detach(),
                "rank_loss": rank_loss.detach(),
                "hard_loss": hard_loss.detach(),
                "probe_loss": probe_loss.detach(),
                "probe_chain_loss": probe_chain_loss.detach(),
                "belief_loss": belief_loss.detach(),
                "risk_loss": risk_loss.detach(),
            }
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
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
    model = RankerModel()
    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=CandidateDataset(),
        data_collator=collate,
    )
    result = trainer.train()
    if _rank() == 0:
        adapter_dir = args.output_dir / "adapter"
        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        model.backbone.save_pretrained(str(adapter_dir))
        heads_path = args.output_dir / "heads.pt"
        torch.save(
            {key: value.detach().cpu() for key, value in model.heads.state_dict().items()},
            heads_path,
        )
        score_head_path = args.output_dir / "score_head.pt"
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in model.heads["utility"].state_dict().items()
            },
            score_head_path,
        )
        output_manifest = {
            "schema_version": 2,
            "kind": "candidate_ranker_checkpoint",
            "created_at": utc_now(),
            "status": "trained_pending_evaluation",
            "objective": args.objective,
            "format_version": FORMAT_VERSION,
            "base_model": model_identity(args.model_path),
            "adapter_path": str(adapter_dir.resolve()),
            "adapter_sha256": sha256_tree(adapter_dir),
            "score_head_path": str(score_head_path.resolve()),
            "score_head_sha256": sha256_file(score_head_path),
            "heads_path": str(heads_path.resolve()),
            "heads_sha256": sha256_file(heads_path),
            "final_action_head": "utility",
            "source_data_manifest": str(args.data_manifest.resolve()),
            "source_data_manifest_sha256": sha256_file(args.data_manifest),
            "train_metrics": dict(result.metrics),
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "seed": args.seed,
        }
        atomic_write_json(args.output_dir / "manifest.json", output_manifest)
        print(json.dumps(output_manifest, ensure_ascii=False, sort_keys=True))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        # Qwen3.5 FLA leaves non-daemon compilation workers alive on the
        # cluster's Python build. The checkpoint is already fsync'ed; force a
        # clean rank exit so torchrun does not hang after successful training.
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
