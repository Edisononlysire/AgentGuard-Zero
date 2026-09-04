#!/usr/bin/env python3
"""Train the direct candidate scorer on policy-consistent defense decisions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import random
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.generator import (
    ACTION_FAMILIES,
    candidate_response_cost,
)
from agentguard_zero.candidate.branching import (
    BRANCH_NAMES,
    BUDGET_TARGET_NAMES,
    MEMORY_OPERATION_NAMES,
    PHASE_NAMES,
    PHASE_NAMES_V2,
    SHARED_LOCAL_LONG_ARCHITECTURE,
    SHARED_LOCAL_LONG_ARCHITECTURES,
    SHARED_LOCAL_LONG_ARCHITECTURE_V2,
    derive_branch_supervision,
    memory_operation_name,
)
from agentguard_zero.candidate.model import (
    BRANCHED_HEAD_NAMES,
    FORMAT_VERSION,
    OUTCOME_HEAD_NAMES,
    RESIDUAL_HEAD_NAMES,
    V9_OUTCOME_SCORE_WEIGHTS,
    activate_ranker_adapter,
    candidate_pair_text,
    compose_candidate_utility,
    load_ranker_components,
    pool_encoded,
    public_state_text,
    route_branched_experts,
    score_all_encoded,
    score_all_pooled,
)
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    model_identity,
    sha256_file,
    sha256_tree,
    utc_now,
)


SUPPORT_FLAGS = (
    "passive_verification",
    "active_probe",
    "trust",
    "memory_operation",
    "memory_use",
    "mitigation",
)
V9_ARMS = (
    "scale_control",
    "hierarchical_distill",
    "hierarchical_outcome",
    "regret_distill",
    "lipo_lambda",
    "causal_conservative",
    "integrated_hrrc",
    "causal_chain",
)


def save_ranker_adapter(backbone: Any, destination: Path) -> None:
    """Persist either the legacy single adapter or both isolated adapters."""

    names = tuple(getattr(backbone, "agentguard_adapter_names", ()))
    kwargs = {"selected_adapters": list(names)} if names else {}
    backbone.save_pretrained(str(destination), **kwargs)


def ranker_adapter_config_path(adapter_root: Path) -> Path:
    """Return the config used for shared LoRA hyperparameter reporting."""

    direct = adapter_root / "adapter_config.json"
    local = adapter_root / "local" / "adapter_config.json"
    if direct.is_file():
        return direct
    if local.is_file():
        return local
    raise FileNotFoundError(f"ranker adapter config missing under {adapter_root}")


def outcome_target_composite(outcome: dict[str, Any]) -> float:
    """Use the pre-existing V9 outcome utility to score one frozen label."""

    return float(
        sum(
            float(weight) * float(outcome[name])
            for name, weight in V9_OUTCOME_SCORE_WEIGHTS.items()
        )
    )


def outcome_aligned_target_family(record: dict[str, Any]) -> str:
    """Return the family with the best candidate-constrained outcome target."""

    candidates = list(record.get("candidates") or [])
    outcomes = list(record.get("outcome_targets") or [])
    if not candidates or len(candidates) != len(outcomes):
        raise ValueError("outcome-aligned family requires candidate/outcome parity")
    family_best: dict[str, float] = {}
    for candidate, outcome in zip(candidates, outcomes, strict=True):
        family = str(candidate.get("action_family", ""))
        if family not in ACTION_FAMILIES:
            raise ValueError(f"unknown outcome-aligned action family: {family}")
        value = outcome_target_composite(dict(outcome))
        family_best[family] = max(family_best.get(family, float("-inf")), value)
    return max(
        family_best,
        key=lambda family: (family_best[family], -ACTION_FAMILIES.index(family)),
    )


def target_family_training_weights(
    records: list[dict[str, Any]],
    family_index: dict[str, int],
    *,
    maximum_weight: float,
) -> list[float]:
    """Upweight sparse defensive targets without downweighting common skills."""

    if maximum_weight < 1.0:
        raise ValueError("maximum family balance weight must be at least one")
    counts = Counter(str(row["target_family"]) for row in records)
    nonobserve_counts = [
        count
        for family, count in counts.items()
        if family != "observe" and count > 0
    ]
    reference = max(nonobserve_counts, default=1)
    weights = [1.0] * len(family_index)
    for family, index in family_index.items():
        count = counts.get(family, 0)
        if count > 0:
            weights[index] = min(
                float(maximum_weight),
                max(1.0, math.sqrt(reference / count)),
            )
    return weights


def outcome_record_training_weights(
    records: list[dict[str, Any]],
    outcome_families: list[str],
    *,
    maximum_weight: float,
) -> list[float]:
    """Balance the task-macro outcome objective without resampling records."""

    if len(records) != len(outcome_families):
        raise ValueError("outcome family count must match the training records")
    if maximum_weight < 1.0:
        raise ValueError("maximum outcome balance weight must be at least one")
    task_counts = Counter(str(row.get("task_id", "unknown")) for row in records)
    task_family_counts = Counter(
        (str(row.get("task_id", "unknown")), family)
        for row, family in zip(records, outcome_families, strict=True)
    )
    task_family_reference = {
        task: max(
            count
            for (family_task, _), count in task_family_counts.items()
            if family_task == task
        )
        for task in task_counts
    }
    largest_task = max(task_counts.values(), default=1)
    weights = []
    for row, family in zip(records, outcome_families, strict=True):
        task = str(row.get("task_id", "unknown"))
        task_weight = math.sqrt(largest_task / task_counts[task])
        family_weight = math.sqrt(
            task_family_reference[task] / task_family_counts[(task, family)]
        )
        weights.append(
            min(float(maximum_weight), max(1.0, task_weight * family_weight))
        )
    return weights


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def initialize_local_heads_from_legacy(heads: Any) -> dict[str, str]:
    """Seed the new local expert from the same D0 legacy decision heads."""

    mapping = {
        "local_utility": "utility",
        "local_family": "family",
    }
    missing = [name for pair in mapping.items() for name in pair if name not in heads]
    if missing:
        raise ValueError(
            "local-head initialization is missing required heads: "
            + ", ".join(sorted(set(missing)))
        )
    for destination, source in mapping.items():
        heads[destination].load_state_dict(heads[source].state_dict(), strict=True)
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progressive-arm")
    parser.add_argument("--experiment-variant")
    parser.add_argument("--v9-arm", choices=V9_ARMS)
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--init-score-head", type=Path)
    parser.add_argument("--init-heads", type=Path)
    parser.add_argument("--initialization-manifest", type=Path)
    parser.add_argument(
        "--initialization-role",
        choices=(
            "d0_fair_main",
            "d0_four_task_joint",
            "d0_v11_t12_native_v2",
            "original_4k_warmstart_ablation",
            "dagger_correction",
            "coevolution_round",
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=("legacy", *SHARED_LOCAL_LONG_ARCHITECTURES),
        default="legacy",
    )
    parser.add_argument(
        "--objective",
        choices=[
            "family",
            "listwise",
            "joint",
            "preference",
            "defense",
            "branched_defense",
        ],
        required=True,
    )
    parser.add_argument(
        "--selection-mode",
        choices=[
            "candidate_argmax",
            "hierarchical_family_then_utility",
            "capability_chain_then_hierarchical",
        ],
        help="Inference policy written into the trained checkpoint manifest.",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-learning-rate", type=float)
    parser.add_argument("--head-learning-rate", type=float)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--save-each-epoch",
        action="store_true",
        help=(
            "Save a weights-only adapter/head checkpoint and bound manifest "
            "at every completed epoch without resetting optimizer state."
        ),
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the base model and LoRA so only learned heads are optimized.",
    )
    parser.add_argument(
        "--separate-local-long-adapters",
        action="store_true",
        help=(
            "Initialize independent local/long LoRA adapters from the same "
            "bound checkpoint and route each public-state batch to exactly one."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--fixed-padding-length",
        type=int,
        help=(
            "Pad every candidate and state sequence to this exact tensor length. "
            "Used by compute-matched ablations; must cover every untruncated input."
        ),
    )
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--candidate-forward-chunk-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--observe-target-weight", type=float, default=0.35)
    parser.add_argument("--max-family-balance-weight", type=float, default=1.0)
    parser.add_argument(
        "--initialize-local-heads-from-legacy",
        action="store_true",
        help=(
            "Copy the D0 utility/family parameters into the new local expert "
            "before training. This does not inherit an original-4K checkpoint."
        ),
    )
    parser.add_argument(
        "--local-teacher-distillation-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--local-acceptable-mass-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("linear", "constant", "cosine"),
        default="linear",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-truncation-rate", type=float, default=0.0)
    parser.add_argument(
        "--cache-frozen-embeddings",
        action="store_true",
        help=(
            "Precompute public state/candidate representations once during "
            "head-only training. Requires --freeze-backbone."
        ),
    )
    parser.add_argument("--embedding-cache-batch-size", type=int, default=8)
    parser.add_argument(
        "--trajectory-ordered-training",
        action="store_true",
        help=(
            "Train a single-rank rollout/DAgger correction pass in complete "
            "trajectory/step order instead of random state order."
        ),
    )
    parser.add_argument(
        "--memory-tier-encoding",
        action="store_true",
        help="Encode episode-local STM separately from evidence-gated LTM.",
    )
    parser.add_argument("--t3-memory-operation-weight", type=float, default=1.0)
    parser.add_argument(
        "--t3-memory-operation-mask-relative-floor", type=float, default=0.10
    )
    parser.add_argument("--t4-mitigation-ready-weight", type=float, default=1.0)
    parser.add_argument("--t4-remaining-budget-weight", type=float, default=4.0)
    parser.add_argument(
        "--disable-t4-binary-mask",
        action="store_true",
        help=(
            "Keep T4 mitigation candidates reachable and use the budget head as "
            "a soft score during training/calibration."
        ),
    )
    args = parser.parse_args()

    lora_learning_rate = float(
        args.lora_learning_rate or args.learning_rate or 3.0e-6
    )
    head_learning_rate = float(
        args.head_learning_rate or args.learning_rate or 3.0e-4
    )

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.v9_arm and args.objective != "defense":
        raise ValueError("V9 arms require the defense objective")
    if (args.architecture in SHARED_LOCAL_LONG_ARCHITECTURES) != (
        args.objective == "branched_defense"
    ):
        raise ValueError(
            "shared_local_long and branched_defense must be selected together"
        )
    if args.objective == "branched_defense" and args.v9_arm:
        raise ValueError("the branched V10 objective does not reuse a V9 arm")
    if (
        args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE
        and args.selection_mode != "hierarchical_family_then_utility"
    ):
        raise ValueError("shared_local_long_v1 requires hierarchical selection")
    if (
        args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
        and args.selection_mode != "capability_chain_then_hierarchical"
    ):
        raise ValueError(
            "shared_local_long_v2 requires the public execution-contract policy"
        )
    if args.progressive_arm and not (
        args.experiment_variant and args.protocol_manifest
    ):
        raise ValueError(
            "progressive training requires experiment variant and protocol manifest"
        )
    if args.protocol_manifest is not None and not args.protocol_manifest.is_file():
        raise FileNotFoundError(args.protocol_manifest)
    if args.initialization_manifest is not None and not args.initialization_manifest.is_file():
        raise FileNotFoundError(args.initialization_manifest)
    manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if manifest.get("accepted") is not True:
        raise RuntimeError("candidate dataset manifest is not accepted")
    if sha256_file(args.train_jsonl) != manifest.get("candidate_sets_sha256"):
        raise RuntimeError("candidate dataset hash mismatch")
    if args.init_score_head and args.init_heads:
        raise ValueError("choose either legacy score head or multi-head initialization")
    if args.init_adapter and not (args.init_score_head or args.init_heads):
        raise ValueError("adapter initialization requires ranker heads")
    if args.objective == "branched_defense" and not (
        args.initialization_manifest
        and args.initialization_role
        and args.init_adapter
        and args.init_heads
        and args.protocol_manifest
    ):
        raise ValueError(
            "branched training requires an initialization manifest, role, "
            "adapter, heads, and protocol manifest"
        )
    if args.cache_frozen_embeddings and not args.freeze_backbone:
        raise ValueError("embedding cache requires --freeze-backbone")
    if args.separate_local_long_adapters and (
        args.architecture != SHARED_LOCAL_LONG_ARCHITECTURE_V2
        or args.freeze_backbone
        or args.per_device_batch != 1
    ):
        raise ValueError(
            "separate local/long adapters require trainable shared_local_long_v2 "
            "with per-device batch size one"
        )
    if args.embedding_cache_batch_size < 1:
        raise ValueError("embedding cache batch size must be positive")
    if args.trajectory_ordered_training and (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or args.per_device_batch != 1
    ):
        raise ValueError(
            "trajectory-ordered DAgger training requires one rank and batch size one"
        )
    if args.fixed_padding_length is not None and not (
        1 <= args.fixed_padding_length <= args.max_length
    ):
        raise ValueError("fixed padding length must be in [1, max_length]")
    if args.save_each_epoch and not float(args.epochs).is_integer():
        raise ValueError("per-epoch checkpoints require an integer epoch count")
    if (
        args.local_teacher_distillation_weight < 0.0
        or args.local_acceptable_mass_weight < 0.0
    ):
        raise ValueError("local calibration loss weights must be non-negative")
    uses_local_joint_objective = bool(
        args.initialize_local_heads_from_legacy
        or args.local_teacher_distillation_weight
        or args.local_acceptable_mass_weight
    )
    if uses_local_joint_objective and args.objective != "branched_defense":
        raise ValueError("local joint options require the branched defense objective")
    if (
        args.t3_memory_operation_weight < 0.0
        or not 0.0 <= args.t3_memory_operation_mask_relative_floor <= 1.0
        or args.t4_mitigation_ready_weight < 0.0
        or args.t4_remaining_budget_weight < 0.0
    ):
        raise ValueError("long-horizon score and mask weights are out of range")

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
        from torch.utils.data import Dataset, SequentialSampler
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover - GPU dependency gate
        raise RuntimeError("candidate training requires torch and transformers") from exc

    random.seed(args.seed)
    set_seed(args.seed)
    records = _read_records(args.train_jsonl)
    if args.trajectory_ordered_training:
        if not all(row.get("trajectory_id") is not None for row in records):
            raise RuntimeError(
                "trajectory-ordered training requires trajectory_id on every row"
            )
        records.sort(
            key=lambda row: (
                str(row.get("trajectory_id", "")),
                int(row.get("trajectory_step", 0)),
                str(row.get("record_id", "")),
            )
        )
    if len(records) != int(manifest.get("record_count", -1)):
        raise RuntimeError("candidate record count disagrees with manifest")
    tokenizer, backbone, heads = load_ranker_components(
        model_path=args.model_path,
        adapter_path=args.init_adapter,
        heads_path=args.init_heads,
        score_head_path=args.init_score_head,
        trainable=True,
        architecture=args.architecture,
        separate_local_long_adapters=args.separate_local_long_adapters,
    )
    local_head_initialization = (
        initialize_local_heads_from_legacy(heads)
        if args.initialize_local_heads_from_legacy
        else None
    )
    initialization_manifest: dict[str, Any] | None = None
    if args.initialization_manifest is not None:
        initialization_manifest = json.loads(
            args.initialization_manifest.read_text(encoding="utf-8")
        )
        if initialization_manifest.get("kind") != "candidate_ranker_checkpoint":
            raise RuntimeError("initialization is not a candidate ranker checkpoint")
        if Path(str(initialization_manifest.get("adapter_path", ""))).resolve() != args.init_adapter.resolve():
            raise RuntimeError("initialization adapter does not match its manifest")
        if Path(str(initialization_manifest.get("heads_path", ""))).resolve() != args.init_heads.resolve():
            raise RuntimeError("initialization heads do not match their manifest")
        if sha256_tree(args.init_adapter) != initialization_manifest.get("adapter_sha256"):
            raise RuntimeError("initialization adapter hash mismatch")
        if sha256_file(args.init_heads) != initialization_manifest.get("heads_sha256"):
            raise RuntimeError("initialization heads hash mismatch")
        if args.initialization_role in {"d0_fair_main", "d0_four_task_joint"}:
            protocol = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
            expected = (protocol.get("initialization") or {}).get("manifest_sha256")
            if sha256_file(args.initialization_manifest) != expected:
                raise RuntimeError("D0 experiment must start from the bound initialization")
        elif args.initialization_role == "d0_v11_t12_native_v2":
            protocol = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
            expected = (protocol.get("artifacts") or {}).get("d0_manifest_sha256")
            if sha256_file(args.initialization_manifest) != expected:
                raise RuntimeError(
                    "V11 T1/T2 experiment must start from the bound D0 initialization"
                )
    if args.freeze_backbone:
        backbone.requires_grad_(False)
    if not args.freeze_backbone and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if args.objective in {"defense", "branched_defense"}:
        trainable_head_names = {
            "utility",
            "family",
            "support",
            *OUTCOME_HEAD_NAMES,
            *RESIDUAL_HEAD_NAMES,
        }
        if args.objective == "branched_defense":
            if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2:
                trainable_head_names = {
                    "utility",
                    "family",
                    "support",
                    *RESIDUAL_HEAD_NAMES,
                    *(
                        name
                        for name in BRANCHED_HEAD_NAMES
                        if name not in {"local_utility", "local_family"}
                    ),
                }
            else:
                trainable_head_names = {
                    "support",
                    *OUTCOME_HEAD_NAMES,
                    *BRANCHED_HEAD_NAMES,
                }
        for name, head in heads.items():
            if name not in trainable_head_names:
                head.requires_grad_(False)
    family_index = {name: index for index, name in enumerate(ACTION_FAMILIES)}
    needs_outcome_alignment = args.v9_arm == "hierarchical_outcome"
    needs_causal_chain = (
        args.v9_arm == "causal_chain" or args.objective == "branched_defense"
    )
    def effective_target_family(record: dict[str, Any]) -> str:
        """Use V10 causal labels when present; preserve retained T1/T2 labels."""

        if args.objective == "branched_defense":
            if (
                args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                and str(record.get("task_id", "")) in {"T1", "T2"}
            ):
                return str(record["target_family"])
            return str(
                record.get("causal_target_family") or record["target_family"]
            )
        if needs_causal_chain:
            return str(record["causal_target_family"])
        return str(record["target_family"])

    family_training_weights = target_family_training_weights(
        [
            {"target_family": effective_target_family(record)}
            for record in records
        ],
        family_index,
        maximum_weight=args.max_family_balance_weight,
    )
    local_original4k_family_training_weights = list(family_training_weights)
    if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2 and any(
        str(record.get("task_id", "")) in {"T1", "T2"} for record in records
    ):
        source_binding = manifest.get("source_binding") or {}
        original4k_path = Path(str(source_binding.get("original4k_data", "")))
        expected_original4k_sha256 = str(
            source_binding.get("original4k_data_sha256", "")
        )
        if original4k_path.is_file() and sha256_file(
            original4k_path
        ) == expected_original4k_sha256:
            original4k_records = _read_records(original4k_path)
            local_original4k_family_training_weights = target_family_training_weights(
                original4k_records,
                family_index,
                maximum_weight=args.max_family_balance_weight,
            )
        elif (
            args.initialization_role == "coevolution_round"
            and initialization_manifest
            and initialization_manifest.get(
                "local_original4k_family_training_weights"
            )
        ):
            inherited_weights = initialization_manifest[
                "local_original4k_family_training_weights"
            ]
            local_original4k_family_training_weights = [
                float(inherited_weights[family]) for family in ACTION_FAMILIES
            ]
        else:
            raise RuntimeError(
                "V2 local branch requires either its hash-bound original-4K "
                "source or an isolated co-evolution parent with bound weights"
            )
    outcome_target_family_names = (
        [outcome_aligned_target_family(record) for record in records]
        if needs_outcome_alignment
        else [
            effective_target_family(record)
            for record in records
        ]
    )
    outcome_family_training_weights = target_family_training_weights(
        [
            {"target_family": family}
            for family in outcome_target_family_names
        ],
        family_index,
        maximum_weight=args.max_family_balance_weight,
    )
    outcome_record_weights = outcome_record_training_weights(
        records,
        outcome_target_family_names,
        maximum_weight=args.max_family_balance_weight,
    )
    score_composition = (
        {
            "kind": (
                "shared_local_long_public_contract_v2"
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else "shared_local_long_phase_routed_v1"
            ),
            "architecture": args.architecture,
            "memory_tier_encoding": True,
            "local_memory_tier_encoding": False,
            "long_memory_tier_encoding": True,
            "t3_memory_operation_weight": (
                args.t3_memory_operation_weight
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else 0.0
            ),
            "t4_mitigation_ready_weight": (
                args.t4_mitigation_ready_weight
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else 0.0
            ),
            "t4_remaining_budget_weight": (
                args.t4_remaining_budget_weight
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else 0.0
            ),
            "t3_memory_operation_mask_relative_floor": (
                args.t3_memory_operation_mask_relative_floor
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else 0.0
            ),
            "t4_binary_mask_threshold": 0.5,
            "t4_binary_mask_enabled": not args.disable_t4_binary_mask,
            "public_state_legality_mask": (
                args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
            ),
            "adapter_strategy": (
                "separate_local_long"
                if args.separate_local_long_adapters
                else "shared"
            ),
            "local_branch_contract": (
                "original_4k_hierarchical_distill"
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else "learned_local_expert_v1"
            ),
        }
        if args.objective == "branched_defense"
        else
        {
            "kind": (
                "v10_causal_chain_composite_v1"
                if args.v9_arm == "causal_chain"
                else "v9_learned_outcome_composite_v1"
            ),
            "utility_weight": 1.0,
            "outcome_weight": (
                1.0
                if args.v9_arm == "hierarchical_outcome"
                else 0.25
                if args.v9_arm == "causal_chain"
                else 0.5
            ),
            "outcome_weights": dict(V9_OUTCOME_SCORE_WEIGHTS),
            "memory_tier_encoding": bool(args.memory_tier_encoding),
        }
        if args.v9_arm
        in {
            "hierarchical_outcome",
            "causal_conservative",
            "integrated_hrrc",
            "causal_chain",
        }
        else None
    )
    needs_hierarchical_distillation = args.v9_arm in {
        "hierarchical_distill",
        "hierarchical_outcome",
        "integrated_hrrc",
    }
    needs_regret_distillation = args.v9_arm in {
        "regret_distill",
        "integrated_hrrc",
    }
    needs_soft_family_distillation = args.v9_arm == "regret_distill"
    needs_lipo_lambda = args.v9_arm in {"lipo_lambda", "integrated_hrrc"}

    def row_memory_tier_encoding(row: dict[str, Any]) -> bool:
        return bool(
            args.memory_tier_encoding
            and not (
                args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                and str(row.get("task_id", "")) in {"T1", "T2"}
            )
        )

    token_lengths: list[int] = []
    state_token_lengths: list[int] = []
    text_batch: list[str] = []
    for row in records:
        candidates = [CandidateOption.from_record(item) for item in row["candidates"]]
        observation = dict(row["public_observation"])
        encode_memory_tiers = row_memory_tier_encoding(row)
        text_batch.extend(
            candidate_pair_text(
                observation,
                item,
                memory_tier_encoding=encode_memory_tiers,
            )
            for item in candidates
        )
        while len(text_batch) >= 256:
            encoded = tokenizer(
                text_batch[:256],
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            token_lengths.extend(map(len, encoded["input_ids"]))
            del text_batch[:256]
        state_encoded = tokenizer(
            [
                public_state_text(
                    observation, memory_tier_encoding=encode_memory_tiers
                )
            ],
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        state_token_lengths.extend(map(len, state_encoded["input_ids"]))
    if text_batch:
        encoded = tokenizer(
            text_batch,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        token_lengths.extend(map(len, encoded["input_ids"]))
    truncated_count = sum(length > args.max_length for length in token_lengths)
    state_truncated_count = sum(
        length > args.max_length for length in state_token_lengths
    )
    total_input_count = len(token_lengths) + len(state_token_lengths)
    truncation_rate = (truncated_count + state_truncated_count) / max(
        1, total_input_count
    )
    if truncation_rate > args.max_truncation_rate:
        raise RuntimeError(
            f"candidate input truncation rate {truncation_rate:.6f} exceeds "
            f"{args.max_truncation_rate:.6f}"
        )
    if args.fixed_padding_length is not None and max(
        max(token_lengths, default=0),
        max(state_token_lengths, default=0),
    ) > args.fixed_padding_length:
        raise RuntimeError(
            "an input exceeds the fixed padding length; refusing unequal truncation"
        )
    sorted_lengths = sorted(token_lengths)
    input_audit = {
        "candidate_pair_count": len(token_lengths),
        "public_state_count": len(state_token_lengths),
        "token_length_median": statistics.median(token_lengths) if token_lengths else 0,
        "token_length_p95": (
            sorted_lengths[min(len(sorted_lengths) - 1, int(len(sorted_lengths) * 0.95))]
            if sorted_lengths
            else 0
        ),
        "token_length_max": max(token_lengths, default=0),
        "max_length": args.max_length,
        "truncated_count": truncated_count,
        "state_truncated_count": state_truncated_count,
        "critical_field_truncated_count": truncated_count
        + state_truncated_count,
        "truncation_rate": truncation_rate,
    }

    cached_candidates: list[Any] | None = None
    cached_states: list[Any] | None = None
    if args.cache_frozen_embeddings:
        cache_padding_length = max(
            max(token_lengths, default=0),
            max(state_token_lengths, default=0),
        )
        if cache_padding_length < 1 or cache_padding_length > args.max_length:
            raise RuntimeError("invalid frozen embedding cache padding length")
        cache_device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        backbone.to(cache_device).eval()

        def cache_texts(texts: list[str]) -> Any:
            chunks = []
            with torch.inference_mode():
                for offset in range(0, len(texts), args.embedding_cache_batch_size):
                    encoded = tokenizer(
                        texts[offset : offset + args.embedding_cache_batch_size],
                        padding="max_length",
                        truncation=False,
                        max_length=cache_padding_length,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
                    encoded = {
                        key: value.to(cache_device) for key, value in encoded.items()
                    }
                    chunks.append(
                        pool_encoded(
                            backbone,
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                        )
                        .detach()
                        .cpu()
                    )
            return torch.cat(chunks, dim=0)

        cached_candidates = []
        state_text_batch = []
        for row in records:
            observation = dict(row["public_observation"])
            encode_memory_tiers = row_memory_tier_encoding(row)
            candidates = [
                CandidateOption.from_record(item) for item in row["candidates"]
            ]
            cached_candidates.append(
                cache_texts(
                    [
                        candidate_pair_text(
                            observation,
                            item,
                            memory_tier_encoding=encode_memory_tiers,
                        )
                        for item in candidates
                    ]
                )
            )
            state_text_batch.append(
                public_state_text(
                    observation, memory_tier_encoding=encode_memory_tiers
                )
            )
        cached_states = [
            row for row in cache_texts(state_text_batch)
        ]
        backbone.train()
        input_audit["frozen_embedding_cache"] = True
        input_audit["embedding_cache_batch_size"] = args.embedding_cache_batch_size
        input_audit["embedding_cache_padding_length"] = cache_padding_length
    else:
        input_audit["frozen_embedding_cache"] = False

    padding: bool | str = (
        "max_length" if args.fixed_padding_length is not None else True
    )
    collate_max_length = args.fixed_padding_length or args.max_length

    class CandidateDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> int:
            return index

    def collate(batch: list[int]) -> dict[str, Any]:
        texts: list[str] = []
        state_texts: list[str] = []
        group_lengths: list[int] = []
        teacher_probabilities: list[float] = []
        teacher_policy_scores: list[float] = []
        teacher_q_values: list[float] = []
        teacher_core_q_values: list[float] = []
        teacher_acceptable_masks: list[list[bool]] = []
        candidate_families: list[int] = []
        candidate_memory_operations: list[int] = []
        candidate_response_costs: list[float] = []
        outcome_targets: dict[str, list[float]] = {
            name: [] for name in OUTCOME_HEAD_NAMES
        }
        memory_gain_masks: list[bool] = []
        target_indices: list[int] = []
        target_families: list[int] = []
        outcome_target_families: list[int] = []
        outcome_state_weights: list[float] = []
        target_support_flags: list[list[bool]] = []
        negative_indices: list[list[int]] = []
        trajectory_negative_indices: list[list[int]] = []
        probe_chain_states: list[bool] = []
        probe_grounded_masks: list[list[bool]] = []
        branch_targets: list[int] = []
        phase_targets: list[int] = []
        memory_operation_targets: list[int] = []
        budget_targets: list[tuple[float, ...]] = []
        budget_masks: list[tuple[bool, ...]] = []
        candidate_embeddings = []
        state_embeddings = []
        for record_index in batch:
            row = records[record_index]
            candidates = [CandidateOption.from_record(item) for item in row["candidates"]]
            observation = dict(row["public_observation"])
            encode_memory_tiers = row_memory_tier_encoding(row)
            if cached_candidates is not None and cached_states is not None:
                candidate_embeddings.append(cached_candidates[record_index])
                state_embeddings.append(cached_states[record_index])
            else:
                texts.extend(
                    candidate_pair_text(
                        observation,
                        item,
                        memory_tier_encoding=encode_memory_tiers,
                    )
                    for item in candidates
                )
                state_texts.append(
                    public_state_text(
                        observation, memory_tier_encoding=encode_memory_tiers
                    )
                )
            group_lengths.append(len(candidates))
            probabilities = list(map(float, row["teacher_probabilities"]))
            if len(probabilities) != len(candidates):
                raise RuntimeError("teacher probability count mismatch")
            teacher_probabilities.extend(probabilities)
            policy_scores = list(map(float, row.get("teacher_policy_scores") or []))
            if len(policy_scores) != len(candidates):
                raise RuntimeError("teacher policy score count mismatch")
            teacher_policy_scores.extend(policy_scores)
            q_values = list(map(float, row.get("teacher_q_values") or []))
            core_q_values = list(map(float, row.get("teacher_core_q_values") or []))
            acceptable = list(map(bool, row.get("teacher_acceptable_mask") or []))
            if not (
                len(q_values)
                == len(core_q_values)
                == len(acceptable)
                == len(candidates)
            ):
                raise RuntimeError("V9 Teacher Q/core/acceptable count mismatch")
            teacher_q_values.extend(q_values)
            teacher_core_q_values.extend(core_q_values)
            teacher_acceptable_masks.append(acceptable)
            candidate_families.extend(family_index[item.action_family] for item in candidates)
            candidate_memory_operations.extend(
                MEMORY_OPERATION_NAMES.index(memory_operation_name(item))
                for item in candidates
            )
            candidate_response_costs.extend(
                candidate_response_cost(observation, item.compiled_packet)
                for item in candidates
            )
            outcomes = list(row.get("outcome_targets") or [])
            if len(outcomes) != len(candidates):
                raise RuntimeError(
                    "outcome-grounded training requires one simulator outcome per candidate"
                )
            for outcome in outcomes:
                for name in OUTCOME_HEAD_NAMES:
                    outcome_targets[name].append(float(outcome[name]))
                memory_gain_masks.append(bool(outcome.get("memory_gain_available", False)))
            ids = [item.candidate_id for item in candidates]
            target_indices.append(ids.index(str(row["target_candidate_id"])))
            has_causal_target = bool(
                row.get("causal_target_candidate_id")
                and row.get("causal_target_family")
            )
            use_causal_target = needs_causal_chain and (
                has_causal_target or args.objective != "branched_defense"
            )
            if (
                args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                and str(row.get("task_id", "")) in {"T1", "T2"}
            ):
                use_causal_target = False
            if use_causal_target:
                causal_candidate_id = str(row.get("causal_target_candidate_id", ""))
                causal_family = str(row.get("causal_target_family", ""))
                if causal_candidate_id not in ids or causal_family not in family_index:
                    raise RuntimeError("causal-chain record lacks an admitted target")
                target_indices[-1] = ids.index(causal_candidate_id)
            target_option = candidates[target_indices[-1]]
            if args.objective == "branched_defense":
                branch = derive_branch_supervision(
                    row,
                    target_option,
                    architecture=args.architecture,
                )
                branch_targets.append(branch.branch_index)
                phase_targets.append(branch.phase_index)
                memory_operation_targets.append(branch.memory_operation_index)
                budget_targets.append(branch.budget_targets)
                budget_masks.append(branch.budget_mask)
            target_families.append(
                family_index[effective_target_family(row)]
            )
            outcome_target_families.append(
                family_index[outcome_target_family_names[record_index]]
            )
            outcome_state_weights.append(outcome_record_weights[record_index])
            target_support_flags.append(
                [bool(target_option.action_flags.to_dict()[name]) for name in SUPPORT_FLAGS]
            )
            negative_indices.append(
                [
                    ids.index(str(item))
                    for item in row.get("hard_negative_candidate_ids", [])
                    if str(item) in ids
                    and ids.index(str(item)) != target_indices[-1]
                ]
            )
            trajectory_negative_indices.append(
                [
                    ids.index(str(item))
                    for item in row.get("trajectory_negative_candidate_ids", [])
                    if str(item) in ids
                    and ids.index(str(item)) != target_indices[-1]
                ]
            )
            probe_chain = row.get("probe_chain_target") or {}
            probe_chain_states.append(
                bool(probe_chain.get("is_probe_followup_state", False))
            )
            probe_evidence_ids = {
                str(item)
                for item in (
                    probe_chain.get("probe_evidence_ids")
                    or [probe_chain.get("probe_evidence_id")]
                )
                if item
            }
            probe_grounded_masks.append(
                [
                    bool(probe_evidence_ids & set(item.referenced_ids))
                    for item in candidates
                ]
            )
        model_inputs: dict[str, Any]
        if candidate_embeddings:
            model_inputs = {
                "candidate_embeddings": torch.cat(candidate_embeddings, dim=0),
                "state_embeddings": torch.stack(state_embeddings, dim=0),
            }
        else:
            encoded = tokenizer(
                texts,
                padding=padding,
                truncation=False,
                max_length=collate_max_length,
                add_special_tokens=True,
                return_tensors="pt",
            )
            state_encoded = tokenizer(
                state_texts,
                padding=padding,
                truncation=False,
                max_length=collate_max_length,
                add_special_tokens=True,
                return_tensors="pt",
            )
            model_inputs = {
                **encoded,
                "state_input_ids": state_encoded["input_ids"],
                "state_attention_mask": state_encoded["attention_mask"],
            }
        return {
            **model_inputs,
            "group_lengths": torch.tensor(group_lengths, dtype=torch.long),
            "teacher_probabilities": torch.tensor(teacher_probabilities, dtype=torch.float32),
            "teacher_policy_scores": torch.tensor(
                teacher_policy_scores, dtype=torch.float32
            ),
            "teacher_q_values": torch.tensor(teacher_q_values, dtype=torch.float32),
            "teacher_core_q_values": torch.tensor(
                teacher_core_q_values, dtype=torch.float32
            ),
            "teacher_acceptable_masks": teacher_acceptable_masks,
            "candidate_families": torch.tensor(candidate_families, dtype=torch.long),
            "candidate_memory_operations": torch.tensor(
                candidate_memory_operations, dtype=torch.long
            ),
            "candidate_response_costs": torch.tensor(
                candidate_response_costs, dtype=torch.float32
            ),
            **{
                f"outcome_{name}": torch.tensor(values, dtype=torch.float32)
                for name, values in outcome_targets.items()
            },
            "memory_gain_masks": torch.tensor(memory_gain_masks, dtype=torch.bool),
            "target_indices": torch.tensor(target_indices, dtype=torch.long),
            "target_families": torch.tensor(target_families, dtype=torch.long),
            "outcome_target_families": torch.tensor(
                outcome_target_families, dtype=torch.long
            ),
            "outcome_state_weights": torch.tensor(
                outcome_state_weights, dtype=torch.float32
            ),
            "target_support_flags": torch.tensor(
                target_support_flags, dtype=torch.float32
            ),
            "negative_indices": negative_indices,
            "trajectory_negative_indices": trajectory_negative_indices,
            "probe_chain_states": probe_chain_states,
            "probe_grounded_masks": probe_grounded_masks,
            **(
                {
                    "branch_targets": torch.tensor(branch_targets, dtype=torch.long),
                    "phase_targets": torch.tensor(phase_targets, dtype=torch.long),
                    "memory_operation_targets": torch.tensor(
                        memory_operation_targets, dtype=torch.long
                    ),
                    "budget_targets": torch.tensor(
                        budget_targets, dtype=torch.float32
                    ),
                    "budget_masks": torch.tensor(budget_masks, dtype=torch.bool),
                }
                if args.objective == "branched_defense"
                else {}
            ),
        }

    class RankerModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.heads = heads

        def _score_chunks(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            outputs: dict[str, list[Any]] = {}
            chunk_size = max(1, int(args.candidate_forward_chunk_size))
            for offset in range(0, int(input_ids.shape[0]), chunk_size):
                chunk = score_all_encoded(
                    self.backbone,
                    self.heads,
                    input_ids=input_ids[offset : offset + chunk_size],
                    attention_mask=attention_mask[offset : offset + chunk_size],
                )
                for name, value in chunk.items():
                    outputs.setdefault(name, []).append(value)
            return {name: torch.cat(values, dim=0) for name, values in outputs.items()}

        def forward(
            self,
            input_ids: Any = None,
            attention_mask: Any = None,
            state_input_ids: Any = None,
            state_attention_mask: Any = None,
            candidate_embeddings: Any = None,
            state_embeddings: Any = None,
            adapter_branch_targets: Any = None,
        ) -> dict[str, Any]:
            if args.separate_local_long_adapters:
                if adapter_branch_targets is None:
                    raise RuntimeError("dual-adapter training requires branch targets")
                active = sorted(
                    {int(value) for value in adapter_branch_targets.detach().cpu().tolist()}
                )
                if len(active) != 1:
                    raise RuntimeError(
                        "one per-device batch cannot mix local and long adapters"
                    )
                activate_ranker_adapter(
                    self.backbone,
                    BRANCH_NAMES[active[0]],
                )
            if candidate_embeddings is not None:
                if state_embeddings is None:
                    raise RuntimeError("cached candidates require cached states")
                candidates = score_all_pooled(self.heads, candidate_embeddings)
                states = score_all_pooled(self.heads, state_embeddings)
            else:
                candidates = self._score_chunks(input_ids, attention_mask)
                states = self._score_chunks(state_input_ids, state_attention_mask)
            return {
                **candidates,
                **{f"state_{name}": value for name, value in states.items()},
            }

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
            kwargs = dict(gradient_checkpointing_kwargs or {})
            self.backbone.gradient_checkpointing_enable(**kwargs)

        def gradient_checkpointing_disable(self) -> None:
            self.backbone.gradient_checkpointing_disable()

    class RankerTrainer(Trainer):
        def __init__(self, *trainer_args: Any, **trainer_kwargs: Any) -> None:
            super().__init__(*trainer_args, **trainer_kwargs)
            self.component_sums: dict[str, float] = {}
            self.component_count = 0

        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            if args.trajectory_ordered_training:
                return SequentialSampler(train_dataset or self.train_dataset)
            return super()._get_train_sampler(train_dataset)

        def create_optimizer(self) -> Any:
            if self.optimizer is not None:
                return self.optimizer
            decay_names = self.get_decay_parameter_names(self.model)
            groups = []
            optimizer_prefixes = [
                ("backbone", lora_learning_rate),
                ("heads.utility", head_learning_rate),
                ("heads.utility_residual", head_learning_rate),
                ("heads.family", head_learning_rate),
                ("heads.family_residual", head_learning_rate),
                ("heads.support", head_learning_rate),
                *[
                    (f"heads.{name}", head_learning_rate)
                    for name in OUTCOME_HEAD_NAMES
                ],
                *[
                    (f"heads.{name}", head_learning_rate)
                    for name in BRANCHED_HEAD_NAMES
                ],
            ]
            for prefix, learning_rate in optimizer_prefixes:
                for decay in (True, False):
                    parameters = [
                        parameter
                        for name, parameter in self.model.named_parameters()
                        if parameter.requires_grad
                        and (name == prefix or name.startswith(f"{prefix}."))
                        and (name in decay_names) == decay
                    ]
                    if parameters:
                        isolated_phase_parameter = bool(
                            args.separate_local_long_adapters
                            and prefix in {"heads.phase_gate", "heads.phase_context"}
                        )
                        groups.append(
                            {
                                "params": parameters,
                                "lr": learning_rate,
                                "weight_decay": (
                                    0.0
                                    if isolated_phase_parameter
                                    else 0.01
                                    if decay
                                    else 0.0
                                ),
                            }
                        )
            assigned = {id(parameter) for group in groups for parameter in group["params"]}
            unassigned = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad and id(parameter) not in assigned
            ]
            if unassigned:
                raise RuntimeError(
                    f"{len(unassigned)} trainable parameters have no optimizer group"
                )
            self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1.0e-8)
            return self.optimizer

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
            teacher_policy_scores = inputs.pop("teacher_policy_scores")
            teacher_q_values = inputs.pop("teacher_q_values")
            teacher_core_q_values = inputs.pop("teacher_core_q_values")
            teacher_acceptable_masks = inputs.pop("teacher_acceptable_masks")
            candidate_families = inputs.pop("candidate_families")
            candidate_memory_operations = inputs.pop(
                "candidate_memory_operations"
            )
            candidate_response_costs = inputs.pop("candidate_response_costs")
            outcomes = {
                name: inputs.pop(f"outcome_{name}")
                for name in OUTCOME_HEAD_NAMES
            }
            memory_gain_masks = inputs.pop("memory_gain_masks")
            target_indices = inputs.pop("target_indices")
            target_families = inputs.pop("target_families")
            outcome_target_families = inputs.pop("outcome_target_families")
            outcome_state_weights = inputs.pop("outcome_state_weights")
            target_support_flags = inputs.pop("target_support_flags")
            negative_indices = inputs.pop("negative_indices")
            trajectory_negative_indices = inputs.pop("trajectory_negative_indices")
            probe_chain_states = inputs.pop("probe_chain_states")
            probe_grounded_masks = inputs.pop("probe_grounded_masks")
            branch_targets = inputs.pop("branch_targets", None)
            phase_targets = inputs.pop("phase_targets", None)
            memory_operation_targets = inputs.pop(
                "memory_operation_targets", None
            )
            budget_targets = inputs.pop("budget_targets", None)
            budget_masks = inputs.pop("budget_masks", None)
            predictions = model(
                **inputs,
                adapter_branch_targets=(
                    branch_targets if args.separate_local_long_adapters else None
                ),
            )
            if args.objective == "branched_defense":
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2:
                    state_routes = functional.one_hot(
                        branch_targets, num_classes=len(BRANCH_NAMES)
                    ).to(dtype=predictions["state_branch_probabilities"].dtype)
                else:
                    state_routes = predictions["state_branch_probabilities"]
                candidate_routes = state_routes.repeat_interleave(
                    group_lengths.to(state_routes.device), dim=0
                )
                if args.separate_local_long_adapters:
                    active = sorted(
                        {int(value) for value in branch_targets.detach().cpu().tolist()}
                    )
                    if len(active) != 1:
                        raise RuntimeError("isolated batch contains multiple branches")
                    candidate_expert = {
                        BRANCH_NAMES.index("local"): (
                            "legacy_local_utility",
                            "legacy_local_family",
                        ),
                        BRANCH_NAMES.index("t3_memory"): (
                            "t3_memory_utility",
                            "t3_memory_family",
                        ),
                        BRANCH_NAMES.index("t4_budget"): (
                            "t4_budget_utility",
                            "t4_budget_family",
                        ),
                    }[active[0]]
                    predictions["utility"] = predictions[candidate_expert[0]]
                    predictions["family"] = predictions[candidate_expert[1]]
                else:
                    predictions.update(
                        route_branched_experts(
                            predictions,
                            candidate_routes,
                            use_legacy_local=(
                                args.architecture
                                == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                            ),
                        )
                    )
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2:
                    state_experts = {
                        name: predictions[f"state_{name}"]
                        for name in (
                            "legacy_local_utility",
                            "legacy_local_family",
                            "local_utility",
                            "local_family",
                            "t3_memory_utility",
                            "t3_memory_family",
                            "t4_budget_utility",
                            "t4_budget_family",
                        )
                    }
                    if args.separate_local_long_adapters:
                        state_expert = {
                            BRANCH_NAMES.index("local"): (
                                "state_legacy_local_utility",
                                "state_legacy_local_family",
                            ),
                            BRANCH_NAMES.index("t3_memory"): (
                                "state_t3_memory_utility",
                                "state_t3_memory_family",
                            ),
                            BRANCH_NAMES.index("t4_budget"): (
                                "state_t4_budget_utility",
                                "state_t4_budget_family",
                            ),
                        }[active[0]]
                        state_routed = {
                            "utility": predictions[state_expert[0]],
                            "family": predictions[state_expert[1]],
                        }
                    else:
                        state_routed = route_branched_experts(
                            state_experts,
                            state_routes,
                            use_legacy_local=True,
                        )
                    predictions["state_utility"] = state_routed["utility"]
                    predictions["state_family"] = state_routed["family"]
            raw_scores = predictions["utility"]
            scores = compose_candidate_utility(predictions, score_composition)
            if (
                args.objective == "branched_defense"
                and args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
            ):
                candidate_branches = branch_targets.repeat_interleave(
                    group_lengths.to(branch_targets.device), dim=0
                )
                state_memory_log_probs = functional.log_softmax(
                    predictions["state_t3_memory_operation"], dim=-1
                ).repeat_interleave(group_lengths.to(branch_targets.device), dim=0)
                memory_compatibility = state_memory_log_probs.gather(
                    1, candidate_memory_operations.unsqueeze(-1)
                ).squeeze(-1)
                t3_mask = candidate_branches == BRANCH_NAMES.index("t3_memory")
                scores = scores + (
                    float(score_composition["t3_memory_operation_weight"])
                    * t3_mask.to(scores.dtype)
                    * memory_compatibility
                )

                budget_logits = predictions["state_t4_budget_state"]
                predicted_remaining_budget = torch.sigmoid(
                    budget_logits[
                        :, BUDGET_TARGET_NAMES.index("remaining_business_budget")
                    ]
                ).repeat_interleave(
                    group_lengths.to(branch_targets.device), dim=0
                )
                ready = functional.logsigmoid(
                    budget_logits[
                        :, BUDGET_TARGET_NAMES.index("mitigation_ready")
                    ]
                )
                not_stopped = functional.logsigmoid(
                    -budget_logits[:, BUDGET_TARGET_NAMES.index("stop_required")]
                )
                budget_compatibility = (ready + not_stopped).repeat_interleave(
                    group_lengths.to(branch_targets.device), dim=0
                )
                t4_mitigation_mask = (
                    candidate_branches == BRANCH_NAMES.index("t4_budget")
                ) & (candidate_families == family_index["mitigation"])
                scores = (
                    scores
                    + float(score_composition["t4_mitigation_ready_weight"])
                    * t4_mitigation_mask.to(scores.dtype)
                    * budget_compatibility
                    - float(score_composition["t4_remaining_budget_weight"])
                    * t4_mitigation_mask.to(scores.dtype)
                    * functional.relu(
                        candidate_response_costs - predicted_remaining_budget
                    )
                )
            family_losses = []
            action_support_losses = []
            best_losses = []
            rank_losses = []
            hard_losses = []
            probe_chain_losses = []
            trajectory_losses = []
            parameter_losses = []
            hierarchical_losses = []
            outcome_family_losses = []
            outcome_candidate_losses = []
            policy_regression_losses = []
            core_regression_losses = []
            soft_family_losses = []
            lipo_losses = []
            conservative_losses = []
            composite_calibration_losses = []
            local_teacher_distillation_losses = []
            local_acceptable_mass_losses = []
            local_hierarchical_losses = []
            local_hard_losses = []
            local_probe_chain_losses = []
            local_trajectory_losses = []
            local_parameter_losses = []
            local_action_support_losses = []
            long_family_losses = []
            long_best_losses = []
            long_hard_losses = []
            long_probe_chain_losses = []
            long_trajectory_losses = []
            long_parameter_losses = []
            long_action_support_losses = []
            offset = 0
            for group_index, raw_length in enumerate(group_lengths.tolist()):
                length = int(raw_length)
                group_scores = scores[offset : offset + length]
                group_probs = teacher_probabilities[offset : offset + length]
                group_policy_scores = teacher_policy_scores[offset : offset + length]
                group_q_values = teacher_q_values[offset : offset + length]
                group_core_q_values = teacher_core_q_values[offset : offset + length]
                group_acceptable = teacher_acceptable_masks[group_index]
                group_families = candidate_families[offset : offset + length]
                target_index = int(target_indices[group_index])
                target_family = int(target_families[group_index])
                outcome_target_family = int(outcome_target_families[group_index])
                is_v2_local = bool(
                    args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                    and int(branch_targets[group_index])
                    == BRANCH_NAMES.index("local")
                )
                state_family_weights = (
                    local_original4k_family_training_weights
                    if is_v2_local
                    else family_training_weights
                )
                state_weight = state_family_weights[target_family] * (
                    args.observe_target_weight
                    if target_family == family_index["observe"]
                    else 1.0
                )
                family_candidate_indices = [
                    index
                    for index in range(length)
                    if int(group_families[index]) == target_family
                ]
                if target_index not in family_candidate_indices:
                    raise RuntimeError("target candidate is outside its declared family")
                group_best_loss = state_weight * functional.cross_entropy(
                    group_scores.unsqueeze(0),
                    target_indices.new_tensor([target_index]),
                )
                best_losses.append(group_best_loss)
                if not is_v2_local:
                    long_best_losses.append(group_best_loss)
                rank_losses.append(
                    state_weight
                    * -(group_probs * functional.log_softmax(group_scores, dim=0)).sum()
                )
                if (
                    args.objective == "branched_defense"
                    and int(branch_targets[group_index])
                    == BRANCH_NAMES.index("local")
                ):
                    local_teacher_distillation_losses.append(
                        state_weight
                        * -(
                            group_probs
                            * functional.log_softmax(group_scores, dim=0)
                        ).sum()
                    )
                    acceptable_indices = [
                        index
                        for index, acceptable in enumerate(group_acceptable)
                        if acceptable
                    ]
                    if acceptable_indices:
                        local_acceptable_mass_losses.append(
                            state_weight
                            * (
                                torch.logsumexp(group_scores, dim=0)
                                - torch.logsumexp(
                                    group_scores[acceptable_indices], dim=0
                                )
                            )
                        )
                group_family_loss = state_weight * functional.cross_entropy(
                    predictions["state_family"][group_index].unsqueeze(0),
                    target_families[group_index].unsqueeze(0),
                )
                family_losses.append(group_family_loss)
                if not is_v2_local:
                    long_family_losses.append(group_family_loss)
                group_support_loss = state_weight * (
                    functional.binary_cross_entropy_with_logits(
                        predictions["state_support"][group_index],
                        target_support_flags[group_index],
                    )
                )
                action_support_losses.append(group_support_loss)
                if is_v2_local:
                    local_action_support_losses.append(group_support_loss)
                else:
                    long_action_support_losses.append(group_support_loss)
                if (
                    needs_hierarchical_distillation
                    or needs_soft_family_distillation
                    or is_v2_local
                ):
                    available_families = sorted(
                        {int(value) for value in group_families.tolist()}
                    )
                    available_family_logits = predictions["state_family"][
                        group_index
                    ][available_families]
                    family_log_probs = functional.log_softmax(
                        available_family_logits, dim=0
                    )
                    family_marginals = group_scores.new_zeros(
                        len(available_families)
                    )
                    candidate_log_probs = (
                        group_scores.new_empty(length)
                        if needs_hierarchical_distillation or is_v2_local
                        else None
                    )
                    for family_position, family in enumerate(available_families):
                        indices = [
                            index
                            for index in range(length)
                            if int(group_families[index]) == family
                        ]
                        if candidate_log_probs is not None:
                            within = functional.log_softmax(
                                group_scores[indices], dim=0
                            )
                            for position, candidate_index in enumerate(indices):
                                candidate_log_probs[candidate_index] = (
                                    family_log_probs[family_position] + within[position]
                                )
                        family_marginals[family_position] = group_probs[indices].sum()
                    if needs_hierarchical_distillation or is_v2_local:
                        assert candidate_log_probs is not None
                        group_hierarchical_loss = (
                            state_weight
                            * -(group_probs * candidate_log_probs).sum()
                        )
                        if is_v2_local:
                            local_hierarchical_losses.append(
                                group_hierarchical_loss
                            )
                        else:
                            hierarchical_losses.append(
                                group_hierarchical_loss
                            )
                    if needs_outcome_alignment:
                        target_composite = group_scores.new_zeros(length)
                        for name, weight in V9_OUTCOME_SCORE_WEIGHTS.items():
                            target_composite = target_composite + float(weight) * outcomes[
                                name
                            ][offset : offset + length]
                        outcome_temperature = 0.25
                        outcome_candidate_probabilities = functional.softmax(
                            target_composite / outcome_temperature, dim=0
                        ).detach()
                        outcome_family_values = torch.stack(
                            [
                                target_composite[
                                    [
                                        index
                                        for index in range(length)
                                        if int(group_families[index]) == family
                                    ]
                                ].amax()
                                for family in available_families
                            ]
                        )
                        outcome_family_probabilities = functional.softmax(
                            outcome_family_values / outcome_temperature, dim=0
                        ).detach()
                        hard_outcome_family = torch.zeros_like(
                            outcome_family_probabilities
                        )
                        hard_outcome_family[
                            available_families.index(outcome_target_family)
                        ] = 1.0
                        blended_family_probabilities = (
                            0.20 * family_marginals.detach()
                            + 0.50 * outcome_family_probabilities
                            + 0.30 * hard_outcome_family
                        )
                        outcome_state_weight = outcome_state_weights[group_index]
                        outcome_family_losses.append(
                            outcome_state_weight
                            * -(
                                blended_family_probabilities * family_log_probs
                            ).sum()
                        )
                        outcome_candidate_losses.append(
                            outcome_state_weight
                            * -(
                                outcome_candidate_probabilities
                                * functional.log_softmax(group_scores, dim=0)
                            ).sum()
                        )
                    if needs_soft_family_distillation:
                        soft_family_losses.append(
                            state_weight
                            * -(family_marginals * family_log_probs).sum()
                        )

                def standardized(values: Any) -> Any:
                    centered = values - values.mean()
                    return centered / values.std(unbiased=False).clamp_min(0.10)

                if needs_regret_distillation:
                    regret_range = torch.maximum(
                        group_policy_scores.amax() - group_policy_scores.amin(),
                        group_core_q_values.amax() - group_core_q_values.amin(),
                    )
                    regret_weight = (1.0 + regret_range.clamp_min(0.0)).clamp_max(
                        3.0
                    )
                    policy_regression_losses.append(
                        state_weight
                        * regret_weight
                        * functional.smooth_l1_loss(
                            standardized(group_scores),
                            standardized(group_policy_scores),
                        )
                    )
                    core_regression_losses.append(
                        state_weight
                        * regret_weight
                        * functional.smooth_l1_loss(
                            standardized(group_scores),
                            standardized(group_core_q_values),
                        )
                    )
                if needs_lipo_lambda:
                    pair_indices = torch.triu_indices(
                        length, length, offset=1, device=group_scores.device
                    )
                    left, right = pair_indices[0], pair_indices[1]
                    teacher_order = group_policy_scores + 0.5 * group_core_q_values
                    gaps = teacher_order[left] - teacher_order[right]
                    acceptable_tensor = torch.tensor(
                        group_acceptable,
                        dtype=torch.bool,
                        device=group_scores.device,
                    )
                    valid_pairs = gaps.abs() > 1.0e-6
                    valid_pairs &= ~(
                        acceptable_tensor[left] & acceptable_tensor[right]
                    )
                    if valid_pairs.any():
                        valid_gaps = gaps[valid_pairs]
                        signed_score_differences = valid_gaps.sign() * (
                            group_scores[left[valid_pairs]]
                            - group_scores[right[valid_pairs]]
                        )
                        gap_weights = (
                            valid_gaps.abs()
                            / valid_gaps.abs().amax().clamp_min(1.0e-6)
                        ).clamp_max(1.0)
                        lipo_losses.append(
                            state_weight
                            * (
                                gap_weights
                                * functional.softplus(-signed_score_differences)
                            ).mean()
                        )
                if score_composition is not None:
                    predicted_composite = group_scores.new_zeros(length)
                    target_composite = group_scores.new_zeros(length)
                    sigmoid_names = {
                        "information_gain",
                        "terminal_attack_mitigation",
                        "business_cost",
                        "overresponse_cost",
                        "final_safe_success",
                    }
                    for name, weight in V9_OUTCOME_SCORE_WEIGHTS.items():
                        predicted = predictions[name][offset : offset + length]
                        if name in sigmoid_names:
                            predicted = torch.sigmoid(predicted)
                        predicted_composite = predicted_composite + weight * predicted
                        target_composite = target_composite + weight * outcomes[name][
                            offset : offset + length
                        ]
                    composite_calibration_losses.append(
                        state_weight
                        * functional.smooth_l1_loss(
                            predicted_composite, target_composite
                        )
                    )
                    conservative_losses.append(
                        state_weight
                        * (
                            torch.logsumexp(predicted_composite, dim=0)
                            - (group_probs * predicted_composite).sum()
                        )
                    )
                positive = group_scores[target_index]
                wrong_parameters = [
                    index for index in family_candidate_indices if index != target_index
                ]
                if wrong_parameters:
                    group_parameter_loss = (
                        state_weight
                        * functional.relu(
                            args.margin
                            - positive
                            + torch.stack(
                                [group_scores[index] for index in wrong_parameters]
                            ).max()
                        )
                    )
                    parameter_losses.append(group_parameter_loss)
                    if is_v2_local:
                        local_parameter_losses.append(group_parameter_loss)
                    else:
                        long_parameter_losses.append(group_parameter_loss)
                if probe_chain_states[group_index]:
                    grounded = probe_grounded_masks[group_index]
                    if grounded[target_index]:
                        ungrounded_indices = [
                            index
                            for index, value in enumerate(grounded)
                            if not value and index in family_candidate_indices
                        ]
                        if ungrounded_indices:
                            strongest_ungrounded = torch.stack(
                                [group_scores[index] for index in ungrounded_indices]
                            ).max()
                            group_probe_chain_loss = functional.relu(
                                args.margin - positive + strongest_ungrounded
                            )
                            probe_chain_losses.append(group_probe_chain_loss)
                            if is_v2_local:
                                local_probe_chain_losses.append(
                                    group_probe_chain_loss
                                )
                            else:
                                long_probe_chain_losses.append(
                                    group_probe_chain_loss
                                )
                for negative in negative_indices[group_index]:
                    policy_gap = max(
                        0.0,
                        float(group_policy_scores[target_index])
                        - float(group_policy_scores[int(negative)]),
                    )
                    gap_weight = min(2.0, max(0.25, policy_gap / max(args.margin, 1.0e-6)))
                    group_hard_loss = (
                        state_weight
                        * gap_weight
                        * functional.relu(
                            args.margin - positive + group_scores[int(negative)]
                        )
                    )
                    hard_losses.append(group_hard_loss)
                    if is_v2_local:
                        local_hard_losses.append(group_hard_loss)
                    else:
                        long_hard_losses.append(group_hard_loss)
                for negative in trajectory_negative_indices[group_index]:
                    group_trajectory_loss = (
                        state_weight
                        * functional.relu(
                            args.margin - positive + group_scores[int(negative)]
                        )
                    )
                    trajectory_losses.append(group_trajectory_loss)
                    if is_v2_local:
                        local_trajectory_losses.append(group_trajectory_loss)
                    else:
                        long_trajectory_losses.append(group_trajectory_loss)
                offset += length
            decision_family_loss = torch.stack(family_losses).mean()
            auxiliary_trainable = args.objective not in {
                "defense",
                "branched_defense",
            }
            candidate_family_loss = (
                functional.cross_entropy(predictions["family"], candidate_families)
                if auxiliary_trainable
                else scores.sum() * 0.0
            )
            family_loss = 0.5 * (decision_family_loss + candidate_family_loss)
            best_loss = torch.stack(best_losses).mean()
            action_support_loss = torch.stack(action_support_losses).mean()
            rank_loss = torch.stack(rank_losses).mean()
            hard_loss = (
                torch.stack(hard_losses).mean() if hard_losses else scores.sum() * 0.0
            )
            probe_chain_loss = (
                torch.stack(probe_chain_losses).mean()
                if probe_chain_losses
                else scores.sum() * 0.0
            )
            trajectory_consistency_loss = (
                torch.stack(trajectory_losses).mean()
                if trajectory_losses
                else scores.sum() * 0.0
            )
            parameter_loss = (
                torch.stack(parameter_losses).mean()
                if parameter_losses
                else scores.sum() * 0.0
            )
            zero = scores.sum() * 0.0

            def mean_or_zero(values: list[Any]) -> Any:
                return torch.stack(values).mean() if values else zero

            hierarchical_loss = mean_or_zero(hierarchical_losses)
            outcome_family_loss = mean_or_zero(outcome_family_losses)
            outcome_candidate_loss = mean_or_zero(outcome_candidate_losses)
            policy_regression_loss = mean_or_zero(policy_regression_losses)
            core_regression_loss = mean_or_zero(core_regression_losses)
            soft_family_loss = mean_or_zero(soft_family_losses)
            lipo_loss = mean_or_zero(lipo_losses)
            conservative_loss = mean_or_zero(conservative_losses)
            composite_calibration_loss = mean_or_zero(
                composite_calibration_losses
            )
            local_teacher_distillation_loss = mean_or_zero(
                local_teacher_distillation_losses
            )
            local_acceptable_mass_loss = mean_or_zero(
                local_acceptable_mass_losses
            )
            local_hierarchical_loss = mean_or_zero(local_hierarchical_losses)
            local_hard_loss = mean_or_zero(local_hard_losses)
            local_probe_chain_loss = mean_or_zero(local_probe_chain_losses)
            local_trajectory_loss = mean_or_zero(local_trajectory_losses)
            local_parameter_loss = mean_or_zero(local_parameter_losses)
            local_action_support_loss = mean_or_zero(
                local_action_support_losses
            )
            long_family_loss = mean_or_zero(long_family_losses)
            long_best_loss = mean_or_zero(long_best_losses)
            long_hard_loss = mean_or_zero(long_hard_losses)
            long_probe_chain_loss = mean_or_zero(long_probe_chain_losses)
            long_trajectory_loss = mean_or_zero(long_trajectory_losses)
            long_parameter_loss = mean_or_zero(long_parameter_losses)
            long_action_support_loss = mean_or_zero(long_action_support_losses)
            binary_outcomes = (
                "terminal_attack_mitigation",
                "overresponse_cost",
                "final_safe_success",
            )
            outcome_components = {
                name: functional.binary_cross_entropy_with_logits(
                    predictions[name], outcomes[name]
                )
                for name in binary_outcomes
            }
            for name in ("information_gain", "business_cost"):
                outcome_components[name] = functional.mse_loss(
                    torch.sigmoid(predictions[name]), outcomes[name]
                )
            outcome_components["trajectory_safe_utility"] = functional.smooth_l1_loss(
                predictions["trajectory_safe_utility"],
                outcomes["trajectory_safe_utility"],
            )
            outcome_components["memory_dependent_utility_gain"] = (
                functional.smooth_l1_loss(
                    predictions["memory_dependent_utility_gain"][memory_gain_masks],
                    outcomes["memory_dependent_utility_gain"][memory_gain_masks],
                )
                if memory_gain_masks.any()
                else predictions["memory_dependent_utility_gain"].sum() * 0.0
            )
            outcome_loss = torch.stack(list(outcome_components.values())).mean()
            local_outcome_loss = zero
            if (
                args.objective == "branched_defense"
                and args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
            ):
                local_candidate_mask = (
                    candidate_branches == BRANCH_NAMES.index("local")
                )
                if local_candidate_mask.any():
                    local_outcome_components = {
                        name: functional.binary_cross_entropy_with_logits(
                            predictions[name][local_candidate_mask],
                            outcomes[name][local_candidate_mask],
                        )
                        for name in binary_outcomes
                    }
                    for name in ("information_gain", "business_cost"):
                        local_outcome_components[name] = functional.mse_loss(
                            torch.sigmoid(predictions[name][local_candidate_mask]),
                            outcomes[name][local_candidate_mask],
                        )
                    local_outcome_components["trajectory_safe_utility"] = (
                        functional.smooth_l1_loss(
                            predictions["trajectory_safe_utility"][
                                local_candidate_mask
                            ],
                            outcomes["trajectory_safe_utility"][local_candidate_mask],
                        )
                    )
                    local_memory_mask = local_candidate_mask & memory_gain_masks
                    local_outcome_components["memory_dependent_utility_gain"] = (
                        functional.smooth_l1_loss(
                            predictions["memory_dependent_utility_gain"][
                                local_memory_mask
                            ],
                            outcomes["memory_dependent_utility_gain"][
                                local_memory_mask
                            ],
                        )
                        if local_memory_mask.any()
                        else zero
                    )
                    local_outcome_loss = torch.stack(
                        list(local_outcome_components.values())
                    ).mean()
            phase_gate_loss = zero
            local_phase_gate_loss = zero
            long_phase_gate_loss = zero
            branch_routing_loss = zero
            expert_action_loss = zero
            memory_operation_loss = zero
            budget_state_loss = zero
            if args.objective == "branched_defense":
                if any(
                    value is None
                    for value in (
                        branch_targets,
                        phase_targets,
                        memory_operation_targets,
                        budget_targets,
                        budget_masks,
                    )
                ):
                    raise RuntimeError("branched objective is missing supervision")
                local_row_mask = branch_targets == BRANCH_NAMES.index("local")
                long_row_mask = ~local_row_mask
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2:
                    def scoped_phase_loss(row_mask: Any) -> Any:
                        losses = []
                        for row_index in row_mask.nonzero(as_tuple=False).flatten().tolist():
                            target = int(phase_targets[row_index])
                            prefix = PHASE_NAMES_V2[target].split(":", 1)[0] + ":"
                            admitted = [
                                index
                                for index, name in enumerate(PHASE_NAMES_V2)
                                if name.startswith(prefix)
                            ]
                            losses.append(
                                functional.cross_entropy(
                                    predictions["state_phase_gate"][
                                        row_index, admitted
                                    ].unsqueeze(0),
                                    phase_targets.new_tensor(
                                        [admitted.index(target)]
                                    ),
                                )
                            )
                        return mean_or_zero(losses)

                    local_phase_gate_loss = scoped_phase_loss(local_row_mask)
                    long_phase_gate_loss = scoped_phase_loss(long_row_mask)
                    phase_gate_loss = (
                        local_phase_gate_loss + long_phase_gate_loss
                        if local_row_mask.any() and long_row_mask.any()
                        else local_phase_gate_loss
                        if local_row_mask.any()
                        else long_phase_gate_loss
                    )
                else:
                    phase_gate_loss = functional.cross_entropy(
                        predictions["state_phase_gate"], phase_targets
                    )
                    if local_row_mask.any():
                        local_phase_gate_loss = functional.cross_entropy(
                            predictions["state_phase_gate"][local_row_mask],
                            phase_targets[local_row_mask],
                        )
                    if long_row_mask.any():
                        long_phase_gate_loss = functional.cross_entropy(
                            predictions["state_phase_gate"][long_row_mask],
                            phase_targets[long_row_mask],
                        )
                routing_mask = (
                    long_row_mask
                    if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                    else torch.ones_like(branch_targets, dtype=torch.bool)
                )
                if routing_mask.any() and args.architecture != SHARED_LOCAL_LONG_ARCHITECTURE_V2:
                    branch_routing_loss = functional.nll_loss(
                        predictions["state_branch_probabilities"][routing_mask]
                        .clamp_min(1.0e-8)
                        .log(),
                        branch_targets[routing_mask],
                    )
                expert_losses = []
                expert_head_names = {
                    BRANCH_NAMES.index("local"): "state_local_family",
                    BRANCH_NAMES.index("t3_memory"): "state_t3_memory_family",
                    BRANCH_NAMES.index("t4_budget"): "state_t4_budget_family",
                }
                for branch_index, prediction_name in expert_head_names.items():
                    if (
                        args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                        and branch_index == BRANCH_NAMES.index("local")
                    ):
                        continue
                    mask = branch_targets == branch_index
                    if mask.any():
                        expert_losses.append(
                            functional.cross_entropy(
                                predictions[prediction_name][mask],
                                target_families[mask],
                            )
                        )
                expert_action_loss = mean_or_zero(expert_losses)
                memory_mask = memory_operation_targets >= 0
                if memory_mask.any():
                    memory_operation_loss = functional.cross_entropy(
                        predictions["state_t3_memory_operation"][memory_mask],
                        memory_operation_targets[memory_mask],
                    )
                budget_row_mask = budget_masks.any(dim=-1)
                if budget_row_mask.any():
                    budget_predictions = predictions["state_t4_budget_state"][
                        budget_row_mask
                    ]
                    budget_labels = budget_targets[budget_row_mask]
                    continuous_loss = functional.smooth_l1_loss(
                        torch.sigmoid(budget_predictions[:, :2]),
                        budget_labels[:, :2],
                    )
                    binary_loss = functional.binary_cross_entropy_with_logits(
                        budget_predictions[:, 2:], budget_labels[:, 2:]
                    )
                    budget_state_loss = 0.5 * (continuous_loss + binary_loss)
            weights = {
                "family": (1.0, 0.1, 0.0),
                "listwise": (0.3, 1.0, 0.0),
                "joint": (0.3, 1.0, 0.3),
                "preference": (0.1, 0.1, 1.0),
                "defense": (0.0, 0.0, 0.0),
                "branched_defense": (0.0, 0.0, 0.0),
            }[args.objective]
            if args.objective == "branched_defense":
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2:
                    local_original_4k_loss = (
                        local_hierarchical_loss
                        + 0.5 * local_hard_loss
                        + 0.5 * local_probe_chain_loss
                        + 0.5 * local_trajectory_loss
                        + local_parameter_loss
                        + local_action_support_loss
                        + 0.25 * local_outcome_loss
                    )
                    long_horizon_loss = (
                        2.0 * long_family_loss
                        + long_best_loss
                        + long_parameter_loss
                        + 0.5 * long_hard_loss
                        + 0.5 * long_probe_chain_loss
                        + 0.5 * long_trajectory_loss
                        + long_phase_gate_loss
                        + expert_action_loss
                        + memory_operation_loss
                        + budget_state_loss
                    )
                    loss = local_original_4k_loss + long_horizon_loss
                else:
                    loss = (
                        2.0 * decision_family_loss
                        + best_loss
                        + parameter_loss
                        + action_support_loss
                        + 0.25 * outcome_loss
                        + phase_gate_loss
                        + 0.5 * branch_routing_loss
                        + expert_action_loss
                        + memory_operation_loss
                        + budget_state_loss
                        + args.local_teacher_distillation_weight
                        * local_teacher_distillation_loss
                        + args.local_acceptable_mass_weight
                        * local_acceptable_mass_loss
                    )
            elif args.objective == "defense":
                common_margin = (
                    0.5 * hard_loss
                    + 0.5 * probe_chain_loss
                    + 0.5 * trajectory_consistency_loss
                    + parameter_loss
                    + action_support_loss
                    + 0.25 * outcome_loss
                )
                if args.v9_arm == "hierarchical_distill":
                    loss = hierarchical_loss + common_margin
                elif args.v9_arm == "hierarchical_outcome":
                    loss = (
                        0.25 * hierarchical_loss
                        + outcome_family_loss
                        + 0.75 * outcome_candidate_loss
                        + common_margin
                        + 0.25 * best_loss
                        + 0.25 * rank_loss
                        + 0.5 * composite_calibration_loss
                        + 0.25 * conservative_loss
                    )
                elif args.v9_arm == "regret_distill":
                    loss = (
                        best_loss
                        + 0.5 * rank_loss
                        + common_margin
                        + soft_family_loss
                        + policy_regression_loss
                        + 0.75 * core_regression_loss
                    )
                elif args.v9_arm == "lipo_lambda":
                    loss = (
                        best_loss
                        + 0.5 * rank_loss
                        + common_margin
                        + decision_family_loss
                        + lipo_loss
                    )
                elif args.v9_arm == "causal_conservative":
                    loss = (
                        best_loss
                        + 0.5 * rank_loss
                        + common_margin
                        + decision_family_loss
                        + 0.5 * composite_calibration_loss
                        + 0.25 * conservative_loss
                    )
                elif args.v9_arm == "integrated_hrrc":
                    loss = (
                        hierarchical_loss
                        + common_margin
                        + 0.75 * policy_regression_loss
                        + 0.5 * core_regression_loss
                        + 0.5 * lipo_loss
                        + 0.5 * composite_calibration_loss
                        + 0.25 * conservative_loss
                    )
                elif args.v9_arm == "causal_chain":
                    loss = (
                        2.0 * decision_family_loss
                        + best_loss
                        + parameter_loss
                        + action_support_loss
                        + 0.25 * outcome_loss
                        + 0.25 * composite_calibration_loss
                    )
                else:
                    loss = (
                        best_loss
                        + 0.5 * rank_loss
                        + common_margin
                        + decision_family_loss
                    )
            else:
                loss = (
                    weights[0] * family_loss
                    + weights[1] * rank_loss
                    + weights[2] * hard_loss
                    + 0.5 * probe_chain_loss
                    + 0.5 * trajectory_consistency_loss
                    + 0.25 * outcome_loss
                    + 0.5 * decision_family_loss
                    + 0.25 * action_support_loss
                )
            components = {
                "best_ce": best_loss,
                "policy_cross_entropy": rank_loss,
                "hard_negative_margin": hard_loss,
                "probe_chain_margin": probe_chain_loss,
                "trajectory_consistency_margin": trajectory_consistency_loss,
                "target_parameter_margin": parameter_loss,
                "decision_family_ce": decision_family_loss,
                "action_support_bce": action_support_loss,
                "simulator_outcome_prediction": outcome_loss,
                "hierarchical_teacher_distillation": hierarchical_loss,
                "outcome_aligned_family_distillation": outcome_family_loss,
                "outcome_aligned_candidate_distillation": outcome_candidate_loss,
                "policy_score_regression": policy_regression_loss,
                "core_q_regression": core_regression_loss,
                "soft_family_distillation": soft_family_loss,
                "lipo_lambda": lipo_loss,
                "conservative_outcome": conservative_loss,
                "outcome_composite_calibration": composite_calibration_loss,
                "phase_gate_ce": phase_gate_loss,
                "local_phase_gate_ce": local_phase_gate_loss,
                "long_phase_gate_ce": long_phase_gate_loss,
                "branch_routing_nll": branch_routing_loss,
                "expert_action_ce": expert_action_loss,
                "t3_memory_operation_ce": memory_operation_loss,
                "t4_budget_state": budget_state_loss,
                "local_teacher_distillation": local_teacher_distillation_loss,
                "local_acceptable_mass": local_acceptable_mass_loss,
                "local_original4k_hierarchical": local_hierarchical_loss,
                "local_original4k_hard_margin": local_hard_loss,
                "local_original4k_probe_chain": local_probe_chain_loss,
                "local_original4k_trajectory": local_trajectory_loss,
                "local_original4k_parameter": local_parameter_loss,
                "local_original4k_support": local_action_support_loss,
                "local_original4k_outcome": local_outcome_loss,
                "long_family_ce": long_family_loss,
                "long_best_ce": long_best_loss,
                "long_hard_margin": long_hard_loss,
                "long_probe_chain": long_probe_chain_loss,
                "long_trajectory": long_trajectory_loss,
                "long_parameter": long_parameter_loss,
                "long_support": long_action_support_loss,
            }
            self.component_count += 1
            for name, value in components.items():
                self.component_sums[name] = self.component_sums.get(name, 0.0) + float(
                    value.detach().cpu()
                )
            outputs = {
                "scores": scores.detach(),
                "best_loss": best_loss.detach(),
                "family_loss": family_loss.detach(),
                "rank_loss": rank_loss.detach(),
                "hard_loss": hard_loss.detach(),
                "probe_chain_loss": probe_chain_loss.detach(),
                "trajectory_consistency_loss": trajectory_consistency_loss.detach(),
                "action_support_loss": action_support_loss.detach(),
                "outcome_loss": outcome_loss.detach(),
            }
            return (loss, outputs) if return_outputs else loss

    selection_mode = args.selection_mode or (
        "hierarchical_family_then_utility"
        if args.objective == "defense"
        else "candidate_argmax"
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=lora_learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=not args.freeze_backbone,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=args.objective == "branched_defense",
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )
    model = RankerModel()

    class EpochCheckpointCallback(TrainerCallback):
        """Persist evaluation snapshots at epoch boundaries on global rank zero."""

        def __init__(self) -> None:
            self.saved_epochs: set[int] = set()

        def on_epoch_end(
            self,
            training_arguments: Any,
            state: Any,
            control: Any,
            model: Any = None,
            **kwargs: Any,
        ) -> Any:
            del training_arguments, kwargs
            if not args.save_each_epoch or state.epoch is None:
                return control
            epoch_index = int(round(float(state.epoch)))
            if epoch_index < 1 or epoch_index in self.saved_epochs:
                return control
            if abs(float(state.epoch) - epoch_index) > 1.0e-6:
                return control
            if state.is_world_process_zero:
                ranker_model = getattr(model, "module", model)
                checkpoint_dir = (
                    args.output_dir / "epoch_checkpoints" / f"epoch_{epoch_index:02d}"
                )
                if checkpoint_dir.exists():
                    raise FileExistsError(
                        f"refusing to overwrite epoch checkpoint {checkpoint_dir}"
                    )
                adapter_dir = checkpoint_dir / "adapter"
                adapter_dir.parent.mkdir(parents=True, exist_ok=False)
                save_ranker_adapter(ranker_model.backbone, adapter_dir)
                heads_path = checkpoint_dir / "heads.pt"
                torch.save(
                    {
                        key: value.detach().cpu()
                        for key, value in ranker_model.heads.state_dict().items()
                    },
                    heads_path,
                )
                score_head_path = checkpoint_dir / "score_head.pt"
                torch.save(
                    {
                        key: value.detach().cpu()
                        for key, value in ranker_model.heads[
                            "utility"
                        ].state_dict().items()
                    },
                    score_head_path,
                )
                state.save_to_json(str(checkpoint_dir / "trainer_state.json"))
                epoch_manifest = {
                    "schema_version": 3,
                    "kind": "candidate_ranker_checkpoint",
                    "created_at": utc_now(),
                    "status": "epoch_checkpoint_pending_dev_evaluation",
                    "checkpoint_role": "per_epoch_dev_evaluation",
                    "weights_only": True,
                    "continuous_optimizer_state_in_parent_run": True,
                    "epoch_index": epoch_index,
                    "completed_epochs": float(state.epoch),
                    "optimizer_steps": int(state.global_step),
                    "objective": args.objective,
                    "format_version": FORMAT_VERSION,
                    "base_model": model_identity(args.model_path),
                    "adapter_path": str(adapter_dir.resolve()),
                    "adapter_sha256": sha256_tree(adapter_dir),
                    "adapter_strategy": (
                        "separate_local_long"
                        if args.separate_local_long_adapters
                        else "shared"
                    ),
                    "heads_path": str(heads_path.resolve()),
                    "heads_sha256": sha256_file(heads_path),
                    "score_head_path": str(score_head_path.resolve()),
                    "score_head_sha256": sha256_file(score_head_path),
                    "selection_mode": selection_mode,
                    "architecture": args.architecture,
                    "score_composition": score_composition,
                    "memory_tier_encoding": bool(args.memory_tier_encoding),
                    "source_data_manifest": str(args.data_manifest.resolve()),
                    "source_data_manifest_sha256": sha256_file(args.data_manifest),
                    "initialization_manifest_path": (
                        str(args.initialization_manifest.resolve())
                        if args.initialization_manifest
                        else None
                    ),
                    "initialization_manifest_sha256": (
                        sha256_file(args.initialization_manifest)
                        if args.initialization_manifest
                        else None
                    ),
                    "initialization_role": args.initialization_role,
                    "local_head_initialization": local_head_initialization,
                    "local_teacher_distillation_weight": (
                        args.local_teacher_distillation_weight
                    ),
                    "local_acceptable_mass_weight": (
                        args.local_acceptable_mass_weight
                    ),
                    "protocol_manifest": (
                        str(args.protocol_manifest.resolve())
                        if args.protocol_manifest
                        else None
                    ),
                    "protocol_manifest_sha256": (
                        sha256_file(args.protocol_manifest)
                        if args.protocol_manifest
                        else None
                    ),
                    "seed": args.seed,
                }
                atomic_write_json(checkpoint_dir / "manifest.json", epoch_manifest)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            self.saved_epochs.add(epoch_index)
            return control

    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=CandidateDataset(),
        data_collator=collate,
        callbacks=[EpochCheckpointCallback()] if args.save_each_epoch else None,
    )
    result = trainer.train()
    if _rank() == 0:
        epoch_manifest_paths = sorted(
            (args.output_dir / "epoch_checkpoints").glob("epoch_*/manifest.json")
        ) if args.save_each_epoch else []
        if args.save_each_epoch and len(epoch_manifest_paths) != int(args.epochs):
            raise RuntimeError(
                "per-epoch checkpoint count differs from the configured epoch count"
            )
        adapter_dir = args.output_dir / "adapter"
        heads_path = args.output_dir / "heads.pt"
        score_head_path = args.output_dir / "score_head.pt"
        if args.save_each_epoch:
            final_epoch_dir = args.output_dir / "epoch_checkpoints" / f"epoch_{int(args.epochs):02d}"
            shutil.copytree(final_epoch_dir / "adapter", adapter_dir)
            shutil.copy2(final_epoch_dir / "heads.pt", heads_path)
            shutil.copy2(final_epoch_dir / "score_head.pt", score_head_path)
        else:
            adapter_dir.parent.mkdir(parents=True, exist_ok=True)
            save_ranker_adapter(model.backbone, adapter_dir)
            torch.save(
                {
                    key: value.detach().cpu()
                    for key, value in model.heads.state_dict().items()
                },
                heads_path,
            )
            torch.save(
                {
                    key: value.detach().cpu()
                    for key, value in model.heads["utility"].state_dict().items()
                },
                score_head_path,
            )
        output_manifest = {
            "schema_version": 3,
            "kind": "candidate_ranker_checkpoint",
            "created_at": utc_now(),
            "status": "trained_pending_evaluation",
            "objective": args.objective,
            "format_version": FORMAT_VERSION,
            "base_model": model_identity(args.model_path),
            "adapter_path": str(adapter_dir.resolve()),
            "adapter_sha256": sha256_tree(adapter_dir),
            "lora_rank": int(
                json.loads(
                    ranker_adapter_config_path(adapter_dir).read_text(
                        encoding="utf-8"
                    )
                )["r"]
            ),
            "lora_alpha": int(
                json.loads(
                    ranker_adapter_config_path(adapter_dir).read_text(
                        encoding="utf-8"
                    )
                )["lora_alpha"]
            ),
            "adapter_strategy": (
                "separate_local_long"
                if args.separate_local_long_adapters
                else "shared"
            ),
            "score_head_path": str(score_head_path.resolve()),
            "score_head_sha256": sha256_file(score_head_path),
            "heads_path": str(heads_path.resolve()),
            "heads_sha256": sha256_file(heads_path),
            "final_action_head": "family_then_utility",
            "selection_mode": selection_mode,
            "source_data_manifest": str(args.data_manifest.resolve()),
            "source_data_manifest_sha256": sha256_file(args.data_manifest),
            "initialization_adapter_path": (
                str(args.init_adapter.resolve()) if args.init_adapter else None
            ),
            "initialization_adapter_sha256": (
                sha256_tree(args.init_adapter) if args.init_adapter else None
            ),
            "initialization_heads_path": (
                str(args.init_heads.resolve()) if args.init_heads else None
            ),
            "initialization_heads_sha256": (
                sha256_file(args.init_heads) if args.init_heads else None
            ),
            "initialization_manifest_path": (
                str(args.initialization_manifest.resolve())
                if args.initialization_manifest
                else None
            ),
            "initialization_manifest_sha256": (
                sha256_file(args.initialization_manifest)
                if args.initialization_manifest
                else None
            ),
            "initialization_role": args.initialization_role,
            "fair_main_from_d0": args.initialization_role == "d0_fair_main",
            "four_task_joint_from_d0": (
                args.initialization_role == "d0_four_task_joint"
            ),
            "v11_t12_from_d0": (
                args.initialization_role == "d0_v11_t12_native_v2"
            ),
            "local_head_initialization": local_head_initialization,
            "inherits_original_4k_checkpoint": (
                args.initialization_role == "original_4k_warmstart_ablation"
            ),
            "inherits_long_memory_200_checkpoint": False,
            "train_metrics": dict(result.metrics),
            "optimizer_steps": int(trainer.state.global_step),
            "epoch_checkpointing": {
                "enabled": bool(args.save_each_epoch),
                "checkpoint_count": len(epoch_manifest_paths),
                "manifests": [
                    {
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                    }
                    for path in epoch_manifest_paths
                ],
                "dev_evaluation_in_runner": bool(args.save_each_epoch),
            },
            "per_device_train_batch_size": args.per_device_batch,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "distributed_world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_global_batch_size": (
                args.per_device_batch
                * args.gradient_accumulation
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
            "training_examples_seen": (
                int(trainer.state.global_step)
                * args.per_device_batch
                * args.gradient_accumulation
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
            "loss_components": {
                name: value / max(1, trainer.component_count)
                for name, value in sorted(trainer.component_sums.items())
            },
            "input_audit": input_audit,
            "learning_rate": lora_learning_rate,
            "lora_learning_rate": lora_learning_rate,
            "head_learning_rate": head_learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "candidate_forward_chunk_size": args.candidate_forward_chunk_size,
            "padding_strategy": (
                "fixed_max_length"
                if args.fixed_padding_length is not None
                else "dynamic_longest_in_batch"
            ),
            "fixed_padding_length": args.fixed_padding_length,
            "sequences_per_example": (
                int(manifest.get("candidate_count_per_record", 0)) + 1
                if int(manifest.get("candidate_count_per_record", 0)) > 0
                else None
            ),
            "token_capacity_budget": (
                int(trainer.state.global_step)
                * args.per_device_batch
                * args.gradient_accumulation
                * int(os.environ.get("WORLD_SIZE", "1"))
                * (int(manifest.get("candidate_count_per_record", 0)) + 1)
                * args.fixed_padding_length
                if args.fixed_padding_length is not None
                and int(manifest.get("candidate_count_per_record", 0)) > 0
                else None
            ),
            "frozen_embedding_cache": args.cache_frozen_embeddings,
            "embedding_cache_batch_size": args.embedding_cache_batch_size,
            "trajectory_ordered_training": args.trajectory_ordered_training,
            "observe_target_weight": args.observe_target_weight,
            "max_family_balance_weight": args.max_family_balance_weight,
            "local_teacher_distillation_weight": (
                args.local_teacher_distillation_weight
            ),
            "local_acceptable_mass_weight": args.local_acceptable_mass_weight,
            "target_family_training_weights": {
                family: family_training_weights[index]
                for family, index in family_index.items()
            },
            "local_original4k_family_training_weights": {
                family: local_original4k_family_training_weights[index]
                for family, index in family_index.items()
            },
            "outcome_target_family_counts": dict(
                sorted(Counter(outcome_target_family_names).items())
            ),
            "outcome_family_training_weights": {
                family: outcome_family_training_weights[index]
                for family, index in family_index.items()
            },
            "outcome_family_teacher_blend": (
                {
                    "teacher_soft": 0.20,
                    "simulator_outcome_soft": 0.50,
                    "simulator_outcome_hard": 0.30,
                    "temperature": 0.25,
                    "task_family_balanced": True,
                }
                if needs_outcome_alignment
                else None
            ),
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "backbone_frozen": args.freeze_backbone,
            "gradient_isolation": (
                {
                    "frozen_shared_base": True,
                    "local_adapter": "T1/T2_only",
                    "long_adapter": "T3/T4_only",
                    "long_gradient_into_legacy_local_heads": False,
                    "long_gradient_into_local_phase_rows": False,
                }
                if args.separate_local_long_adapters
                else None
            ),
            "auxiliary_target_contract": (
                "phase_memory_budget_public_supervision_v2"
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else "phase_memory_budget_public_supervision_v1"
                if args.objective == "branched_defense"
                else "simulator_outcomes_v1"
            ),
            "causal_target_contract": (
                "original4k_local_plus_v10_long_causal_v2"
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else "v10_generated_causal_target_with_retained_teacher_fallback_v1"
                if args.objective == "branched_defense"
                else "public_task_phase_then_teacher_within_family_v1"
                if needs_causal_chain
                else None
            ),
            "outcome_heads": list(OUTCOME_HEAD_NAMES),
            "residual_heads": list(RESIDUAL_HEAD_NAMES),
            "decision_head_architecture": (
                args.architecture
                if args.objective == "branched_defense"
                else "legacy_affine_plus_zero_init_residual_mlp"
            ),
            "architecture": args.architecture,
            "phase_gate_routing": (
                "deterministic_public_contract_then_long_phase_v2"
                if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else "public_state_shared_across_candidate_set_v1"
                if args.objective == "branched_defense"
                else None
            ),
            "branch_names": (
                list(BRANCH_NAMES) if args.objective == "branched_defense" else None
            ),
            "phase_gate_targets": (
                list(
                    PHASE_NAMES_V2
                    if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                    else PHASE_NAMES
                )
                if args.objective == "branched_defense"
                else None
            ),
            "t3_memory_targets": (
                list(MEMORY_OPERATION_NAMES)
                if args.objective == "branched_defense"
                else None
            ),
            "t4_budget_targets": (
                list(BUDGET_TARGET_NAMES)
                if args.objective == "branched_defense"
                else None
            ),
            "branch_loss_weights": (
                {
                    "phase_gate_ce": (
                        "local=0.25,long=1.0"
                        if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                        else 1.0
                    ),
                    "branch_routing_nll": (
                        0.0
                        if args.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                        else 0.5
                    ),
                    "expert_action_ce": 1.0,
                    "t3_memory_operation_ce": 1.0,
                    "t4_budget_state": 1.0,
                    "local_teacher_distillation": (
                        args.local_teacher_distillation_weight
                    ),
                    "local_acceptable_mass": args.local_acceptable_mass_weight,
                    **(
                        {
                            "local_original4k_hierarchical": 1.0,
                            "local_original4k_hard_margin": 0.5,
                            "local_original4k_probe_chain": 0.5,
                            "local_original4k_trajectory": 0.5,
                            "local_original4k_parameter": 1.0,
                            "local_original4k_support": 1.0,
                            "local_original4k_outcome": 0.25,
                            "local_phase_gate_ce": 0.0,
                            "long_family_ce": 2.0,
                            "long_best_ce": 1.0,
                            "long_hard_margin": 0.5,
                            "long_probe_chain": 0.5,
                            "long_trajectory": 0.5,
                            "long_parameter": 1.0,
                            "long_support": 0.0,
                            "shared_outcome_auxiliary": 0.0,
                        }
                        if args.architecture
                        == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                        else {}
                    ),
                }
                if args.objective == "branched_defense"
                else None
            ),
            "v9_arm": args.v9_arm,
            "score_composition": score_composition,
            "memory_tier_encoding": bool(args.memory_tier_encoding),
            "seed": args.seed,
            "progressive_arm": args.progressive_arm,
            "experiment_variant": args.experiment_variant,
            "protocol_manifest": (
                str(args.protocol_manifest.resolve())
                if args.protocol_manifest is not None
                else None
            ),
            "protocol_manifest_sha256": (
                sha256_file(args.protocol_manifest)
                if args.protocol_manifest is not None
                else None
            ),
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
