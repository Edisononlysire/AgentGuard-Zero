#!/usr/bin/env python3
"""Measure held-out candidate ranking accuracy and Teacher utility regret."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.model import (
    FORMAT_VERSION,
    compose_candidate_utility,
    encode_candidate_pairs,
    encode_public_state,
    load_ranker_components,
    route_branched_experts,
    score_all_encoded,
)
from agentguard_zero.candidate.generator import ACTION_FAMILIES
from agentguard_zero.candidate.policy import capability_chain_family
from agentguard_zero.candidate.supervision import policy_regret
from agentguard_zero.candidate.types import ActionFlags, CandidateOption
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ranker-manifest", type=Path, required=True)
    parser.add_argument("--candidate-sets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--score-batch-size", type=int, default=2)
    parser.add_argument(
        "--target-contract",
        choices=("auto", "causal", "teacher"),
        default="auto",
        help=(
            "Choose evaluation labels explicitly for cross-version comparisons; "
            "auto preserves the checkpoint-native contract."
        ),
    )
    args = parser.parse_args()
    cache_root = Path(
        os.environ.get(
            "AGZ_TRITON_CACHE_ROOT",
            f"/tmp/agentguard_zero_triton_{os.environ.get('USER', 'user')}",
        )
    )
    cache = cache_root / f"candidate_offline_eval_{os.getpid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    import torch

    manifest = json.loads(args.ranker_manifest.read_text(encoding="utf-8"))
    uses_v10_causal_targets = (
        args.target_contract == "causal"
        or (
            args.target_contract == "auto"
            and manifest.get("objective") == "branched_defense"
        )
    )
    memory_tier_encoding = bool(manifest.get("memory_tier_encoding", False))
    tokenizer, backbone, heads = load_ranker_components(
        model_path=args.model_path,
        adapter_path=manifest["adapter_path"],
        heads_path=manifest.get("heads_path"),
        score_head_path=None if manifest.get("heads_path") else manifest["score_head_path"],
        trainable=False,
        architecture=str(manifest.get("architecture", "legacy")),
    )
    device = torch.device(args.device)
    backbone.to(device).eval()
    heads.to(device).eval()
    rows = [
        json.loads(line)
        for line in args.candidate_sets.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    correct = 0
    acceptable_correct = 0
    family_correct = 0
    selected_families = Counter()
    regret_sum = 0.0
    raw_regret_sum = 0.0
    actionable_observe = 0
    actionable_count = 0
    traces = []
    core_regret_sum = 0.0
    probe_followup_count = 0
    probe_followup_nonobserve = 0
    by_task: dict[str, defaultdict[str, float]] = {}
    score_variance_sum = 0.0
    target_margin_sum = 0.0
    target_family_totals = Counter()
    target_family_correct = Counter()
    for row in rows:
        candidates = [CandidateOption.from_record(item) for item in row["candidates"]]
        score_chunks = []
        with torch.inference_mode():
            state_encoded = encode_public_state(
                tokenizer,
                row["public_observation"],
                max_length=args.max_length,
                memory_tier_encoding=memory_tier_encoding,
            )
            state_predictions = score_all_encoded(
                backbone,
                heads,
                input_ids=state_encoded["input_ids"].to(device),
                attention_mask=state_encoded["attention_mask"].to(device),
            )
            for offset in range(0, len(candidates), args.score_batch_size):
                encoded = encode_candidate_pairs(
                    tokenizer,
                    row["public_observation"],
                    candidates[offset : offset + args.score_batch_size],
                    max_length=args.max_length,
                    memory_tier_encoding=memory_tier_encoding,
                )
                predictions = score_all_encoded(
                    backbone,
                    heads,
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                )
                if "branch_probabilities" in state_predictions:
                    state_routes = state_predictions["branch_probabilities"].expand(
                        len(candidates[offset : offset + args.score_batch_size]), -1
                    )
                    predictions.update(
                        route_branched_experts(predictions, state_routes)
                    )
                score_chunks.append(
                    compose_candidate_utility(
                        predictions, manifest.get("score_composition")
                    )
                    .detach()
                    .cpu()
                )
        scores = torch.cat(score_chunks)
        candidate_indices = list(range(len(candidates)))
        gated_family = None
        if manifest.get("selection_mode") in {
            "hierarchical_family_then_utility",
            "capability_chain_then_hierarchical",
        }:
            family_logits = state_predictions["family"][0].detach().cpu()
            available = [
                index
                for index, family in enumerate(ACTION_FAMILIES)
                if any(candidate.action_family == family for candidate in candidates)
            ]
            chain_family = (
                capability_chain_family(row["public_observation"])
                if manifest.get("selection_mode")
                == "capability_chain_then_hierarchical"
                else None
            )
            if chain_family and any(
                candidate.action_family == chain_family for candidate in candidates
            ):
                gated_family = chain_family
            else:
                selected_family_index = max(
                    available,
                    key=lambda index: (
                        float(family_logits[index]),
                        -index,
                    ),
                )
                gated_family = ACTION_FAMILIES[selected_family_index]
            candidate_indices = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.action_family == gated_family
            ]
        selected_index = max(
            candidate_indices, key=lambda index: (float(scores[index]), -index)
        )
        selected = candidates[selected_index]
        task_id = str(row.get("task_id", "unknown"))
        task = by_task.setdefault(task_id, defaultdict(float))
        target_id = str(
            row.get("causal_target_candidate_id")
            if uses_v10_causal_targets and row.get("causal_target_candidate_id")
            else row["target_candidate_id"]
        )
        target_family = str(
            row.get("causal_target_family")
            if uses_v10_causal_targets and row.get("causal_target_family")
            else row["target_family"]
        )
        target_index = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate.candidate_id == target_id
        )
        score_variance_sum += float(scores.var(unbiased=False))
        strongest_incorrect = max(
            float(scores[index])
            for index in range(len(candidates))
            if index != target_index
        )
        target_margin_sum += float(scores[target_index]) - strongest_incorrect
        correct += int(selected.candidate_id == target_id)
        family_correct += int(selected.action_family == target_family)
        selected_families[selected.action_family] += 1
        target_family_totals[target_family] += 1
        target_family_correct[target_family] += int(
            selected.candidate_id == target_id
        )
        q_values = list(map(float, row["teacher_q_values"]))
        policy_scores = list(map(float, row.get("teacher_policy_scores") or []))
        if len(policy_scores) != len(candidates):
            raise RuntimeError("evaluation data lacks teacher policy scores")
        acceptable_mask = list(row.get("teacher_acceptable_mask") or [])
        if len(acceptable_mask) != len(candidates):
            raise RuntimeError("evaluation data lacks acceptable candidate mask")
        acceptable_correct += int(bool(acceptable_mask[selected_index]))
        regret = policy_regret(policy_scores, selected_index)
        regret_sum += regret
        raw_regret = max(q_values) - q_values[selected_index]
        raw_regret_sum += raw_regret
        core_q_values = list(map(float, row.get("teacher_core_q_values", q_values)))
        core_regret = max(core_q_values) - core_q_values[selected_index]
        core_regret_sum += core_regret
        actionable = float((row.get("audit") or {}).get("teacher_advantage", 0.0)) > 0.05
        actionable_count += int(actionable)
        actionable_observe += int(actionable and selected.action_flags.observe_only)
        probe_followup = bool(
            (row.get("probe_chain_target") or {}).get("is_probe_followup_state", False)
        )
        probe_followup_count += int(probe_followup)
        probe_followup_nonobserve += int(
            probe_followup and not selected.action_flags.observe_only
        )
        task["record_count"] += 1
        task["top1"] += int(selected.candidate_id == target_id)
        task["acceptable"] += int(bool(acceptable_mask[selected_index]))
        task["family_correct"] += int(
            selected.action_family == target_family
        )
        task["regret_sum"] += regret
        task["raw_regret_sum"] += raw_regret
        task["core_regret_sum"] += core_regret
        task["actionable"] += int(actionable)
        task["actionable_observe"] += int(
            actionable and selected.action_flags.observe_only
        )
        task["probe_followup"] += int(probe_followup)
        task["probe_followup_nonobserve"] += int(
            probe_followup and not selected.action_flags.observe_only
        )
        for name, enabled in selected.action_flags.to_dict().items():
            task[name] += int(enabled)
        traces.append(
            {
                "record_id": row["record_id"],
                "task_id": task_id,
                "selected_candidate_id": selected.candidate_id,
                "selected_semantic_id": selected.semantic_id,
                "selected_family": selected.action_family,
                "gated_family": gated_family,
                "family_logits": {
                    family: float(state_predictions["family"][0, index])
                    for index, family in enumerate(ACTION_FAMILIES)
                },
                "target_candidate_id": target_id,
                "target_semantic_id": candidates[target_index].semantic_id,
                "target_family": target_family,
                "selected_utility_score": float(scores[selected_index]),
                "target_utility_score": float(scores[target_index]),
                "strongest_incorrect_utility_score": strongest_incorrect,
                "utility_score_variance": float(scores.var(unbiased=False)),
                "target_strongest_negative_margin": float(scores[target_index])
                - strongest_incorrect,
                "teacher_regret": regret,
                "teacher_raw_q_regret": raw_regret,
                "teacher_core_regret": core_regret,
                "teacher_actionable": actionable,
                "probe_followup_state": probe_followup,
                "action_flags": selected.action_flags.to_dict(),
            }
        )
    count = max(1, len(rows))

    def task_metrics(counter: defaultdict[str, float]) -> dict[str, Any]:
        task_count = max(1.0, counter["record_count"])
        actionable_total = max(1.0, counter["actionable"])
        followup_total = counter["probe_followup"]
        return {
            "record_count": int(counter["record_count"]),
            "candidate_top1_accuracy": counter["top1"] / task_count,
            "acceptable_candidate_accuracy": counter["acceptable"] / task_count,
            "action_family_accuracy": counter["family_correct"] / task_count,
            "mean_teacher_regret": counter["regret_sum"] / task_count,
            "mean_raw_teacher_q_regret": counter["raw_regret_sum"] / task_count,
            "mean_teacher_core_regret": counter["core_regret_sum"] / task_count,
            "actionable_observe_rate": counter["actionable_observe"]
            / actionable_total,
            "probe_followup_nonobserve_rate": (
                counter["probe_followup_nonobserve"] / followup_total
                if followup_total
                else None
            ),
            **{
                f"{name}_rate": counter[name] / task_count
                for name in ActionFlags.__dataclass_fields__
            },
        }

    metrics = {
        "record_count": len(rows),
        "candidate_top1_accuracy": correct / count,
        "acceptable_candidate_accuracy": acceptable_correct / count,
        "action_family_accuracy": family_correct / count,
        "state_conditioned_family_accuracy": family_correct / count,
        "mean_teacher_regret": regret_sum / count,
        "mean_raw_teacher_q_regret": raw_regret_sum / count,
        "mean_teacher_core_regret": core_regret_sum / count,
        "actionable_observe_rate": actionable_observe / max(1, actionable_count),
        "active_probe_rate": sum(
            int(bool(row["action_flags"]["active_probe"])) for row in traces
        )
        / count,
        "probe_followup_nonobserve_rate": (
            probe_followup_nonobserve / probe_followup_count
            if probe_followup_count
            else None
        ),
        "selected_family_counts": dict(sorted(selected_families.items())),
        "mean_utility_score_variance": score_variance_sum / count,
        "mean_target_strongest_negative_margin": target_margin_sum / count,
        "target_family_accuracy": {
            family: target_family_correct[family] / total
            for family, total in sorted(target_family_totals.items())
        },
        "probe_target_accuracy": (
            target_family_correct["active_probe"]
            / max(1, target_family_totals["active_probe"])
        ),
        "mitigation_target_accuracy": (
            target_family_correct["mitigation"]
            / max(1, target_family_totals["mitigation"])
        ),
        "memory_lifecycle_accuracy": (
            target_family_correct["memory"]
            / max(1, target_family_totals["memory"])
        ),
        "decoding": str(manifest.get("selection_mode", "candidate_argmax")),
        "target_contract": (
            "v10_generated_causal_target_with_retained_teacher_fallback_v1"
            if uses_v10_causal_targets
            else "frozen_teacher_target_v1"
        ),
        "by_task": {
            task_id: task_metrics(counter)
            for task_id, counter in sorted(by_task.items())
        },
    }
    payload = {
        "schema_version": 2,
        "kind": "candidate_ranker_offline_evaluation",
        "created_at": utc_now(),
        "input_format_version": FORMAT_VERSION,
        "candidate_sets_sha256": sha256_file(args.candidate_sets),
        "ranker_manifest_sha256": sha256_file(args.ranker_manifest),
        "metrics": metrics,
        "traces": traces,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
