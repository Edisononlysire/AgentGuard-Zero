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
    encode_candidate_pairs,
    load_ranker_components,
    score_encoded,
)
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
    tokenizer, backbone, heads = load_ranker_components(
        model_path=args.model_path,
        adapter_path=manifest["adapter_path"],
        heads_path=manifest.get("heads_path"),
        score_head_path=None if manifest.get("heads_path") else manifest["score_head_path"],
        trainable=False,
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
    family_correct = 0
    selected_families = Counter()
    regret_sum = 0.0
    actionable_observe = 0
    actionable_count = 0
    traces = []
    core_regret_sum = 0.0
    probe_followup_count = 0
    probe_followup_nonobserve = 0
    by_task: dict[str, defaultdict[str, float]] = {}
    for row in rows:
        candidates = [CandidateOption.from_record(item) for item in row["candidates"]]
        score_chunks = []
        with torch.inference_mode():
            for offset in range(0, len(candidates), args.score_batch_size):
                encoded = encode_candidate_pairs(
                    tokenizer,
                    row["public_observation"],
                    candidates[offset : offset + args.score_batch_size],
                    max_length=args.max_length,
                )
                score_chunks.append(
                    score_encoded(
                        backbone,
                        heads,
                        input_ids=encoded["input_ids"].to(device),
                        attention_mask=encoded["attention_mask"].to(device),
                    ).detach().cpu()
                )
        scores = torch.cat(score_chunks)
        selected_index = max(
            range(len(candidates)), key=lambda index: (float(scores[index]), -index)
        )
        selected = candidates[selected_index]
        task_id = str(row.get("task_id", "unknown"))
        task = by_task.setdefault(task_id, defaultdict(float))
        target_id = str(row["target_candidate_id"])
        correct += int(selected.candidate_id == target_id)
        family_correct += int(selected.action_family == str(row["target_family"]))
        selected_families[selected.action_family] += 1
        q_values = list(map(float, row["teacher_q_values"]))
        regret = max(q_values) - q_values[selected_index]
        regret_sum += regret
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
        task["family_correct"] += int(
            selected.action_family == str(row["target_family"])
        )
        task["regret_sum"] += regret
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
                "target_candidate_id": target_id,
                "target_semantic_id": row.get("target_semantic_id"),
                "target_family": row["target_family"],
                "teacher_regret": regret,
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
            "action_family_accuracy": counter["family_correct"] / task_count,
            "mean_teacher_regret": counter["regret_sum"] / task_count,
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
        "action_family_accuracy": family_correct / count,
        "mean_teacher_regret": regret_sum / count,
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
        "by_task": {
            task_id: task_metrics(counter)
            for task_id, counter in sorted(by_task.items())
        },
    }
    payload = {
        "schema_version": 1,
        "kind": "candidate_ranker_offline_evaluation",
        "created_at": utc_now(),
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
