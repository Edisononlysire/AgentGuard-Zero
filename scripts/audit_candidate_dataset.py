#!/usr/bin/env python3
"""Audit compiler validity, public references, recall, and duplicates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.compiler import CandidateCompiler
from agentguard_zero.candidate.gates import evaluate_compiler_gate
from agentguard_zero.candidate.generator import CandidateGenerator
from agentguard_zero.candidate.model import candidate_pair_text
from agentguard_zero.candidate.semantic import semantic_digest
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("inf")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = [
        json.loads(line)
        for line in args.candidate_sets.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    compiler = CandidateCompiler()
    generator = CandidateGenerator()
    compiler_valid = 0
    public_reference_valid = 0
    best_recall = 0
    near_recall = 0
    regrets: list[float] = []
    duplicate_count = 0
    candidate_count = 0
    permutation_consistent = 0
    target_by_semantic_state: dict[str, set[str]] = {}
    for row in rows:
        candidates = [CandidateOption.from_record(item) for item in row["candidates"]]
        target_by_semantic_state.setdefault(
            str(row.get("semantic_public_state_digest", row["public_state_digest"])), set()
        ).add(str(row.get("target_semantic_id", row["target_candidate_id"])))
        semantic_ids = [item.semantic_id for item in candidates]
        q_values = dict(zip(semantic_ids, map(float, row["teacher_q_values"])))
        best = str(row["teacher_full_q_best_candidate_id"])
        best_recall += int(best in semantic_ids)
        full_best_q = float(row["teacher_full_q_best_value"])
        selected_best_q = max(q_values.values())
        near_recall += int(full_best_q - selected_best_q <= 0.02 + 1.0e-12)
        regrets.append(full_best_q - selected_best_q)
        summaries = [
            (
                item.action_family,
                item.public_summary,
                tuple(item.referenced_ids),
                semantic_digest(item.compiled_packet),
            )
            for item in candidates
        ]
        duplicate_count += len(summaries) - len(set(summaries))
        candidate_count += len(candidates)
        regenerated = {
            item.semantic_id for item in generator.generate_all(row["public_observation"])
        }
        first = generator._remap_keys(
            row["public_observation"], candidates, permutation_seed=101
        )
        second = generator._remap_keys(
            row["public_observation"], candidates, permutation_seed=202
        )
        first_text = {
            item.semantic_id: candidate_pair_text(row["public_observation"], item)
            for item in first
        }
        second_text = {
            item.semantic_id: candidate_pair_text(row["public_observation"], item)
            for item in second
        }
        permutation_consistent += int(
            first_text == second_text
            and [item.semantic_id for item in first]
            != [item.semantic_id for item in second]
            and {item.candidate_key for item in first}.isdisjoint(
                item.candidate_key for item in second
            )
        )
        for item in candidates:
            try:
                compiler.compile(item.candidate_id, candidates)
                compiler_valid += 1
            except Exception:
                pass
            public_reference_valid += int(item.semantic_id in regenerated)
    states = max(1, len(rows))
    candidates = max(1, candidate_count)
    metrics = {
        "state_count": len(rows),
        "candidate_count": candidate_count,
        "compiler_validity": compiler_valid / candidates,
        "public_reference_validity": public_reference_valid / candidates,
        "teacher_best_candidate_recall": best_recall / states,
        "near_optimal_candidate_recall": near_recall / states,
        "core_regret_p95": percentile(regrets, 0.95),
        "semantic_duplicate_rate": duplicate_count / candidates,
        "candidate_permutation_consistency": permutation_consistent / states,
        "semantic_target_conflict_rate": sum(
            len(targets) > 1 for targets in target_by_semantic_state.values()
        )
        / max(1, len(target_by_semantic_state)),
        "mean_candidate_count": statistics.mean(
            [len(row["candidates"]) for row in rows]
        )
        if rows
        else 0.0,
    }
    verdict = evaluate_compiler_gate(metrics)
    payload = {
        "schema_version": 1,
        "kind": "candidate_dataset_audit",
        "created_at": utc_now(),
        "candidate_sets_sha256": sha256_file(args.candidate_sets),
        "metrics": metrics,
        "verdict": verdict.to_dict(),
        "accepted": verdict.accepted,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if verdict.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
