#!/usr/bin/env python3
"""Merge Teacher shards into one low-regret, action-balanced SFT dataset.

The public-state Teacher evaluates many legal actions, but its strict argmax can
over-represent Observe even when other action families are nearly tied.  This
merger keeps one target per unique public prompt and assigns a deterministic
category quota using the lowest-regret legal candidate already scored by the
Teacher.  It never introduces a new scenario, candidate, hidden field, or
model-performance label.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.bootstrap_data import (
    ACTION_CATEGORIES,
    audit_bootstrap_records,
)
from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION, RecoveryConfig
from agentguard_zero.recovery.public_teacher import (
    ActionCandidate,
    compact_wire_json,
    enumerate_public_candidates,
)
from agentguard_zero.schemas.action_schema_v4 import parse_action_json_v4
from agentguard_zero.world.public_projector import assert_public


DEFAULT_CATEGORY_RATIOS = {
    "observe": 0.18,
    "passive_verification": 0.12,
    "active_probe": 0.15,
    "trust": 0.12,
    "memory": 0.10,
    "mitigation": 0.33,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def public_context(prompt: str) -> dict[str, Any]:
    marker = "\nCurrent decision instance:"
    if marker not in prompt:
        raise ValueError("Teacher prompt is missing the public-context marker")
    value = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(value, dict):
        raise ValueError("Teacher prompt public context is not an object")
    assert_public(value)
    return value


def _pair_shard_records(
    directory: Path,
    train_filename: str,
    audit_filename: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    train = pd.read_parquet(directory / train_filename).to_dict(orient="records")
    audits = load_jsonl(directory / audit_filename)
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in audits:
        by_id[str(audit.get("record_id", ""))].append(audit)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in train:
        record_id = str(row.get("record_id", ""))
        if not by_id[record_id]:
            raise RuntimeError(f"missing audit for Teacher record {record_id}")
        pairs.append((row, by_id[record_id].pop(0)))
    if any(values for values in by_id.values()):
        raise RuntimeError(f"unpaired Teacher audits in {directory}")
    return pairs


def _aggregate_unique_prompts(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Collapse duplicate public prompts with max-min utility aggregation."""

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for train, audit in pairs:
        grouped[str(train["prompt"])].append((train, audit))

    states: list[dict[str, Any]] = []
    for prompt, copies in grouped.items():
        context = public_context(prompt)
        candidates = enumerate_public_candidates(context["observation"], max_candidates=96)
        candidate_by_id = {item.candidate_id: item for item in candidates}
        q_maps = [dict(audit.get("q_audit", {})) for _, audit in copies]
        core_maps = [dict(audit.get("core_q_audit", {})) for _, audit in copies]
        shared = set(candidate_by_id)
        for values in q_maps:
            shared.intersection_update(values)
        for values in core_maps:
            shared.intersection_update(values)
        if not shared:
            raise RuntimeError("duplicate Teacher prompt has no shared scored candidate")
        # If the same public prompt came from multiple fixed source scenarios,
        # preserve the Teacher's robust semantics by taking the worst value.
        q_audit = {
            key: min(float(values[key]) for values in q_maps) for key in sorted(shared)
        }
        core_q_audit = {
            key: min(float(values[key]) for values in core_maps)
            for key in sorted(shared)
        }
        category_best: dict[str, tuple[float, ActionCandidate]] = {}
        for candidate_id in sorted(shared):
            candidate = candidate_by_id[candidate_id]
            value = q_audit[candidate_id]
            current = category_best.get(candidate.category)
            if current is None or (value, candidate_id) > (
                current[0],
                current[1].candidate_id,
            ):
                category_best[candidate.category] = (value, candidate)
        maximum = max(q_audit.values())
        states.append(
            {
                "prompt": prompt,
                "prompt_hash": stable_hash(prompt),
                "public_state_digest": str(copies[0][0]["public_state_digest"]),
                "copies": copies,
                "q_audit": q_audit,
                "core_q_audit": core_q_audit,
                "category_best": category_best,
                "maximum": maximum,
            }
        )
    return sorted(states, key=lambda item: item["prompt_hash"])


def _quota_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1.0e-9):
        raise ValueError("action category ratios must sum to one")
    exact = {category: total * float(ratios[category]) for category in ACTION_CATEGORIES}
    quotas = {category: int(math.floor(exact[category])) for category in ACTION_CATEGORIES}
    remainder = total - sum(quotas.values())
    order = sorted(
        ACTION_CATEGORIES,
        key=lambda category: (-(exact[category] - quotas[category]), category),
    )
    for category in order[:remainder]:
        quotas[category] += 1
    return quotas


def assign_low_regret_categories(
    states: list[dict[str, Any]],
    quotas: Mapping[str, int],
) -> dict[str, str]:
    """Assign one category per state, filling scarce families first.

    Scarcity ordering prevents common Observe/mitigation candidates from taking
    states that are among the few legal low-regret memory/trust examples.
    Within a category, assignment is strictly lowest Teacher regret first.
    """

    availability = {
        category: sum(category in state["category_best"] for state in states)
        for category in ACTION_CATEGORIES
    }
    for category in ACTION_CATEGORIES:
        if availability[category] < int(quotas[category]):
            raise RuntimeError(
                f"insufficient legal {category} candidates: "
                f"{availability[category]} < {quotas[category]}"
            )
    category_order = sorted(
        ACTION_CATEGORIES,
        key=lambda category: (
            availability[category] / max(1, int(quotas[category])),
            category,
        ),
    )
    unassigned = {state["prompt_hash"]: state for state in states}
    assignments: dict[str, str] = {}
    for category in category_order:
        ranked = sorted(
            (
                state
                for state in unassigned.values()
                if category in state["category_best"]
            ),
            key=lambda state: (
                state["maximum"] - state["category_best"][category][0],
                -state["category_best"][category][0],
                state["prompt_hash"],
            ),
        )
        take = int(quotas[category])
        if len(ranked) < take:
            raise RuntimeError(
                f"quota competition left too few {category} states: {len(ranked)} < {take}"
            )
        for state in ranked[:take]:
            assignments[state["prompt_hash"]] = category
            unassigned.pop(state["prompt_hash"])
    if unassigned or len(assignments) != len(states):
        raise RuntimeError("action-balanced assignment did not cover every public state")
    return assignments


def _materialize(
    states: list[dict[str, Any]], assignments: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    train_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    regrets: list[float] = []
    for state in states:
        category = assignments[state["prompt_hash"]]
        value, candidate = state["category_best"][category]
        target = compact_wire_json(candidate.packet)
        _, valid, reason = parse_action_json_v4(target)
        if not valid:
            raise RuntimeError(f"balanced Teacher target is invalid: {reason}")
        record_id = stable_hash(
            f"balanced-v1:{state['prompt_hash']}:{candidate.candidate_id}"
        )
        train_rows.append(
            {
                "record_id": record_id,
                "messages": [
                    {"role": "user", "content": state["prompt"]},
                    {"role": "assistant", "content": target},
                ],
                "prompt": state["prompt"],
                "target": target,
                "public_state_digest": state["public_state_digest"],
                "action_category": category,
                "source_policy": "low_regret_action_balanced_public_state_teacher",
            }
        )
        source_hashes = sorted(
            {
                str(value)
                for _, audit in state["copies"]
                for value in audit.get("source_scenario_hashes", [])
            }
        )
        original = max(
            state["q_audit"], key=lambda key: (state["q_audit"][key], key)
        )
        regret = float(state["maximum"] - value)
        regrets.append(regret)
        audit_rows.append(
            {
                "record_id": record_id,
                "public_state_digest": state["public_state_digest"],
                "selected_candidate_id": candidate.candidate_id,
                "selected_category": category,
                "robust_value": float(value),
                "observe_value": float(state["category_best"]["observe"][0]),
                "advantage_over_observe": float(
                    value - state["category_best"]["observe"][0]
                ),
                "public_candidate_count": len(state["q_audit"]),
                "admitted_candidate_count": len(state["q_audit"]),
                "world_count": len(source_hashes),
                "search_horizon": 3,
                "q_audit": state["q_audit"],
                "core_q_audit": state["core_q_audit"],
                "hidden_state_in_target": False,
                "source_scenario_hashes": source_hashes,
                "target_sha256": stable_hash(target),
                "model_input_hidden_state": False,
                "model_target_hidden_state": False,
                "hidden_state_usage": "offline_robust_utility_only",
                "original_teacher_selected_candidate_id": original,
                "original_teacher_selected_category": next(
                    item.category
                    for item in enumerate_public_candidates(
                        public_context(state["prompt"])["observation"],
                        max_candidates=96,
                    )
                    if item.candidate_id == original
                ),
                "balanced_assignment_regret": regret,
                "balanced_assignment_policy": "fixed_quota_lowest_teacher_regret_v1",
                "duplicate_public_prompt_count": len(state["copies"]),
            }
        )
    return train_rows, audit_rows, regrets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-scenarios", type=int, required=True)
    parser.add_argument("--min-records", type=int, required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--target-records", type=int, default=3000)
    parser.add_argument("--kind", default="recovery_bootstrap_teacher_dataset")
    parser.add_argument("--train-filename", default="bootstrap_sft.parquet")
    parser.add_argument("--audit-filename", default="teacher_selection_audit.jsonl")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    manifests: list[dict[str, Any]] = []
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for directory in args.shards:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("accepted") is not True:
            raise RuntimeError(f"rejected shard: {directory}")
        manifests.append(manifest)
        pairs.extend(
            _pair_shard_records(
                directory, args.train_filename, args.audit_filename
            )
        )

    expected_shards = set(range(len(manifests)))
    actual_shards = {int(row.get("shard_index", -1)) for row in manifests}
    shard_counts = {int(row.get("shard_count", -1)) for row in manifests}
    if actual_shards != expected_shards or shard_counts != {len(manifests)}:
        raise RuntimeError("incomplete or inconsistent Teacher shard coverage")
    source_hashes = {str(row.get("source_scenarios_sha256")) for row in manifests}
    if len(source_hashes) != 1:
        raise RuntimeError("Teacher shards disagree on the fixed source dataset")

    states = _aggregate_unique_prompts(pairs)
    raw_unique_count = len(states)
    target_count = min(args.target_records, args.max_records, raw_unique_count)
    if target_count < args.min_records:
        raise RuntimeError(
            f"only {target_count} unique public states; need at least {args.min_records}"
        )
    # Sampling uses only a fixed prompt hash and is independent of action score.
    states = states[:target_count]
    quotas = _quota_counts(target_count, DEFAULT_CATEGORY_RATIOS)
    assignments = assign_low_regret_categories(states, quotas)
    train_rows, audit_rows, regrets = _materialize(states, assignments)

    manifest = audit_bootstrap_records(train_rows, audit_rows)
    source_count = sum(int(row.get("source_scenario_count", 0)) for row in manifests)
    record_gate = args.min_records <= len(train_rows) <= args.max_records
    source_gate = source_count == args.expected_source_scenarios
    rank_count = int(manifest.get("teacher_core_rank_correlation_state_count", 0))
    rank_value = manifest.get("teacher_core_rank_correlation_mean")
    cfg = RecoveryConfig()
    rank_gate = (
        rank_count >= 200
        and isinstance(rank_value, (int, float))
        and float(rank_value) > cfg.teacher.core_rank_correlation_min_exclusive
    )
    sorted_regrets = sorted(regrets)
    manifest.update(
        {
            "schema_version": 1,
            "kind": args.kind,
            "protocol_version": RECOVERY_PROTOCOL_VERSION,
            "source_scenarios_sha256": next(iter(source_hashes)),
            "source_scenario_count": source_count,
            "raw_teacher_record_count": len(pairs),
            "raw_unique_public_prompt_count": raw_unique_count,
            "audit_world_count": sum(
                int(row.get("audit_world_count", 0)) for row in manifests
            ),
            "counterfactual_worlds_training_visible": False,
            "human_action_labels": 0,
            "action_balance_policy": {
                "name": "fixed_quota_lowest_teacher_regret_v1",
                "category_ratios": DEFAULT_CATEGORY_RATIOS,
                "category_quotas": quotas,
                "selection": "fixed_prompt_hash_then_lowest_teacher_regret",
                "one_target_per_unique_public_prompt": True,
                "new_candidates_created": False,
                "model_performance_used": False,
            },
            "assignment_regret": {
                "mean": statistics.fmean(regrets),
                "median": statistics.median(regrets),
                "p90": sorted_regrets[min(len(sorted_regrets) - 1, int(0.90 * len(sorted_regrets)))],
                "p95": sorted_regrets[min(len(sorted_regrets) - 1, int(0.95 * len(sorted_regrets)))],
                "maximum": max(regrets),
                "at_most_0_10_ratio": sum(value <= 0.10 + 1.0e-12 for value in regrets)
                / len(regrets),
            },
            "record_count_gate": {
                "minimum": args.min_records,
                "maximum": args.max_records,
                "accepted": record_gate,
            },
            "source_scenario_gate": {
                "expected": args.expected_source_scenarios,
                "actual": source_count,
                "accepted": source_gate,
            },
            "teacher_core_rank_correlation_gate": {
                "minimum_states": 200,
                "minimum_exclusive": cfg.teacher.core_rank_correlation_min_exclusive,
                "actual_states": rank_count,
                "actual": rank_value,
                "accepted": rank_gate,
            },
            "input_shards": {
                str(path): sha256(path / "manifest.json") for path in args.shards
            },
            "recovery_config": cfg.to_dict(),
        }
    )
    manifest["accepted"] = bool(
        manifest.get("accepted") and record_gate and source_gate and rank_gate
    )
    manifest["status"] = "accepted" if manifest["accepted"] else "rejected"
    manifest["next_stage"] = (
        "direct_teacher_sft" if manifest["accepted"] else "stop_and_repair_teacher_data"
    )

    args.output_dir.mkdir(parents=True)
    train = args.output_dir / args.train_filename
    audit = args.output_dir / args.audit_filename
    output_manifest = args.output_dir / "manifest.json"
    pd.DataFrame(train_rows).to_parquet(train, index=False)
    with audit.open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256(path) for path in (train, audit, output_manifest)}
    (args.output_dir / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
