#!/usr/bin/env python3
"""Filter a fresh DCA_{r+1} pool and create isolated VDA round splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentguard_zero.env.checker import full_check
from agentguard_zero.protocol import (
    FAMILY_TASK_MAP,
    TASK_FAMILY_MAP,
    TMCD_RELEASE_REVISION,
    task_id_from_focus,
)
from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    feedback_fingerprints,
    load_checkpoint_manifest,
    parse_utc,
    read_json,
    scenario_fingerprint,
    sha256_file,
    utc_now,
)
from agentguard_zero.training.vda_dataset import scenario_to_training_row
from agentguard_zero.schemas.scenario_schema_v2 import paired_counterpart_v2, validate_pair_v2
from agentguard_zero.variants import experiment_variant
from generate_level1_frontier import compute_cfc_metrics


TASK_IDS = ("T1", "T2", "T3", "T4")
PILOT_DIFFICULTY_MIX = {
    "easy": 0.50,
    "frontier": 0.40,
    "hard_reachable": 0.10,
}


def _safe_cfc_metrics(scenario: dict[str, Any]) -> dict[str, Any] | None:
    """Treat malformed generated scenarios as rejected candidates, not fatal input."""
    try:
        return compute_cfc_metrics(scenario)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _task_id(record: dict[str, Any], scenario: dict[str, Any]) -> str:
    metadata = scenario.get("metadata", {}) or {}
    family = str(scenario.get("scenario_family", "")).strip()
    family_task_id = FAMILY_TASK_MAP.get(family, "unknown")
    focus_task_id = task_id_from_focus(str(record.get("task_focus", "")))
    metadata_task_id = str(metadata.get("task_id", "")).strip().upper()
    manipulation_family = str(metadata.get("manipulation_family", "")).strip()
    if (
        family_task_id == "unknown"
        or focus_task_id != family_task_id
        or metadata_task_id != family_task_id
        or manipulation_family != TASK_FAMILY_MAP[family_task_id]
    ):
        return "unknown"
    return family_task_id


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _split_task_quotas(split_counts: dict[str, int]) -> dict[str, dict[str, int]]:
    quotas: dict[str, dict[str, int]] = {}
    remainder_offset = 0
    for split, count in split_counts.items():
        base, remainder = divmod(count, len(TASK_IDS))
        quota = {task_id: base for task_id in TASK_IDS}
        for offset in range(remainder):
            quota[TASK_IDS[(remainder_offset + offset) % len(TASK_IDS)]] += 1
        remainder_offset = (remainder_offset + remainder) % len(TASK_IDS)
        # T2 is represented by inseparable betrayal/legitimate-change pairs.
        # Odd split sizes such as the 500-row micro pilot therefore use the
        # closest possible task balance while keeping the T2 quota even.
        if quota["T2"] % 2:
            quota["T2"] -= 1
            recipients = [task_id for task_id in TASK_IDS if task_id != "T2"]
            recipient = min(
                recipients,
                key=lambda task_id: (
                    quota[task_id],
                    -TASK_IDS.index(task_id),
                ),
            )
            quota[recipient] += 1
        if sum(quota.values()) != count or quota["T2"] % 2:
            raise LineageError(f"invalid task quota for {split}: {quota}")
        quotas[split] = quota
    return quotas


def _largest_remainder_totals(
    count: int, ratios: dict[str, float]
) -> dict[str, int]:
    raw = {name: count * float(ratio) for name, ratio in ratios.items()}
    totals = {name: int(value) for name, value in raw.items()}
    remainder = count - sum(totals.values())
    order = sorted(ratios, key=lambda name: (raw[name] - totals[name], name), reverse=True)
    for name in order[:remainder]:
        totals[name] += 1
    return totals


def _pilot_mix_task_quotas(
    task_quotas: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Allocate an exact 50/40/10 train mix with even T2 category counts."""

    total = sum(task_quotas.values())
    category_targets = _largest_remainder_totals(total, PILOT_DIFFICULTY_MIX)
    categories = tuple(PILOT_DIFFICULTY_MIX)
    t2_total = task_quotas["T2"]
    if t2_total % 2:
        raise LineageError("pilot T2 quota must be even")

    feasible_t2: list[tuple[float, tuple[int, ...]]] = []
    for easy in range(0, t2_total + 1, 2):
        for frontier in range(0, t2_total - easy + 1, 2):
            hard = t2_total - easy - frontier
            values = (easy, frontier, hard)
            if any(values[index] > category_targets[name] for index, name in enumerate(categories)):
                continue
            error = sum(
                (values[index] - t2_total * PILOT_DIFFICULTY_MIX[name]) ** 2
                for index, name in enumerate(categories)
            )
            feasible_t2.append((error, values))
    if not feasible_t2:
        raise LineageError("no even T2 allocation satisfies the pilot difficulty mix")
    _, chosen_t2 = min(feasible_t2, key=lambda item: (item[0], item[1]))

    allocation = {
        task_id: {name: 0 for name in categories} for task_id in TASK_IDS
    }
    allocation["T2"] = dict(zip(categories, chosen_t2))
    remaining_rows = {
        task_id: task_quotas[task_id]
        for task_id in TASK_IDS
        if task_id != "T2"
    }
    remaining_categories = {
        name: category_targets[name] - allocation["T2"][name]
        for name in categories
    }
    while sum(remaining_rows.values()):
        candidates = [
            (task_id, name)
            for task_id, rows in remaining_rows.items()
            if rows > 0
            for name, needed in remaining_categories.items()
            if needed > 0
        ]
        if not candidates:
            raise LineageError("pilot difficulty quota allocation became infeasible")
        task_id, name = max(
            candidates,
            key=lambda cell: (
                task_quotas[cell[0]] * PILOT_DIFFICULTY_MIX[cell[1]]
                - allocation[cell[0]][cell[1]],
                remaining_categories[cell[1]],
                -TASK_IDS.index(cell[0]),
            ),
        )
        allocation[task_id][name] += 1
        remaining_rows[task_id] -= 1
        remaining_categories[name] -= 1

    if any(remaining_categories.values()):
        raise LineageError(f"unfilled pilot difficulty quotas: {remaining_categories}")
    if any(sum(allocation[task].values()) != task_quotas[task] for task in TASK_IDS):
        raise LineageError("pilot per-task difficulty quotas do not sum correctly")
    if any(allocation["T2"][name] % 2 for name in categories):
        raise LineageError("pilot T2 difficulty strata must preserve complete pairs")
    return allocation


def _split_stratified(
    items: list[dict[str, Any]],
    split_counts: dict[str, int],
    seed: int,
    *,
    rank_by_frontier: bool = True,
    train_difficulty_mix: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[item["task_id"]].append(item)
    rng = random.Random(seed)
    task_quotas = _split_task_quotas(split_counts)
    split_items: dict[str, list[dict[str, Any]]] = {
        split: [] for split in split_counts
    }
    pilot_train_quotas = (
        _pilot_mix_task_quotas(task_quotas["train"])
        if train_difficulty_mix
        else None
    )

    def unit_score(unit: list[dict[str, Any]]) -> float:
        return max(float(row["cfc"].get("frontier_score", 0.0)) for row in unit)

    def allocate_units(
        units: list[list[dict[str, Any]]],
        quotas: dict[str, int],
        *,
        use_frontier_rank: bool = rank_by_frontier,
    ) -> dict[str, list[list[dict[str, Any]]]]:
        required = sum(quotas.values())
        if len(units) < required:
            raise LineageError(
                f"only {len(units)} eligible units for {required} requested"
            )
        if use_frontier_rank:
            selected = sorted(units, key=unit_score, reverse=True)[:required]
            strata = min(10, required)
            for stratum in range(strata):
                start = stratum * required // strata
                end = (stratum + 1) * required // strata
                bucket = selected[start:end]
                rng.shuffle(bucket)
                selected[start:end] = bucket
        else:
            selected = list(units)
            rng.shuffle(selected)
            selected = selected[:required]

        allocated = {split: [] for split in quotas}
        assigned = {split: 0 for split in quotas}
        tie_order = list(quotas)
        rng.shuffle(tie_order)
        tie_rank = {split: index for index, split in enumerate(tie_order)}
        for position, unit in enumerate(selected, start=1):
            eligible = [
                split for split in quotas if assigned[split] < quotas[split]
            ]
            split = max(
                eligible,
                key=lambda name: (
                    quotas[name] * position / required - assigned[name],
                    -tie_rank[name],
                ),
            )
            allocated[split].append(unit)
            assigned[split] += 1
        if assigned != quotas:
            raise LineageError(
                f"difficulty-stratified allocation mismatch: {assigned} != {quotas}"
            )
        return allocated

    def select_pilot_train_units(
        units: list[list[dict[str, Any]]],
        task_id: str,
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        if pilot_train_quotas is None:
            return [], units
        row_quotas = pilot_train_quotas[task_id]
        divisor = 2 if task_id == "T2" else 1
        unit_quotas = {
            name: count // divisor for name, count in row_quotas.items()
        }
        ordered = sorted(units, key=unit_score)
        boundaries = (0, len(ordered) // 3, 2 * len(ordered) // 3, len(ordered))
        buckets = {
            "easy": ordered[boundaries[0] : boundaries[1]],
            "frontier": ordered[boundaries[1] : boundaries[2]],
            "hard_reachable": ordered[boundaries[2] : boundaries[3]],
        }
        selected: list[list[dict[str, Any]]] = []
        selected_ids: set[int] = set()
        for name in ("easy", "frontier", "hard_reachable"):
            bucket = list(buckets[name])
            needed = unit_quotas[name]
            if len(bucket) < needed:
                raise LineageError(
                    f"{task_id} {name} has {len(bucket)} units for {needed} requested"
                )
            rng.shuffle(bucket)
            chosen = bucket[:needed]
            for unit in chosen:
                selected_ids.add(id(unit))
                for row in unit:
                    row["selection_difficulty_stratum"] = name
            selected.extend(chosen)
        remaining = [unit for unit in units if id(unit) not in selected_ids]
        if sum(len(unit) for unit in selected) != task_quotas["train"][task_id]:
            raise LineageError(f"{task_id} pilot train quota mismatch")
        return selected, remaining

    for task_id in TASK_IDS:
        values = by_task.get(task_id, [])
        if task_id == "T2":
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in values:
                pair_id = str(item["scenario"].get("pair_id", "")).strip()
                if not pair_id:
                    raise LineageError("T2 candidate is missing pair_id")
                grouped[pair_id].append(item)
            invalid_pairs = {
                pair_id: len(pair)
                for pair_id, pair in grouped.items()
                if len(pair) != 2
                or {
                    str(row["scenario"].get("trajectory_type", ""))
                    for row in pair
                }
                != {"betrayal", "legitimate_change"}
            }
            if invalid_pairs:
                raise LineageError(
                    "T2 pair IDs must identify exactly one betrayal/change pair: "
                    f"{dict(list(invalid_pairs.items())[:8])}"
                )
            units = [
                sorted(
                    pair,
                    key=lambda row: str(row["scenario"].get("trajectory_type", "")),
                )
                for pair in grouped.values()
            ]
            unit_quotas: dict[str, int] = {}
            for split in split_counts:
                required = task_quotas[split][task_id]
                if required % 2:
                    raise LineageError(
                        "T2 split quota must be even so paired branches stay together"
                    )
                unit_quotas[split] = required // 2
        else:
            units = [[item] for item in values]
            unit_quotas = {
                split: task_quotas[split][task_id] for split in split_counts
            }

        if train_difficulty_mix:
            selected_train, remaining_units = select_pilot_train_units(units, task_id)
            allocated = {split: [] for split in split_counts}
            allocated["train"] = selected_train
            remaining_quotas = {
                split: count for split, count in unit_quotas.items() if split != "train"
            }
            if remaining_quotas:
                remainder_allocated = allocate_units(
                    remaining_units,
                    remaining_quotas,
                    use_frontier_rank=False,
                )
                allocated.update(remainder_allocated)
        else:
            allocated = allocate_units(units, unit_quotas)
        for split, selected_units in allocated.items():
            split_items[split].extend(
                row for unit in selected_units for row in unit
            )

    for split, selected in split_items.items():
        rng.shuffle(selected)
        actual_task_counts = Counter(item["task_id"] for item in selected)
        if len(selected) != split_counts[split]:
            raise LineageError(
                f"split {split} has {len(selected)} rows, expected {split_counts[split]}"
            )
        if actual_task_counts != Counter(task_quotas[split]):
            raise LineageError(
                f"split {split} task quotas mismatch: "
                f"{dict(actual_task_counts)} != {task_quotas[split]}"
            )
    return split_items


def _task_batched_order(
    items: list[dict[str, Any]], batch_size: int, seed: int
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        return items
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[item["task_id"]].append(item)
    rng = random.Random(seed)
    batches: list[list[dict[str, Any]]] = []
    leftovers: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        values = by_task.get(task_id, [])
        rng.shuffle(values)
        full = len(values) - (len(values) % batch_size)
        batches.extend(values[start : start + batch_size] for start in range(0, full, batch_size))
        leftovers.extend(values[full:])
    rng.shuffle(leftovers)
    batches.extend(
        leftovers[start : start + batch_size]
        for start in range(0, len(leftovers), batch_size)
    )
    rng.shuffle(batches)
    ordered = [item for batch in batches for item in batch]
    if len(ordered) != len(items) or {
        item["scenario_fingerprint"] for item in ordered
    } != {item["scenario_fingerprint"] for item in items}:
        raise LineageError("task-batched VDA ordering changed the selected training set")
    return ordered


def _training_row(
    item: dict[str, Any],
    *,
    split: str,
    dca_manifest_path: Path,
    dca_manifest_sha: str,
    target_round: int,
    candidate_pool_path: Path,
    candidate_pool_sha: str,
    dca_adapter_sha: str,
    base_model_sha: str,
) -> dict[str, Any]:
    row = scenario_to_training_row(item["scenario"], split=split)
    extra = dict(row.get("extra_info", {}) or {})
    extra.update(
        {
            "task_id": item["task_id"],
            "task_focus": item.get("task_focus", ""),
            "scenario_fingerprint": item["scenario_fingerprint"],
            "source_role": "dca",
            "source_dca_round": int(target_round),
            "source_checkpoint_manifest": str(dca_manifest_path),
            "source_checkpoint_manifest_sha256": dca_manifest_sha,
            "source_candidate_pool": str(candidate_pool_path),
            "source_candidate_pool_sha256": candidate_pool_sha,
            "frontier_score": float(item["cfc"].get("frontier_score", 0.0)),
            "difficulty": float(item["cfc"].get("difficulty", 0.0)),
            "selection_difficulty_stratum": str(
                item.get("selection_difficulty_stratum", "not_applicable")
            ),
            "oracle_solvable": bool(item["cfc"].get("oracle_solvable", False)),
            "protocol_version": "tmcd-v2",
            "schema_version": int(item["scenario"].get("schema_version", 4)),
            "generator_checkpoint_hash": dca_manifest_sha,
            "base_model_hash": base_model_sha,
            "adapter_hash": dca_adapter_sha,
            "seed": int(item["scenario"].get("metadata", {}).get("generation_seed", 0)),
            "scenario_family": str(item["scenario"].get("scenario_family", "")),
            "parent_scenario_id": str(
                item["scenario"].get("metadata", {}).get("derived_from_scenario_id", "")
            ),
            "split": split,
            "fingerprint": item["scenario_fingerprint"],
            "creation_time": str(item["scenario"].get("metadata", {}).get("generated_at", "")),
        }
    )
    row["extra_info"] = extra
    row["frontier_score"] = extra["frontier_score"]
    row["difficulty"] = extra["difficulty"]
    row["task_id"] = item["task_id"]
    row["cfc_metrics"] = json.dumps(item["cfc"], ensure_ascii=False, sort_keys=True)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--feedback-log", required=True)
    parser.add_argument("--dca-checkpoint-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--dev-size", type=int, required=True)
    parser.add_argument("--xplay-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--train-batch-size", type=int, default=0)
    parser.add_argument(
        "--selection-policy",
        choices=("formal_top_pool_rank_stratified", "pilot_balanced_50_40_10"),
        default="formal_top_pool_rank_stratified",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = args.train_size + args.dev_size + args.xplay_size
    if min(args.train_size, args.dev_size, args.xplay_size) < 0 or requested <= 0:
        raise SystemExit("split sizes must be non-negative and total size must be positive")

    candidate_path = Path(args.candidate_pool).resolve()
    feedback_path = Path(args.feedback_log).resolve()
    dca_manifest_path = Path(args.dca_checkpoint_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    pool = read_json(candidate_path)
    raw_manifest = read_json(dca_manifest_path)
    target_round = int(raw_manifest.get("round", -1))
    backbone = str(raw_manifest.get("backbone", ""))
    dca_manifest = load_checkpoint_manifest(
        dca_manifest_path,
        role="dca",
        backbone=backbone,
        round_index=target_round,
    )
    dca_manifest_sha = sha256_file(dca_manifest_path)
    variant_name = str(
        dca_manifest.get("training_config", {}).get("experiment_variant", "full")
    )
    variant = experiment_variant(variant_name)
    if pool.get("kind") != "dca_candidate_pool":
        raise LineageError("candidate input is not a dca_candidate_pool")
    if int(pool.get("num_candidates_generated", -1)) != len(
        pool.get("candidates", []) or []
    ):
        raise LineageError("candidate pool count does not match its records")
    if int(pool.get("generation_prompt_version", 0)) <= 0:
        raise LineageError("candidate pool is missing a generation prompt version")
    if int(pool.get("candidate_normalization_version", 0)) <= 0:
        raise LineageError("candidate pool is missing a normalization version")
    if str(pool.get("tmcd_release_revision", "")) != TMCD_RELEASE_REVISION:
        raise LineageError("candidate pool does not match the active TMCD release revision")
    if str(
        (dca_manifest.get("training_config", {}) or {}).get(
            "tmcd_release_revision", ""
        )
    ) != TMCD_RELEASE_REVISION:
        raise LineageError("DCA checkpoint does not match the active TMCD release revision")
    if int(pool.get("max_attempts", 0)) <= 0:
        raise LineageError("candidate pool is missing a positive retry budget")
    if int(pool.get("source_dca_round", -1)) != target_round:
        raise LineageError("candidate pool DCA round does not match checkpoint")
    if pool.get("source_dca_checkpoint_manifest_sha256") != dca_manifest_sha:
        raise LineageError("candidate pool checkpoint hash does not match DCA manifest")
    if str(pool.get("experiment_variant", "full")) != variant_name:
        raise LineageError("candidate pool experiment variant does not match DCA manifest")
    if parse_utc(pool["created_at"]) < parse_utc(dca_manifest["created_at"]):
        raise LineageError("candidate pool predates DCA_{r+1} checkpoint")

    feedback = feedback_fingerprints(feedback_path)
    candidate_pool_sha = sha256_file(candidate_path)
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    excluded = Counter()
    for record in pool.get("candidates", []) or []:
        scenario = record.get("scenario")
        if not record.get("parse_ok", False) or not isinstance(scenario, dict):
            excluded["parse_or_type"] += 1
            continue
        fingerprint = scenario_fingerprint(scenario)
        if fingerprint != str(record.get("scenario_fingerprint", "")):
            excluded["fingerprint_mismatch"] += 1
            continue
        if fingerprint in feedback:
            excluded["dca_feedback_overlap"] += 1
            continue
        if fingerprint in seen or record.get("duplicate", False):
            excluded["duplicate"] += 1
            continue
        metadata = scenario.get("metadata", {}) or {}
        if str(metadata.get("experiment_variant", "full")) != variant_name:
            excluded["wrong_experiment_variant"] += 1
            continue
        if int(metadata.get("source_dca_round", -1)) != target_round:
            excluded["wrong_dca_round"] += 1
            continue
        if metadata.get("source_checkpoint_manifest_sha256") != dca_manifest_sha:
            excluded["wrong_checkpoint"] += 1
            continue
        checks = full_check(scenario)
        if not checks.get("all_ok", False):
            excluded["hard_check"] += 1
            continue
        task_id = _task_id(record, scenario)
        if task_id == "unknown":
            excluded["task_family_mismatch"] += 1
            continue
        cfc = _safe_cfc_metrics(scenario)
        if cfc is None:
            excluded["cfc_exception"] += 1
            continue
        if not cfc.get("oracle_solvable", False) or (
            variant.frontier_filtering
            and float(cfc.get("frontier_score", 0.0)) <= 0.0
        ):
            excluded["not_hard_but_solvable"] += 1
            continue
        seen.add(fingerprint)
        base_item = {
            **record,
            "scenario_fingerprint": fingerprint,
            "task_id": task_id,
            "cfc": cfc,
        }
        if base_item["task_id"] == "T2" and scenario.get("protocol_version") == "tmcd-v2":
            counterpart = paired_counterpart_v2(scenario)
            pair_ok, pair_reason = validate_pair_v2(scenario, counterpart)
            counterpart_checks = full_check(counterpart)
            counterpart_cfc = _safe_cfc_metrics(counterpart)
            if not pair_ok or not counterpart_checks.get("all_ok", False) or counterpart_cfc is None:
                excluded[f"invalid_t2_counterpart:{pair_reason}"] += 1
                continue
            else:
                counterpart_metadata = dict(counterpart.get("metadata", {}) or {})
                counterpart_metadata.update(
                    {
                        "source_dca_round": target_round,
                        "source_checkpoint_manifest_sha256": dca_manifest_sha,
                        "derived_from_scenario_id": scenario.get("scenario_id"),
                    }
                )
                counterpart["metadata"] = counterpart_metadata
                counterpart_fingerprint = scenario_fingerprint(counterpart)
                if counterpart_fingerprint in feedback or counterpart_fingerprint in seen:
                    excluded["t2_counterpart_overlap"] += 1
                    continue
                seen.add(counterpart_fingerprint)
                valid.append(base_item)
                valid.append(
                    {
                        **record,
                        "scenario": counterpart,
                        "scenario_fingerprint": counterpart_fingerprint,
                        "duplicate": False,
                        "task_id": "T2",
                        "cfc": counterpart_cfc,
                        "derived_counterpart": True,
                    }
                )
        else:
            valid.append(base_item)

    if len(valid) < requested:
        raise LineageError(
            f"only {len(valid)} isolated hard-but-solvable candidates for {requested} requested; "
            f"excluded={dict(excluded)}"
        )

    split_counts = {
        "train": args.train_size,
        "dev": args.dev_size,
        "xplay": args.xplay_size,
    }
    eligible_task_counts = Counter(item["task_id"] for item in valid)
    quotas = _split_task_quotas(split_counts)
    required_task_counts = {
        task_id: sum(quota[task_id] for quota in quotas.values())
        for task_id in TASK_IDS
    }
    shortages = {
        task_id: {
            "eligible": int(eligible_task_counts.get(task_id, 0)),
            "required": required,
        }
        for task_id, required in required_task_counts.items()
        if eligible_task_counts.get(task_id, 0) < required
    }
    if shortages:
        raise LineageError(
            "candidate pool cannot satisfy all task quotas before splitting: "
            f"shortages={shortages}, eligible={dict(eligible_task_counts)}"
        )

    split_items = _split_stratified(
        valid,
        split_counts,
        args.seed,
        rank_by_frontier=variant.frontier_filtering,
        train_difficulty_mix=(args.selection_policy == "pilot_balanced_50_40_10"),
    )
    split_items["train"] = _task_batched_order(
        split_items["train"], args.train_batch_size, args.seed + 17
    )
    selected = [item for split in ("train", "dev", "xplay") for item in split_items[split]]
    paths = {
        "train": output_dir / "vda_train" / "train.parquet",
        "dev": output_dir / "vda_dev" / "dev.parquet",
        "xplay": output_dir / "vda_xplay" / "xplay.parquet",
    }
    for split, items in split_items.items():
        rows = [
            _training_row(
                item,
                split=split,
                dca_manifest_path=dca_manifest_path,
                dca_manifest_sha=dca_manifest_sha,
                target_round=target_round,
                candidate_pool_path=candidate_path,
                candidate_pool_sha=candidate_pool_sha,
                dca_adapter_sha=str(dca_manifest.get("adapter_sha256") or ""),
                base_model_sha=str(dca_manifest.get("base_model", {}).get("identity_sha256", "")),
            )
            for item in items
        ]
        _write_parquet(paths[split], rows)

    frontier_path = output_dir / "vda_candidates" / "frontier.json"
    atomic_write_json(
        frontier_path,
        {
            "kind": "vda_frontier",
            "created_at": utc_now(),
            "source_candidate_pool": str(candidate_path),
            "selected": selected,
        },
    )
    split_fingerprints = {
        split: [item["scenario_fingerprint"] for item in items] for split, items in split_items.items()
    }
    all_split_fingerprints = [value for values in split_fingerprints.values() for value in values]
    if len(all_split_fingerprints) != len(set(all_split_fingerprints)):
        raise LineageError("VDA train/dev/xplay splits overlap")
    if feedback.intersection(all_split_fingerprints):
        raise LineageError("DCA feedback leaked into a VDA split")

    manifest_path = output_dir / "vda_pool_manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "vda_pool",
        "created_at": utc_now(),
        "seed": args.seed,
        "backbone": backbone,
        "target_round": target_round,
        "protocol_version": "tmcd-v2",
        "experiment_variant": variant_name,
        "frontier_filtering_enabled": bool(variant.frontier_filtering),
        "selection_policy": (
            args.selection_policy
            if args.selection_policy == "pilot_balanced_50_40_10"
            else (
                "security_cfc_top_pool_difficulty_stratified"
                if variant.frontier_filtering
                else "safety_valid_oracle_solvable_stratified_random"
            )
        ),
        "source_dca_checkpoint_manifest": str(dca_manifest_path),
        "source_dca_checkpoint_manifest_sha256": dca_manifest_sha,
        "source_dca_adapter_sha256": dca_manifest.get("adapter_sha256"),
        "candidate_pool": str(candidate_path),
        "candidate_pool_sha256": candidate_pool_sha,
        "feedback_log": str(feedback_path),
        "feedback_log_sha256": sha256_file(feedback_path),
        "feedback_fingerprint_count": len(feedback),
        "feedback_vda_overlap_count": 0,
        "candidate_count": len(pool.get("candidates", []) or []),
        "candidate_generation_prompt_version": pool.get("generation_prompt_version"),
        "candidate_normalization_version": pool.get("candidate_normalization_version"),
        "tmcd_release_revision": TMCD_RELEASE_REVISION,
        "candidate_generation_max_attempts": pool.get("max_attempts"),
        "eligible_count": len(valid),
        "eligible_task_counts": dict(eligible_task_counts),
        "required_task_counts": required_task_counts,
        "selected_count": len(selected),
        "excluded_counts": dict(excluded),
        "split_counts": {split: len(items) for split, items in split_items.items()},
        "split_task_counts": {
            split: dict(Counter(item["task_id"] for item in items)) for split, items in split_items.items()
        },
        "difficulty_split_policy": (
            "per_task_score_tertiles_train_50_40_10_then_random_holdout"
            if args.selection_policy == "pilot_balanced_50_40_10"
            else "frontier_score_top_pool_then_rank_stratified"
        ),
        "train_difficulty_mix_target": (
            PILOT_DIFFICULTY_MIX
            if args.selection_policy == "pilot_balanced_50_40_10"
            else None
        ),
        "train_difficulty_counts": dict(
            Counter(
                str(item.get("selection_difficulty_stratum", "not_applicable"))
                for item in split_items["train"]
            )
        ),
        "train_task_difficulty_counts": {
            task_id: dict(
                Counter(
                    str(item.get("selection_difficulty_stratum", "not_applicable"))
                    for item in split_items["train"]
                    if item["task_id"] == task_id
                )
            )
            for task_id in TASK_IDS
        },
        "train_ordering": (
            f"task_batched_{args.train_batch_size}"
            if args.train_batch_size > 0
            else "row_shuffled"
        ),
        "split_fingerprints": split_fingerprints,
        "paths": {split: str(path) for split, path in paths.items()} | {"frontier": str(frontier_path)},
        "sha256": {split: sha256_file(path) for split, path in paths.items()}
        | {"frontier": sha256_file(frontier_path)},
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({**manifest, "split_fingerprints": "omitted"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
