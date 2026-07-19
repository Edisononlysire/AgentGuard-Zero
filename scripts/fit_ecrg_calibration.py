#!/usr/bin/env python3
"""Fit and freeze ECRG rule parameters from sealed public-feature traces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    canonical_json,
    load_checkpoint_manifest,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_tree,
    utc_now,
)
from agentguard_zero.inference_contract import (
    FORMAL_VDA_MAX_NEW_TOKENS,
    TRAINED_VDA_PROMPT_CONTRACT,
    require_candidate_quality,
    summarize_candidate_quality,
)
from ecrg_calibration_lib import (
    immutable_safety_invariants,
    resolve_config,
    search_space,
    select_trace_decision,
)


TASK_IDS = ("T1", "T2", "T3", "T4")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _evaluate(
    traces: list[dict[str, Any]],
    resolved_config: dict[str, Any],
) -> dict[str, Any]:
    scenario_utility: list[float] = []
    scenario_business: list[float] = []
    decisions = 0
    hard_violations = 0
    unsupported_high_impact = 0
    evidence_supported = 0
    fallbacks = 0
    empty_sets = 0
    empty_recoveries = 0
    for trace in traces:
        utilities = []
        business = []
        for decision in trace.get("decisions", []) or []:
            selected = select_trace_decision(decision, resolved_config)
            features = selected["features"]
            label = selected["label"]
            reason = str(selected.get("selection_reason", ""))
            decisions += 1
            hard_violations += int(
                not bool(features.get("parse_ok"))
                or not bool(features.get("base_admissible"))
            )
            unsupported_high_impact += int(
                bool(label.get("unsupported_high_impact", False))
            )
            evidence_supported += int(
                bool(label.get("evidence_supported_execution", False))
            )
            is_fallback = reason in {
                "empty_admissible_set",
                "calibrated_safe_probe_trigger",
            }
            fallbacks += int(is_fallback)
            if reason == "empty_admissible_set":
                empty_sets += 1
                empty_recoveries += int(
                    bool(label.get("attack_mitigated", False))
                    or float(label.get("counterfactual_utility", 0.0)) > 0.0
                )
            utilities.append(float(label.get("counterfactual_utility", 0.0)))
            business.append(float(label.get("business_cost", 0.0)))
        scenario_utility.append(_mean(utilities))
        scenario_business.append(_mean(business))
    sorted_utility = sorted(scenario_utility)
    worst_count = max(1, int(math.ceil(0.10 * len(sorted_utility))))
    return {
        "scenario_count": len(traces),
        "decision_count": decisions,
        "hard_violation_rate": hard_violations / max(1, decisions),
        "unsupported_high_impact_rate": unsupported_high_impact / max(1, decisions),
        "evidence_supported_execution_rate": evidence_supported / max(1, decisions),
        "mean_counterfactual_utility": _mean(scenario_utility),
        "worst_10pct_counterfactual_utility": _mean(sorted_utility[:worst_count]),
        "mean_business_cost": _mean(scenario_business),
        "fallback_rate": fallbacks / max(1, decisions),
        "empty_set_count": empty_sets,
        "empty_set_recovery_rate": empty_recoveries / max(1, empty_sets),
    }


def _metric_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic safety first, then utility, robustness, and cost."""

    return (
        -float(metrics["hard_violation_rate"]),
        -float(metrics["unsupported_high_impact_rate"]),
        float(metrics["mean_counterfactual_utility"]),
        float(metrics["worst_10pct_counterfactual_utility"]),
        float(metrics["evidence_supported_execution_rate"]),
        float(metrics["empty_set_recovery_rate"]),
        -float(metrics["mean_business_cost"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifests", nargs=4, required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--vda-manifest", required=True)
    parser.add_argument("--dca-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=20260718)
    parser.add_argument("--trace-scenarios-per-task", type=int, default=200)
    parser.add_argument("--fit-per-task", type=int, default=160)
    parser.add_argument("--select-per-task", type=int, default=40)
    parser.add_argument("--fit-finalists", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = output_dir / "manifest.json"
    if final_manifest_path.exists():
        raise LineageError(f"refusing to replace frozen ECRG manifest: {final_manifest_path}")

    calibration_manifest_path = Path(args.calibration_manifest).resolve()
    calibration_manifest = read_json(calibration_manifest_path)
    if calibration_manifest.get("status") != "sealed" or calibration_manifest.get(
        "task_counts"
    ) != {"T1": 200, "T2": 200, "T3": 200, "T4": 200}:
        raise LineageError("ECRG-Cal is not the sealed balanced 800-scenario set")

    vda_manifest_path = Path(args.vda_manifest).resolve()
    dca_manifest_path = Path(args.dca_manifest).resolve()
    vda_manifest = load_checkpoint_manifest(
        vda_manifest_path,
        role="vda",
        backbone="qwen3.5-4b",
        round_index=3,
    )
    dca_manifest = load_checkpoint_manifest(
        dca_manifest_path,
        role="dca",
        backbone="qwen3.5-4b",
        round_index=3,
    )
    vda_adapter_before = vda_manifest["adapter_sha256"]
    dca_adapter_before = dca_manifest["adapter_sha256"]

    traces = []
    trace_sources = []
    for raw_manifest_path in args.trace_manifests:
        trace_manifest_path = Path(raw_manifest_path).resolve()
        trace_manifest = read_json(trace_manifest_path)
        if (
            trace_manifest.get("status") != "sealed"
            or trace_manifest.get("candidate_count_per_decision") != 6
            or trace_manifest.get("feature_access") != "public_only"
            or trace_manifest.get("label_access") != "hidden_state_offline_only"
            or trace_manifest.get("prompt_contract") != TRAINED_VDA_PROMPT_CONTRACT
            or trace_manifest.get("max_new_tokens") != FORMAL_VDA_MAX_NEW_TOKENS
            or trace_manifest.get("parameter_training") is not False
            or trace_manifest.get("vda_adapter_sha256_before") != vda_adapter_before
            or trace_manifest.get("vda_adapter_sha256_after") != vda_adapter_before
        ):
            raise LineageError(f"invalid ECRG trace manifest: {trace_manifest_path}")
        trace_path = Path(trace_manifest["trace"]).resolve()
        if sha256_file(trace_path) != trace_manifest["trace_sha256"]:
            raise LineageError(f"trace hash mismatch: {trace_path}")
        shard_rows = _read_jsonl(trace_path)
        if len(shard_rows) != args.trace_scenarios_per_task:
            raise LineageError(
                f"trace shard does not contain {args.trace_scenarios_per_task} "
                f"scenarios: {trace_path}"
            )
        declared_quality = trace_manifest.get("candidate_quality", {})
        try:
            require_candidate_quality(
                declared_quality, context=f"ECRG trace {trace_manifest_path}"
            )
        except ValueError as exc:
            raise LineageError(str(exc)) from exc
        recomputed_quality = summarize_candidate_quality(
            (
                [
                    candidate.get("features", {})
                    for candidate in decision.get("candidates", [])
                ]
                for row in shard_rows
                for decision in row.get("decisions", [])
            ),
            expected_candidates_per_decision=6,
        )
        if recomputed_quality != declared_quality:
            raise LineageError(
                f"candidate-quality audit mismatch: {trace_manifest_path}"
            )
        traces.extend(shard_rows)
        trace_sources.append(
            {
                "task_id": trace_manifest["task_id"],
                "manifest": str(trace_manifest_path),
                "manifest_sha256": sha256_file(trace_manifest_path),
                "trace": str(trace_path),
                "trace_sha256": sha256_file(trace_path),
                "scenario_count": len(shard_rows),
                "decision_count": trace_manifest["decision_count"],
            }
        )

    fingerprints = [str(row.get("scenario_fingerprint", "")) for row in traces]
    expected_total = args.trace_scenarios_per_task * len(TASK_IDS)
    if (
        len(traces) != expected_total
        or len(set(fingerprints)) != expected_total
        or "" in fingerprints
    ):
        raise LineageError(
            f"merged ECRG traces are not {expected_total} unique calibration scenarios"
        )
    task_counts = Counter(str(row.get("task_id", "")) for row in traces)
    expected_task_counts = Counter(
        {task_id: args.trace_scenarios_per_task for task_id in TASK_IDS}
    )
    if task_counts != expected_task_counts:
        raise LineageError(f"merged ECRG trace task imbalance: {dict(task_counts)}")
    candidate_quality = summarize_candidate_quality(
        (
            [
                candidate.get("features", {})
                for candidate in decision.get("candidates", [])
            ]
            for row in traces
            for decision in row.get("decisions", [])
        ),
        expected_candidates_per_decision=6,
    )
    try:
        require_candidate_quality(candidate_quality, context="merged ECRG traces")
    except ValueError as exc:
        raise LineageError(str(exc)) from exc

    fit_rows = []
    selection_rows = []
    split = {"fit": {}, "selection": {}}
    for task_id in TASK_IDS:
        task_rows = [row for row in traces if row["task_id"] == task_id]
        task_rows.sort(
            key=lambda row: sha256_bytes(
                f"{args.split_seed}:{row['scenario_fingerprint']}".encode("utf-8")
            )
        )
        fit = task_rows[: args.fit_per_task]
        selection = task_rows[
            args.fit_per_task : args.fit_per_task + args.select_per_task
        ]
        if len(fit) != args.fit_per_task or len(selection) != args.select_per_task:
            raise LineageError(f"ECRG split quota failed for {task_id}")
        fit_rows.extend(fit)
        selection_rows.extend(selection)
        split["fit"][task_id] = [row["scenario_fingerprint"] for row in fit]
        split["selection"][task_id] = [
            row["scenario_fingerprint"] for row in selection
        ]
    split_manifest = {
        "schema_version": 1,
        "kind": "ecrg_calibration_fit_selection_split",
        "created_at": utc_now(),
        "seed": args.split_seed,
        "policy": "per_task_hash_order",
        "fit_count": len(fit_rows),
        "selection_count": len(selection_rows),
        "fit_per_task": args.fit_per_task,
        "selection_per_task": args.select_per_task,
        "fingerprints": split,
    }
    split_path = output_dir / "split_manifest.json"
    atomic_write_json(split_path, split_manifest)

    fit_results = []
    for profile_names in search_space():
        resolved = resolve_config(profile_names)
        fit_results.append(
            {
                "profiles": profile_names,
                "metrics": _evaluate(fit_rows, resolved),
            }
        )
    fit_results.sort(key=lambda row: _metric_key(row["metrics"]), reverse=True)
    finalists = fit_results[: args.fit_finalists]
    selection_results = []
    for finalist in finalists:
        resolved = resolve_config(finalist["profiles"])
        selection_results.append(
            {
                "profiles": finalist["profiles"],
                "fit_metrics": finalist["metrics"],
                "selection_metrics": _evaluate(selection_rows, resolved),
            }
        )
    selection_results.sort(
        key=lambda row: _metric_key(row["selection_metrics"]), reverse=True
    )
    winner = selection_results[0]
    resolved_winner = resolve_config(winner["profiles"])

    search_audit = {
        "schema_version": 1,
        "kind": "ecrg_fixed_small_search_audit",
        "created_at": utc_now(),
        "search_space_size": len(fit_results),
        "fit_finalist_count": len(finalists),
        "selection_policy": (
            "lexicographic: hard violations, unsupported high impact, mean utility, "
            "worst-10% utility, evidence support, empty-set recovery, business cost"
        ),
        "fit_results": fit_results,
        "selection_results": selection_results,
        "winner": winner,
        "hidden_labels_used_for_objective_only": True,
        "ecrg_feature_access": "public_only",
    }
    search_audit_path = output_dir / "search_audit.json"
    atomic_write_json(search_audit_path, search_audit)

    frozen_config = {
        "schema_version": 1,
        "kind": "evidence_constrained_runtime_governor_config",
        "paper_name": "ECRG",
        "code_name": "V5-C",
        "status": "frozen",
        "frozen_at": utc_now(),
        "candidate_count": 6,
        "prompt_contract": TRAINED_VDA_PROMPT_CONTRACT,
        "max_new_tokens": FORMAL_VDA_MAX_NEW_TOKENS,
        "candidate_quality": candidate_quality,
        "parameters": resolved_winner,
        "immutable_safety_invariants": immutable_safety_invariants(),
        "feature_access": "public_evidence_trust_memory_business_only",
        "hidden_state_access": False,
        "oracle_usage": "offline_calibration_labels_only",
        "parameter_training": False,
        "vda_parameter_update": False,
        "dca_parameter_update": False,
        "fit_scenarios": len(fit_rows),
        "selection_scenarios": len(selection_rows),
        "fit_metrics": winner["fit_metrics"],
        "selection_metrics": winner["selection_metrics"],
        "vda_manifest_sha256": sha256_file(vda_manifest_path),
        "vda_adapter_sha256": vda_adapter_before,
        "dca_manifest_sha256": sha256_file(dca_manifest_path),
        "dca_adapter_sha256": dca_adapter_before,
        "calibration_manifest_sha256": sha256_file(calibration_manifest_path),
        "split_manifest_sha256": sha256_file(split_path),
        "search_audit_sha256": sha256_file(search_audit_path),
    }
    config_path = output_dir / "ecrg_config.json"
    atomic_write_json(config_path, frozen_config)

    vda_adapter_after = sha256_tree(vda_manifest["adapter_path"])
    dca_adapter_after = sha256_tree(dca_manifest["adapter_path"])
    if vda_adapter_after != vda_adapter_before or dca_adapter_after != dca_adapter_before:
        raise LineageError("VDA or DCA adapter changed during ECRG calibration")

    manifest = {
        "schema_version": 1,
        "kind": "ecrg_calibration_manifest",
        "status": "frozen",
        "frozen_at": utc_now(),
        "candidate_count": 6,
        "prompt_contract": TRAINED_VDA_PROMPT_CONTRACT,
        "max_new_tokens": FORMAL_VDA_MAX_NEW_TOKENS,
        "candidate_quality": candidate_quality,
        "calibration_scenario_count": len(traces),
        "task_counts": dict(task_counts),
        "fit_scenario_count": len(fit_rows),
        "selection_scenario_count": len(selection_rows),
        "parameter_training": False,
        "vda_frozen": True,
        "dca_frozen": True,
        "feature_access": "public_only",
        "oracle_usage": "offline_labels_only",
        "tmcd_test_used": False,
        "immutable_safety_invariants": immutable_safety_invariants(),
        "calibration_manifest": str(calibration_manifest_path),
        "calibration_manifest_sha256": sha256_file(calibration_manifest_path),
        "vda_manifest": str(vda_manifest_path),
        "vda_manifest_sha256": sha256_file(vda_manifest_path),
        "vda_adapter_sha256_before": vda_adapter_before,
        "vda_adapter_sha256_after": vda_adapter_after,
        "dca_manifest": str(dca_manifest_path),
        "dca_manifest_sha256": sha256_file(dca_manifest_path),
        "dca_adapter_sha256_before": dca_adapter_before,
        "dca_adapter_sha256_after": dca_adapter_after,
        "trace_sources": trace_sources,
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "search_audit": str(search_audit_path),
        "search_audit_sha256": sha256_file(search_audit_path),
        "ecrg_config": str(config_path),
        "ecrg_config_sha256": sha256_file(config_path),
        "search_space_sha256": sha256_bytes(
            canonical_json(search_space()).encode("utf-8")
        ),
        "winner_profiles": winner["profiles"],
        "winner_fit_metrics": winner["fit_metrics"],
        "winner_selection_metrics": winner["selection_metrics"],
    }
    atomic_write_json(final_manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
