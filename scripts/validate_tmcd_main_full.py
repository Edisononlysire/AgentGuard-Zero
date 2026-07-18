#!/usr/bin/env python3
"""Validate and seal the formal AgentGuard-Zero-Full TMCD evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentguard_zero.training.coevolution import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_tree,
    utc_now,
)
from agentguard_zero.inference_contract import (  # noqa: E402
    FORMAL_VDA_MAX_NEW_TOKENS,
    TRAINED_VDA_PROMPT_CONTRACT,
    require_candidate_quality,
    summarize_candidate_quality,
)
import eval_tmcd_systems as ev  # noqa: E402


EXPECTED_TASK_COUNTS = {"T1": 600, "T2": 600, "T3": 600, "T4": 600}
EXPECTED_HORIZONS = {"T1": 10, "T2": 16, "T3": 14, "T4": 10}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def task_id(value: str) -> str:
    normalized = str(value).strip().upper()
    for candidate in EXPECTED_TASK_COUNTS:
        if normalized.startswith(candidate):
            return candidate
    raise ValueError(f"unknown TMCD task: {value}")


def assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{name} mismatch: actual={actual}, expected={expected}")


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--vda-manifest", required=True)
    parser.add_argument("--ecrg-config", required=True)
    parser.add_argument("--ecrg-manifest", required=True)
    parser.add_argument("--source-hashes", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    data_path = Path(args.data).resolve()
    data_manifest_path = Path(args.data_manifest).resolve()
    vda_manifest_path = Path(args.vda_manifest).resolve()
    ecrg_config_path = Path(args.ecrg_config).resolve()
    ecrg_manifest_path = Path(args.ecrg_manifest).resolve()
    source_hashes_path = Path(args.source_hashes).resolve()
    output_path = Path(args.output).resolve()

    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    for required in (
        results_path,
        summary_path,
        data_path,
        data_manifest_path,
        vda_manifest_path,
        ecrg_config_path,
        ecrg_manifest_path,
        source_hashes_path,
    ):
        if not required.is_file():
            raise ValueError(f"required formal artifact is missing: {required}")
    if output_path.exists():
        raise ValueError(f"refusing to replace formal acceptance manifest: {output_path}")

    data_manifest = read_json(data_manifest_path)
    if (
        data_manifest.get("status") != "sealed"
        or int(data_manifest.get("selected_count", -1)) != 2400
        or data_manifest.get("task_counts") != EXPECTED_TASK_COUNTS
    ):
        raise ValueError("TMCD-Test manifest is not the sealed balanced 2,400 set")

    ecrg_config = ev.load_frozen_ecrg_config(ecrg_config_path)
    ecrg_manifest = read_json(ecrg_manifest_path)
    if (
        ecrg_manifest.get("status") != "frozen"
        or int(ecrg_manifest.get("candidate_count", -1)) != 6
        or ecrg_manifest.get("tmcd_test_used") is not False
        or ecrg_manifest.get("parameter_training") is not False
        or ecrg_manifest.get("prompt_contract") != TRAINED_VDA_PROMPT_CONTRACT
        or ecrg_manifest.get("max_new_tokens") != FORMAL_VDA_MAX_NEW_TOKENS
        or ecrg_manifest.get("ecrg_config_sha256") != sha256_file(ecrg_config_path)
    ):
        raise ValueError("ECRG manifest is not the accepted frozen calibration")
    require_candidate_quality(
        ecrg_manifest.get("candidate_quality", {}), context="frozen ECRG"
    )
    if (
        ecrg_config.get("prompt_contract") != TRAINED_VDA_PROMPT_CONTRACT
        or ecrg_config.get("max_new_tokens") != FORMAL_VDA_MAX_NEW_TOKENS
        or ecrg_config.get("candidate_quality")
        != ecrg_manifest.get("candidate_quality")
    ):
        raise ValueError("ECRG config inference contract mismatch")

    vda_manifest = read_json(vda_manifest_path)
    if (
        vda_manifest.get("role") != "vda"
        or int(vda_manifest.get("round", -1)) != 3
        or vda_manifest.get("status") != "trained"
        or ecrg_config.get("vda_manifest_sha256") != sha256_file(vda_manifest_path)
        or ecrg_config.get("vda_adapter_sha256") != vda_manifest.get("adapter_sha256")
    ):
        raise ValueError("formal evaluation is not bound to the calibrated VDA3")
    adapter_sha_before = str(vda_manifest["adapter_sha256"])
    adapter_sha_after = sha256_tree(vda_manifest["adapter_path"])
    if adapter_sha_after != adapter_sha_before:
        raise ValueError("VDA3 adapter changed during formal evaluation")

    frame = pd.read_parquet(data_path)
    if len(frame) != 2400:
        raise ValueError(f"TMCD-Test parquet rows={len(frame)}, expected=2400")
    data_ids = [str(value) for value in frame["scenario_id"].tolist()]
    if len(set(data_ids)) != 2400:
        raise ValueError("TMCD-Test scenario IDs are not unique")

    results = read_jsonl(results_path)
    if len(results) != 2400:
        raise ValueError(f"formal results rows={len(results)}, expected=2400")
    result_ids = [str(row.get("scenario_id", "")) for row in results]
    if len(set(result_ids)) != 2400 or set(result_ids) != set(data_ids):
        raise ValueError("formal result scenarios do not exactly match TMCD-Test")
    if any(row.get("system") != "agentguard_zero_full" for row in results):
        raise ValueError("formal results contain a non-Full system")

    counts = Counter(task_id(row.get("task", "")) for row in results)
    if dict(counts) != EXPECTED_TASK_COUNTS:
        raise ValueError(f"formal task counts mismatch: {dict(counts)}")

    decision_count = 0
    raw_candidate_count = 0
    fallback_count = 0
    selection_reasons = Counter()
    candidate_decisions = []
    for row in results:
        tid = task_id(row["task"])
        actions = row.get("selected_actions", []) or []
        if len(actions) != EXPECTED_HORIZONS[tid]:
            raise ValueError(
                f"scenario {row['scenario_id']} decisions={len(actions)}, "
                f"expected={EXPECTED_HORIZONS[tid]}"
            )
        decision_count += len(actions)
        for action in actions:
            if int(action.get("candidate_count", -1)) != 6:
                raise ValueError("formal Full decision did not generate exactly K=6")
            raw_candidates = action.get("raw_candidates", []) or []
            raw_hashes = action.get("raw_candidate_sha256", []) or []
            if len(raw_candidates) != 6 or len(raw_hashes) != 6:
                raise ValueError("formal Full decision did not preserve all six candidates")
            if [
                hashlib.sha256(str(item).encode("utf-8")).hexdigest()
                for item in raw_candidates
            ] != raw_hashes:
                raise ValueError("saved raw candidate hash mismatch")
            if not str(action.get("public_state_sha256", "")):
                raise ValueError("formal Full decision is missing its public-state hash")
            raw_candidate_count += int(action["candidate_count"])
            diagnostics = action.get("diagnostics", {}) or {}
            if (
                diagnostics.get("selector") != "ecrg_frozen_config"
                or diagnostics.get("ecrg_config_sha256") != sha256_file(ecrg_config_path)
                or diagnostics.get("feature_access") != "public_only"
                or diagnostics.get("hidden_state_access") is not False
                or diagnostics.get("hard_gate_enforced") is not True
            ):
                raise ValueError("a formal decision bypassed the frozen public-only ECRG")
            candidate_rows = diagnostics.get("candidates", []) or []
            if len(candidate_rows) != 6:
                raise ValueError("formal ECRG diagnostics omitted raw candidate validity")
            candidate_decisions.append(candidate_rows)
            fallback_count += int(bool(diagnostics.get("fallback")))
            selection_reasons[str(diagnostics.get("selection_reason", ""))] += 1
    if decision_count != 30000 or raw_candidate_count != 180000:
        raise ValueError(
            f"formal inference volume mismatch: decisions={decision_count}, "
            f"raw_candidates={raw_candidate_count}"
        )
    candidate_quality = summarize_candidate_quality(
        candidate_decisions, expected_candidates_per_decision=6
    )
    require_candidate_quality(candidate_quality, context="formal TMCD Full inference")

    summary = read_json(summary_path)
    if (
        summary.get("system") != "agentguard_zero_full"
        or int(summary.get("num_scenarios", -1)) != 2400
        or int(summary.get("candidate_count", -1)) != 6
        or summary.get("ecrg_config_sha256") != sha256_file(ecrg_config_path)
    ):
        raise ValueError("formal summary invariants failed")
    task_metrics = summary.get("task_metrics", {}) or {}
    for name, metrics in task_metrics.items():
        tid = task_id(name)
        if int(metrics.get("num_scenarios", -1)) != EXPECTED_TASK_COUNTS[tid]:
            raise ValueError(f"summary task count failed for {name}")

    by_task = {
        tid: [row for row in results if task_id(row["task"]) == tid]
        for tid in EXPECTED_TASK_COUNTS
    }
    assert_close(
        summary["probe_yield"],
        mean([float(row["tmcd_metrics"]["probe_yield"]) for row in by_task["T1"]]),
        "T1-only Probe Yield",
    )
    assert_close(
        summary["betrayal_detection"],
        mean([float(row["tmcd_metrics"]["betrayal_detection"]) for row in by_task["T2"]]),
        "T2-only Betrayal Detection",
    )
    assert_close(
        summary["poison_success"],
        mean([float(row["tmcd_metrics"]["poison_success"]) for row in by_task["T3"]]),
        "T3-only Poison Success",
    )
    assert_close(
        summary["overresponse_rate"],
        mean([float(row["tmcd_metrics"]["overresponse"]) for row in by_task["T4"]]),
        "T4-only Overresponse",
    )
    task_su = {
        tid: mean([float(row["tmcd_metrics"]["safe_utility"]) for row in entries])
        for tid, entries in by_task.items()
    }
    assert_close(summary["safe_utility"], mean(list(task_su.values())), "macro task Safe Utility")

    shard_configs = []
    for index in range(4):
        config_path = run_dir / f"shard_{index}" / "run_config.json"
        config = read_json(config_path)
        if (
            config.get("candidate_count") != 6
            or config.get("seed") != 20260718
            or config.get("max_turns") != 16
            or config.get("max_input_tokens") != 2048
            or config.get("max_new_tokens") != FORMAL_VDA_MAX_NEW_TOKENS
            or config.get("prompt_contract") != TRAINED_VDA_PROMPT_CONTRACT
            or config.get("dtype") != "bf16"
            or config.get("attn_implementation") != "sdpa"
            or config.get("ecrg_config_sha256") != sha256_file(ecrg_config_path)
        ):
            raise ValueError(f"formal shard config mismatch: {config_path}")
        shard_configs.append(
            {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "shard_index": config["shard_index"],
                "torch_seed": config["torch_seed"],
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "tmcd_main_agentguard_zero_full_evaluation",
        "status": "sealed",
        "sealed_at": utc_now(),
        "system": "AgentGuard-Zero-Full",
        "system_id": "agentguard_zero_full",
        "backbone": "qwen3.5-4b",
        "candidate_count_per_decision": 6,
        "prompt_contract": TRAINED_VDA_PROMPT_CONTRACT,
        "max_new_tokens": FORMAL_VDA_MAX_NEW_TOKENS,
        "candidate_quality": candidate_quality,
        "scenario_count": 2400,
        "task_counts": EXPECTED_TASK_COUNTS,
        "decision_count": decision_count,
        "raw_candidate_count": raw_candidate_count,
        "raw_candidate_actions_preserved": True,
        "public_state_hashes_preserved": True,
        "candidate_replay_scope": "fixed_public_state_one_step_counterfactual",
        "selection_reasons": dict(selection_reasons),
        "fallback_count": fallback_count,
        "feature_access": "public_only",
        "hidden_state_access": False,
        "test_time_parameter_update": False,
        "test_time_threshold_tuning": False,
        "test_time_model_performance_filtering": False,
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "vda_manifest": str(vda_manifest_path),
        "vda_manifest_sha256": sha256_file(vda_manifest_path),
        "vda_adapter_sha256_before": adapter_sha_before,
        "vda_adapter_sha256_after": adapter_sha_after,
        "ecrg_config": str(ecrg_config_path),
        "ecrg_config_sha256": sha256_file(ecrg_config_path),
        "ecrg_manifest": str(ecrg_manifest_path),
        "ecrg_manifest_sha256": sha256_file(ecrg_manifest_path),
        "source_hashes": str(source_hashes_path),
        "source_hashes_sha256": sha256_file(source_hashes_path),
        "shard_configs": shard_configs,
        "results": str(results_path),
        "results_sha256": sha256_file(results_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "task_safe_utility": task_su,
    }
    atomic_write_json(output_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
