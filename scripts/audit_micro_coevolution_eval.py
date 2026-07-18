#!/usr/bin/env python3
"""Fail-closed pre/post gate for an isolated co-evolution scale pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ACTIVE_PROBES = {"SourceChallenge", "CanaryProbe", "DecoyProbe", "ShadowActionProbe"}
PASSIVE_TOOLS = {"LogQuery", "CrossCheck", "ProvenanceCheck", "GraphQuery"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _category(action: dict[str, Any]) -> str:
    packet = action.get("selected_packet", {}) or {}
    tool = str((packet.get("tool_call", {}) or {}).get("name", "None"))
    if tool in ACTIVE_PROBES:
        return "active_probe"
    if tool in PASSIVE_TOOLS:
        return "passive_verification"
    if packet.get("trust_operations"):
        return "trust"
    if packet.get("memory_operations") or packet.get("memory_usage"):
        return "memory"
    response = packet.get("response", {}) or {}
    if str(response.get("action", "Observe")) != "Observe" or str(
        response.get("tier", "L0")
    ) != "L0":
        return "mitigation"
    return "observe"


def _action_profile(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    decisions = 0
    for result in results:
        for action in result.get("selected_actions", []) or []:
            if isinstance(action, dict):
                counts[_category(action)] += 1
                decisions += 1
    return {
        "decision_count": decisions,
        "counts": dict(counts),
        "rates": {
            name: counts[name] / max(1, decisions)
            for name in (
                "observe",
                "passive_verification",
                "active_probe",
                "trust",
                "memory",
                "mitigation",
            )
        },
        "supported_category_count": sum(value > 0 for value in counts.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-run-dir", type=Path, required=True)
    parser.add_argument("--post-run-dir", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--vda-manifest", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--expected-scenarios", type=int, default=200)
    parser.add_argument("--expected-candidates", type=int, default=1000)
    parser.add_argument("--expected-train-size", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    pre_summary_path = args.pre_run_dir / "summary.json"
    post_summary_path = args.post_run_dir / "summary.json"
    pre_results_path = args.pre_run_dir / "results.jsonl"
    post_results_path = args.post_run_dir / "results.jsonl"
    pre = _read_json(pre_summary_path)
    post = _read_json(post_summary_path)
    pre_results = _read_jsonl(pre_results_path)
    post_results = _read_jsonl(post_results_path)
    pool = _read_json(args.pool_manifest)
    vda = _read_json(args.vda_manifest)
    failures: list[str] = []

    if len(pre_results) != args.expected_scenarios or len(post_results) != args.expected_scenarios:
        failures.append("scenario_count")
    pre_ids = {str(item.get("scenario_id", "")) for item in pre_results}
    post_ids = {str(item.get("scenario_id", "")) for item in post_results}
    if pre_ids != post_ids or len(pre_ids) != args.expected_scenarios:
        failures.append("pre_post_scenario_identity")
    if int(pre.get("candidate_count", -1)) != 1 or int(post.get("candidate_count", -1)) != 1:
        failures.append("candidate_count_not_k1")

    if pool.get("selection_policy") != "pilot_balanced_50_40_10":
        failures.append("selection_policy")
    if int(pool.get("initial_candidate_count", -1)) != args.expected_candidates:
        failures.append("initial_candidate_pool_size")
    if int(pool.get("candidate_count", -1)) < args.expected_candidates:
        failures.append("candidate_pool_size")
    if int((pool.get("split_counts", {}) or {}).get("train", -1)) != args.expected_train_size:
        failures.append("train_size")
    if args.expected_train_size % 10:
        raise ValueError("expected train size must be divisible by 10")
    expected_difficulty_counts = {
        "easy": args.expected_train_size * 5 // 10,
        "frontier": args.expected_train_size * 4 // 10,
        "hard_reachable": args.expected_train_size // 10,
    }
    if (pool.get("train_difficulty_counts", {}) or {}) != expected_difficulty_counts:
        failures.append("train_difficulty_mix")
    if int(vda.get("round", -1)) != args.round:
        failures.append("vda_round")
    training = vda.get("training_config", {}) or {}
    if int(training.get("train_size", -1)) != args.expected_train_size:
        failures.append("vda_manifest_train_size")
    if training.get("pilot_selection_policy") != "pilot_balanced_50_40_10":
        failures.append("vda_manifest_selection_policy")
    if training.get("use_kl_loss") is not True or not math.isclose(
        float(training.get("kl_loss_coef", -1.0)), 0.02, abs_tol=1.0e-12
    ):
        failures.append("vda_kl_gate")
    if not math.isclose(
        float(training.get("learning_rate", -1.0)), 1.0e-6, abs_tol=1.0e-12
    ):
        failures.append("vda_learning_rate")

    pre_profile = _action_profile(pre_results)
    post_profile = _action_profile(post_results)
    post_validity = 1.0 - max(
        float(post.get("json_parse_failure_rate", 1.0)),
        float(post.get("invalid_tool_call_rate", 1.0)),
        float(post.get("invalid_response_action_rate", 1.0)),
    )
    if post_validity < 0.98:
        failures.append("post_action_validity")
    if float(post_profile["rates"]["observe"]) > 0.80:
        failures.append("post_observe_above_80pct")
    if float(post_profile["rates"]["active_probe"]) <= 0.0:
        failures.append("post_active_probe_zero")
    if int(post_profile["supported_category_count"]) < 3:
        failures.append("post_action_support_below_three_categories")
    if float(post.get("attack_mitigation", 0.0)) < 0.10:
        failures.append("post_attack_mitigation_below_10pct")
    if float(post.get("safe_success_rate", 0.0)) < 0.10:
        failures.append("post_safe_success_below_10pct")
    if float(post.get("probe_yield", 0.0)) <= 0.0:
        failures.append("post_probe_yield_zero")
    if float(post.get("safe_utility", -math.inf)) <= float(pre.get("safe_utility", math.inf)):
        failures.append("safe_utility_not_improved")
    if float(post.get("attack_mitigation", 0.0)) < float(pre.get("attack_mitigation", 0.0)):
        failures.append("attack_mitigation_decreased")

    output = {
        "schema_version": 1,
        "kind": "tmcd_micro_coevolution_round_gate",
        "round": args.round,
        "accepted": not failures,
        "status": "accepted" if not failures else "rejected",
        "failures": failures,
        "thresholds": {
            "scenario_count": args.expected_scenarios,
            "initial_candidate_pool_size": args.expected_candidates,
            "candidate_pool_size_min": args.expected_candidates,
            "train_size": args.expected_train_size,
            "train_difficulty_counts": expected_difficulty_counts,
            "candidate_count": 1,
            "post_action_validity_min": 0.98,
            "post_observe_max": 0.80,
            "post_attack_mitigation_min": 0.10,
            "post_safe_success_min": 0.10,
            "post_probe_yield_positive": True,
            "safe_utility_must_improve": True,
            "attack_mitigation_must_not_decrease": True,
        },
        "pre_summary": pre,
        "post_summary": post,
        "pre_action_profile": pre_profile,
        "post_action_profile": post_profile,
        "lineage": {
            "pool_manifest": str(args.pool_manifest.resolve()),
            "pool_manifest_sha256": _sha256(args.pool_manifest),
            "vda_manifest": str(args.vda_manifest.resolve()),
            "vda_manifest_sha256": _sha256(args.vda_manifest),
            "pre_results_sha256": _sha256(pre_results_path),
            "post_results_sha256": _sha256(post_results_path),
        },
        "next_stage": "next_micro_round" if not failures else "stop_and_scale_or_repair",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
