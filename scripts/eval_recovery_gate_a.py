#!/usr/bin/env python3
"""Run or merge frozen K=1 greedy Gate-A evaluation shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.gates import evaluate_gate_a
from agentguard_zero.recovery.model_policy import RecoveryModelPolicy
from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION
from agentguard_zero.recovery.public_teacher import (
    PublicStateRobustTeacher,
    public_state_digest,
)
from agentguard_zero.recovery.utility import recovery_core_utility
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    sha256_file,
    sha256_tree,
    utc_now,
)
from agentguard_zero.training.vda_dataset import build_vda_prompt


def _scenario_from_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("scenario", "scenario_json"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    if row.get("protocol_version") == "tmcd-v2":
        return row
    raise ValueError("row does not contain a TMCD scenario")


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict(orient="records")
    elif path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload.get("groups"), list):
            rows = [item for group in payload["groups"] for item in group]
        else:
            rows = payload.get("scenarios", [])
    return [_scenario_from_row(dict(row)) for row in rows]


def _group_scenarios(
    rows: Iterable[tuple[str, dict[str, Any]]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source, scenario in rows:
        digest = public_state_digest(instantiate_scenario(scenario).observe())
        grouped[(source, digest)].append(scenario)
    singletons = [key for key, values in grouped.items() if len(values) < 2]
    if singletons:
        raise RuntimeError(
            f"Gate-A input has {len(singletons)} unmatched public-state group(s)"
        )
    return [(key[0], grouped[key]) for key in sorted(grouped)]


def _load_adapter_manifest(path: Path, arm: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != RECOVERY_PROTOCOL_VERSION:
        raise RuntimeError("adapter manifest has the wrong recovery protocol")
    if payload.get("arm") != arm:
        raise RuntimeError("adapter manifest arm mismatch")
    if payload.get("status") != "trained_pending_gate_a":
        raise RuntimeError("adapter is not in trained_pending_gate_a state")
    adapter = Path(str(payload.get("adapter_path", "")))
    if not adapter.is_dir() or sha256_tree(adapter) != payload.get("adapter_sha256"):
        raise RuntimeError("adapter hash verification failed")
    return payload


def _terminal(env: Any) -> bool:
    return bool(env.t >= env.max_steps or env.attack_mitigated or env.attack_success)


def _run_shard(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = _load_adapter_manifest(args.adapter_manifest, args.arm)
    original = [("original", row) for row in _load_scenarios(args.original)]
    canonical = [("canonical", row) for row in _load_scenarios(args.canonical)]
    groups = _group_scenarios([*original, *canonical])
    selected_groups = [
        item
        for index, item in enumerate(groups)
        if index % args.shard_count == args.shard_index
    ]
    policy = RecoveryModelPolicy(
        model_path=args.model_path,
        adapter_path=manifest["adapter_path"],
        device=args.device,
        max_new_tokens=320,
    )
    teacher = PublicStateRobustTeacher()
    counts: defaultdict[str, float] = defaultdict(float)
    traces: list[dict[str, Any]] = []
    for source, scenarios in selected_groups:
        live = [instantiate_scenario(copy.deepcopy(row)) for row in scenarios]
        counts[f"{source}_scenario_count"] += len(live)
        while live:
            public_groups: dict[str, list[Any]] = defaultdict(list)
            for env in live:
                if not _terminal(env):
                    public_groups[public_state_digest(env.observe())].append(env)
            if not public_groups:
                break
            next_live: list[Any] = []
            for digest, worlds in sorted(public_groups.items()):
                teacher_decision = teacher.decide(
                    worlds,
                    horizon=3,
                    enforce_min_worlds=False,
                )
                prompt_scenario = copy.deepcopy(worlds[0].scenario)
                prompt_scenario["scenario_id"] = f"recovery-public-{digest[:20]}"
                prompt = build_vda_prompt(
                    prompt_scenario,
                    worlds[0].observe(),
                    experiment_variant="full",
                )
                model_decision = policy.decide(prompt)
                counts["decision_count"] += 1
                counts["valid_decision_count"] += int(model_decision.valid)
                counts[f"category:{model_decision.action_category}"] += 1
                actionable = teacher_decision.advantage_over_observe > 0.05 + 1.0e-12
                counts["actionable_decision_count"] += int(actionable)
                counts["actionable_observe_count"] += int(
                    actionable and model_decision.action_category == "observe"
                )
                traces.append(
                    {
                        "public_state_digest": digest,
                        "source": source,
                        "model_valid": model_decision.valid,
                        "model_parse_reason": model_decision.parse_reason,
                        "model_action_category": model_decision.action_category,
                        "teacher_advantage": teacher_decision.advantage_over_observe,
                        "teacher_action_category": teacher_decision.selected_category,
                    }
                )
                for env in worlds:
                    env.step(copy.deepcopy(model_decision.packet))
                    if _terminal(env):
                        score = score_trajectory_v2(env)
                        counts["completed_scenarios"] += 1
                        counts["attack_mitigation_sum"] += int(
                            bool(score.get("attack_mitigated", False))
                        )
                        counts["safe_success_sum"] += int(
                            bool(score.get("safe_success", False))
                        )
                        counts["probe_yield_sum"] += float(
                            score.get("probe_yield", 0.0)
                        )
                        counts["invalid_operation_sum"] += float(
                            score.get("invalid_state_operations", 0.0)
                        )
                        counts["safe_utility_sum"] += recovery_core_utility(env, score)
                    else:
                        next_live.append(env)
            live = next_live

    payload = {
        "schema_version": 1,
        "kind": "recovery_gate_a_shard",
        "protocol_version": RECOVERY_PROTOCOL_VERSION,
        "created_at": utc_now(),
        "arm": args.arm,
        "candidate_count": 1,
        "decoding": "greedy",
        "max_new_tokens": 320,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "model_identity": manifest["base_model"],
        "adapter_sha256": manifest["adapter_sha256"],
        "adapter_manifest_sha256": sha256_file(args.adapter_manifest),
        "original_source_sha256": sha256_file(args.original),
        "canonical_source_sha256": sha256_file(args.canonical),
        "raw_counts": dict(counts),
        "traces": traces,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _merge(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not shards:
        raise ValueError("no Gate-A shards supplied")
    invariant_keys = (
        "arm",
        "candidate_count",
        "decoding",
        "max_new_tokens",
        "adapter_sha256",
        "adapter_manifest_sha256",
        "original_source_sha256",
        "canonical_source_sha256",
        "shard_count",
    )
    for key in invariant_keys:
        if len({json.dumps(row.get(key), sort_keys=True) for row in shards}) != 1:
            raise RuntimeError(f"Gate-A shard invariant mismatch: {key}")
    expected = set(range(int(shards[0]["shard_count"])))
    actual = {int(row["shard_index"]) for row in shards}
    if actual != expected:
        raise RuntimeError(f"Gate-A shard coverage mismatch: {sorted(actual)}")
    totals: defaultdict[str, float] = defaultdict(float)
    for shard in shards:
        for key, value in shard["raw_counts"].items():
            totals[key] += float(value)
    decisions = max(1.0, totals["decision_count"])
    scenarios = int(totals["completed_scenarios"])
    actionable = totals["actionable_decision_count"]
    metrics = {
        "scenario_count": scenarios,
        "original_gate_scenarios": int(totals["original_scenario_count"]),
        "new_canonical_scenarios": int(totals["canonical_scenario_count"]),
        "candidate_count": 1,
        "decoding": "greedy",
        "action_validity": totals["valid_decision_count"] / decisions,
        "actionable_observe_rate": totals["actionable_observe_count"]
        / max(1.0, actionable),
        "active_probe_rate": totals["category:active_probe"] / decisions,
        "attack_mitigation": totals["attack_mitigation_sum"] / max(1, scenarios),
        "safe_success": totals["safe_success_sum"] / max(1, scenarios),
        "probe_yield": totals["probe_yield_sum"] / max(1, scenarios),
        "trust_memory_operation_rate": (
            totals["category:trust"] + totals["category:memory"]
        )
        / decisions,
        "invalid_operation_rate": totals["invalid_operation_sum"] / decisions,
        "safe_utility": totals["safe_utility_sum"] / max(1, scenarios),
        "decision_count": int(totals["decision_count"]),
        "actionable_decision_count": int(actionable),
    }
    verdict = evaluate_gate_a(metrics, arm=str(shards[0]["arm"]))
    output = {
        "schema_version": 1,
        "kind": "recovery_gate_a_merged",
        "protocol_version": RECOVERY_PROTOCOL_VERSION,
        "created_at": utc_now(),
        "metrics": metrics,
        "verdict": verdict.to_dict(),
        "accepted": verdict.accepted,
        "shard_sha256": {path.name: sha256_file(path) for path in args.inputs},
        "adapter_sha256": shards[0]["adapter_sha256"],
        "next_stage": verdict.next_stage,
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if verdict.accepted else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=["qwen3.5_base", "vda_1"], required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--adapter-manifest", type=Path, required=True)
    run.add_argument("--original", type=Path, required=True)
    run.add_argument("--canonical", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--shard-count", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("invalid Gate-A shard index/count")
        return _run_shard(args)
    return _merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
