#!/usr/bin/env python3
"""Evaluate a candidate-ranker VDA on deterministic T1-T4 canonical suites."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.metrics import summarize_candidate_traces
from agentguard_zero.candidate.policy import CandidateRankerPolicy
from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.canonical_scenarios import canonical_recovery_suite
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


def _task_id(env: Any) -> str:
    metadata = env.scenario.get("metadata", {}) or {}
    return str(metadata.get("task_id", "unknown"))


def _terminal(env: Any) -> bool:
    return bool(env.t >= env.max_steps or env.attack_mitigated or env.attack_success)


def _manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "candidate_ranker_checkpoint":
        raise RuntimeError("not a candidate ranker checkpoint manifest")
    adapter = Path(str(payload.get("adapter_path", "")))
    head = Path(str(payload.get("heads_path") or payload.get("score_head_path", "")))
    if sha256_tree(adapter) != payload.get("adapter_sha256"):
        raise RuntimeError("candidate adapter hash mismatch")
    expected_head_hash = payload.get("heads_sha256") or payload.get("score_head_sha256")
    if sha256_file(head) != expected_head_hash:
        raise RuntimeError("candidate score head hash mismatch")
    return payload


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = _manifest(args.ranker_manifest)
    cache_root = Path(
        os.environ.get(
            "AGZ_TRITON_CACHE_ROOT",
            f"/tmp/agentguard_zero_triton_{os.environ.get('USER', 'user')}",
        )
    )
    cache = cache_root / f"candidate_policy_eval_{os.getpid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    groups = canonical_recovery_suite(
        scenario_count=args.scenario_count,
        group_offset=args.group_offset,
    )
    groups = [
        group
        for index, group in enumerate(groups)
        if index % args.shard_count == args.shard_index
    ]
    policy = CandidateRankerPolicy(
        model_path=args.model_path,
        adapter_path=manifest["adapter_path"],
        heads_path=manifest.get("heads_path"),
        score_head_path=None if manifest.get("heads_path") else manifest["score_head_path"],
        device=args.device,
        max_length=args.max_length,
        score_batch_size=args.score_batch_size,
    )
    teacher = PublicStateRobustTeacher()
    traces: list[dict[str, Any]] = []
    totals: defaultdict[str, float] = defaultdict(float)
    by_task: dict[str, defaultdict[str, float]] = {}
    probe_pending: dict[int, bool] = {}
    for scenarios in groups:
        live = [instantiate_scenario(copy.deepcopy(row)) for row in scenarios]
        while live:
            public_groups: dict[str, list[Any]] = defaultdict(list)
            for env in live:
                if not _terminal(env):
                    public_groups[public_state_digest(env.observe())].append(env)
            if not public_groups:
                break
            next_live: list[Any] = []
            for digest, worlds in sorted(public_groups.items()):
                observation = worlds[0].observe()
                decision = policy.decide(observation)
                teacher_decision = (
                    teacher.decide(worlds, horizon=3, enforce_min_worlds=True)
                    if len(worlds) >= 2
                    else None
                )
                selected_q = (
                    teacher_decision.q_audit.get(str(decision.semantic_id))
                    if teacher_decision is not None and decision.semantic_id
                    else None
                )
                candidate_regret = (
                    max(teacher_decision.q_audit.values()) - float(selected_q)
                    if teacher_decision is not None and selected_q is not None
                    else None
                )
                actionable = bool(
                    teacher_decision is not None
                    and teacher_decision.advantage_over_observe > 0.05 + 1.0e-12
                )
                trace = {
                    "public_state_digest": digest,
                    "task_id": _task_id(worlds[0]),
                    "candidate_id": decision.candidate_id,
                    "candidate_semantic_id": decision.semantic_id,
                    "candidate_count": decision.candidate_count,
                    "candidate_score": decision.score,
                    "candidate_regret": candidate_regret,
                    "valid": decision.valid,
                    "invalid_noop": decision.invalid_noop,
                    "reason": decision.reason,
                    "action_flags": decision.action_flags.to_dict(),
                    "follows_active_probe": any(
                        probe_pending.get(id(env), False) for env in worlds
                    ),
                    "teacher_actionable": actionable,
                    "teacher_candidate_id": (
                        teacher_decision.selected_candidate_id
                        if teacher_decision is not None
                        else None
                    ),
                    "teacher_action_family": (
                        teacher_decision.selected_category
                        if teacher_decision is not None
                        else None
                    ),
                }
                traces.append(trace)
                totals["decision_count"] += 1
                totals["valid_decision_count"] += int(decision.valid)
                totals["actionable_count"] += int(actionable)
                totals["actionable_observe_count"] += int(
                    actionable and decision.action_flags.observe_only
                )
                if candidate_regret is not None:
                    totals["candidate_regret_sum"] += float(candidate_regret)
                    totals["candidate_regret_count"] += 1
                for env in worlds:
                    probe_pending[id(env)] = decision.action_flags.active_probe
                    env.step(copy.deepcopy(decision.packet))
                    if _terminal(env):
                        task = _task_id(env)
                        task_totals = by_task.setdefault(task, defaultdict(float))
                        score = score_trajectory_v2(env)
                        utility = recovery_core_utility(env, score)
                        for counter in (totals, task_totals):
                            counter["scenario_count"] += 1
                            counter["safe_success_sum"] += int(score.get("safe_success", False))
                            counter["attack_mitigation_sum"] += int(
                                score.get("attack_mitigated", False)
                            )
                            counter["probe_yield_sum"] += float(score.get("probe_yield", 0.0))
                            counter["invalid_operation_sum"] += float(
                                score.get("invalid_state_operations", 0.0)
                            )
                            counter["safe_utility_sum"] += float(utility)
                    else:
                        next_live.append(env)
            live = next_live

    def terminal_metrics(counter: dict[str, float]) -> dict[str, Any]:
        scenarios = max(1.0, counter.get("scenario_count", 0.0))
        return {
            "scenario_count": int(counter.get("scenario_count", 0.0)),
            "safe_success": counter.get("safe_success_sum", 0.0) / scenarios,
            "attack_mitigation": counter.get("attack_mitigation_sum", 0.0) / scenarios,
            "probe_yield": counter.get("probe_yield_sum", 0.0) / scenarios,
            "invalid_operation_rate": counter.get("invalid_operation_sum", 0.0)
            / scenarios,
            "safe_utility": counter.get("safe_utility_sum", 0.0) / scenarios,
        }

    action_metrics = summarize_candidate_traces(traces)
    decisions = max(1.0, totals["decision_count"])
    metrics = {
        **terminal_metrics(totals),
        **action_metrics,
        "action_validity": totals["valid_decision_count"] / decisions,
        "actionable_observe_rate": totals["actionable_observe_count"]
        / max(1.0, totals["actionable_count"]),
        "actionable_decision_count": int(totals["actionable_count"]),
        "decoding": "candidate_argmax",
        "candidate_count": 1,
        "by_task_terminal": {
            task: terminal_metrics(counter) for task, counter in sorted(by_task.items())
        },
    }
    payload = {
        "schema_version": 1,
        "kind": "candidate_policy_evaluation_shard",
        "created_at": utc_now(),
        "ranker_manifest_sha256": sha256_file(args.ranker_manifest),
        "scenario_count_requested": args.scenario_count,
        "group_offset": args.group_offset,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "metrics": metrics,
        "raw_totals": dict(totals),
        "traces": traces,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


def merge(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not shards:
        raise ValueError("no evaluation shards")
    expected = set(range(int(shards[0]["shard_count"])))
    if {int(row["shard_index"]) for row in shards} != expected:
        raise RuntimeError("evaluation shard coverage mismatch")
    invariant = ("ranker_manifest_sha256", "group_offset", "shard_count")
    for key in invariant:
        if len({json.dumps(row[key], sort_keys=True) for row in shards}) != 1:
            raise RuntimeError(f"evaluation shard invariant mismatch: {key}")
    traces = [trace for shard in shards for trace in shard["traces"]]
    terminal_sums: defaultdict[str, float] = defaultdict(float)
    by_task: dict[str, defaultdict[str, float]] = {}
    for shard in shards:
        metrics = shard["metrics"]
        scenarios = int(metrics["scenario_count"])
        terminal_sums["scenario_count"] += scenarios
        for key in ("safe_success", "attack_mitigation", "probe_yield", "safe_utility"):
            terminal_sums[f"{key}_sum"] += float(metrics[key]) * scenarios
        for task, values in metrics.get("by_task_terminal", {}).items():
            counter = by_task.setdefault(task, defaultdict(float))
            count = int(values["scenario_count"])
            counter["scenario_count"] += count
            for key in ("safe_success", "attack_mitigation", "probe_yield", "safe_utility"):
                counter[f"{key}_sum"] += float(values[key]) * count

    def pack(counter: dict[str, float]) -> dict[str, Any]:
        count = max(1.0, counter.get("scenario_count", 0.0))
        return {
            "scenario_count": int(counter.get("scenario_count", 0.0)),
            **{
                key: counter.get(f"{key}_sum", 0.0) / count
                for key in ("safe_success", "attack_mitigation", "probe_yield", "safe_utility")
            },
        }

    action = summarize_candidate_traces(traces)
    actionable = [row for row in traces if bool(row.get("teacher_actionable", False))]
    actionable_observe = sum(
        bool((row.get("action_flags") or {}).get("observe_only", False))
        for row in actionable
    )
    metrics = {
        **pack(terminal_sums),
        **action,
        "action_validity": 1.0 - action["invalid_noop_rate"],
        "actionable_observe_rate": actionable_observe / max(1, len(actionable)),
        "actionable_decision_count": len(actionable),
        "decoding": "candidate_argmax",
        "by_task_terminal": {task: pack(row) for task, row in sorted(by_task.items())},
    }
    payload = {
        "schema_version": 1,
        "kind": "candidate_policy_evaluation",
        "created_at": utc_now(),
        "metrics": metrics,
        "traces": traces,
        "shard_sha256": {path.name: sha256_file(path) for path in args.inputs},
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--ranker-manifest", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--scenario-count", type=int, default=32)
    run_parser.add_argument("--group-offset", type=int, default=20000)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--shard-count", type=int, default=1)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--max-length", type=int, default=2048)
    run_parser.add_argument("--score-batch-size", type=int, default=4)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
