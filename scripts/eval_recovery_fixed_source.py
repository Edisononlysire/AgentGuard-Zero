#!/usr/bin/env python3
"""Evaluate one frozen recovery VDA on exact fixed source scenarios, K=1 greedy."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.model_policy import RecoveryModelPolicy
from agentguard_zero.recovery.action_intent import INTENT_FORMAT
from agentguard_zero.recovery.public_teacher import (
    PublicStateRobustTeacher,
    public_state_digest,
)
from agentguard_zero.recovery.source_counterfactuals import load_source_scenarios
from agentguard_zero.recovery.utility import recovery_core_utility
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    sha256_file,
    sha256_tree,
    utc_now,
)
from agentguard_zero.training.vda_dataset import build_vda_prompt


def terminal(env: Any) -> bool:
    return bool(env.t >= env.max_steps or env.attack_mitigated or env.attack_success)


def balanced_scenario_prefix(
    scenarios: list[dict[str, Any]], limit: int | None
) -> tuple[list[dict[str, Any]], str]:
    """Select a deterministic task-balanced quick subset when possible."""

    if limit is None or limit >= len(scenarios):
        return scenarios, "all"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        task = str((scenario.get("metadata", {}) or {}).get("task_id", ""))
        grouped[task].append(scenario)
    tasks = [task for task in ("T1", "T2", "T3", "T4") if grouped.get(task)]
    if len(tasks) != 4 or limit < 4:
        return scenarios[:limit], "source_prefix"
    quota, remainder = divmod(limit, len(tasks))
    selected: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        count = quota + int(index < remainder)
        selected.extend(grouped[task][:count])
    if len(selected) != limit:
        raise RuntimeError("task-balanced scenario selection is incomplete")
    return selected, "task_balanced_prefix"


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    scenarios = load_source_scenarios(args.scenarios)
    scenarios, scenario_selection = balanced_scenario_prefix(
        scenarios, args.scenario_limit
    )
    selected = [
        row
        for index, row in enumerate(scenarios)
        if index % args.shard_count == args.shard_index
    ]
    policy = RecoveryModelPolicy(
        model_path=args.model_path,
        adapter_path=args.adapter,
        device=args.device,
        max_new_tokens=320,
        output_format=args.output_format,
    )
    teacher_diagnostics = bool(
        args.teacher_diagnostics and not args.skip_teacher_diagnostics
    )
    teacher = PublicStateRobustTeacher() if teacher_diagnostics else None
    counts: defaultdict[str, float] = defaultdict(float)
    traces: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(selected, start=1):
        task_id = str((scenario.get("metadata", {}) or {}).get("task_id", ""))
        task_prefix = f"task:{task_id}:"
        env = instantiate_scenario(copy.deepcopy(scenario))
        counts["source_scenario_count"] += 1
        counts[f"{task_prefix}source_scenario_count"] += 1
        while not terminal(env):
            observation = env.observe()
            digest = public_state_digest(observation)
            teacher_decision = (
                teacher.decide([env], horizon=3, enforce_min_worlds=False)
                if teacher is not None
                else None
            )
            prompt_scenario = copy.deepcopy(env.scenario)
            prompt_scenario["scenario_id"] = f"recovery-public-{digest[:20]}"
            prompt = build_vda_prompt(
                prompt_scenario, observation, experiment_variant="full"
            )
            decision = policy.decide(prompt)
            actionable = bool(
                teacher_decision is not None
                and teacher_decision.advantage_over_observe > 0.05 + 1.0e-12
            )
            counts["decision_count"] += 1
            counts[f"{task_prefix}decision_count"] += 1
            counts["teacher_diagnostic_decision_count"] += int(
                teacher_decision is not None
            )
            counts["valid_decision_count"] += int(decision.valid)
            counts[f"{task_prefix}valid_decision_count"] += int(decision.valid)
            counts[f"category:{decision.action_category}"] += 1
            counts[f"{task_prefix}category:{decision.action_category}"] += 1
            counts["actionable_decision_count"] += int(actionable)
            counts["actionable_observe_count"] += int(
                actionable and decision.action_category == "observe"
            )
            traces.append(
                {
                    "scenario_id": str(scenario.get("scenario_id", "")),
                    "task_id": str(
                        (scenario.get("metadata", {}) or {}).get("task_id", "")
                    ),
                    "public_state_digest": digest,
                    "model_valid": decision.valid,
                    "model_parse_reason": decision.parse_reason,
                    "model_action_category": decision.action_category,
                    "model_generated_token_count": decision.generated_token_count,
                    "model_text_length": len(decision.text),
                    "model_text_preview": decision.text[:512],
                    "teacher_action_category": (
                        teacher_decision.selected_category
                        if teacher_decision is not None
                        else None
                    ),
                    "teacher_advantage": (
                        teacher_decision.advantage_over_observe
                        if teacher_decision is not None
                        else None
                    ),
                }
            )
            env.step(copy.deepcopy(decision.packet))
        score = score_trajectory_v2(env)
        counts["completed_scenarios"] += 1
        counts[f"{task_prefix}completed_scenarios"] += 1
        counts["attack_mitigation_sum"] += int(
            bool(score.get("attack_mitigated", False))
        )
        counts[f"{task_prefix}attack_mitigation_sum"] += int(
            bool(score.get("attack_mitigated", False))
        )
        counts["safe_success_sum"] += int(bool(score.get("safe_success", False)))
        counts[f"{task_prefix}safe_success_sum"] += int(
            bool(score.get("safe_success", False))
        )
        counts["probe_yield_sum"] += float(score.get("probe_yield", 0.0))
        counts[f"{task_prefix}probe_yield_sum"] += float(
            score.get("probe_yield", 0.0)
        )
        counts["invalid_operation_sum"] += float(
            score.get("invalid_state_operations", 0.0)
        )
        counts[f"{task_prefix}invalid_operation_sum"] += float(
            score.get("invalid_state_operations", 0.0)
        )
        counts["safe_utility_sum"] += recovery_core_utility(env, score)
        counts[f"{task_prefix}safe_utility_sum"] += recovery_core_utility(env, score)
        counts["business_cost_sum"] += float(score.get("business_cost", 0.0))
        counts[f"{task_prefix}business_cost_sum"] += float(
            score.get("business_cost", 0.0)
        )
        counts["overresponse_sum"] += float(score.get("overresponse", 0.0))
        counts[f"{task_prefix}overresponse_sum"] += float(
            score.get("overresponse", 0.0)
        )
        counts["poison_success_sum"] += float(score.get("poison_success", 0.0))
        counts[f"{task_prefix}poison_success_sum"] += float(
            score.get("poison_success", 0.0)
        )
        counts["betrayal_detection_sum"] += float(
            score.get("betrayal_detection", 0.0)
        )
        counts[f"{task_prefix}betrayal_detection_sum"] += float(
            score.get("betrayal_detection", 0.0)
        )
        print(
            json.dumps(
                {
                    "event": "scenario_complete",
                    "variant": args.variant,
                    "shard_index": args.shard_index,
                    "completed": scenario_index,
                    "total": len(selected),
                    "decisions": int(counts["decision_count"]),
                    "valid_decisions": int(counts["valid_decision_count"]),
                    "attack_mitigated": int(counts["attack_mitigation_sum"]),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "kind": "recovery_fixed_source_k1_shard",
        "created_at": utc_now(),
        "variant": args.variant,
        "candidate_count": 1,
        "decoding": "greedy",
        "thinking_mode": policy.thinking_mode,
        "output_format": policy.output_format,
        "teacher_diagnostics": teacher_diagnostics,
        "scenario_limit": args.scenario_limit,
        "scenario_selection": scenario_selection,
        "max_new_tokens": 320,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_sha256": sha256_file(args.scenarios),
        "adapter_sha256": sha256_tree(args.adapter),
        "raw_counts": dict(counts),
        "traces": traces,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def merge(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not shards:
        raise ValueError("no evaluation shards supplied")
    for key in (
        "variant",
        "candidate_count",
        "decoding",
        "thinking_mode",
        "output_format",
        "max_new_tokens",
        "source_sha256",
        "adapter_sha256",
        "shard_count",
        "teacher_diagnostics",
        "scenario_limit",
        "scenario_selection",
    ):
        if len({json.dumps(row.get(key), sort_keys=True) for row in shards}) != 1:
            raise RuntimeError(f"evaluation shard invariant mismatch: {key}")
    expected = set(range(int(shards[0]["shard_count"])))
    actual = {int(row["shard_index"]) for row in shards}
    if actual != expected:
        raise RuntimeError("incomplete evaluation shard coverage")
    totals: defaultdict[str, float] = defaultdict(float)
    traces: list[dict[str, Any]] = []
    for shard in shards:
        for key, value in shard["raw_counts"].items():
            totals[key] += float(value)
        traces.extend(shard.get("traces", []))
    decisions = max(1.0, totals["decision_count"])
    scenarios = max(1.0, totals["completed_scenarios"])
    actionable = max(1.0, totals["actionable_decision_count"])
    has_teacher_diagnostics = totals["teacher_diagnostic_decision_count"] > 0
    metrics = {
        "scenario_count": int(totals["completed_scenarios"]),
        "decision_count": int(totals["decision_count"]),
        "candidate_count": 1,
        "decoding": "greedy",
        "thinking_mode": shards[0]["thinking_mode"],
        "output_format": shards[0]["output_format"],
        "action_validity": totals["valid_decision_count"] / decisions,
        "actionable_observe_rate": (
            totals["actionable_observe_count"] / actionable
            if has_teacher_diagnostics
            else None
        ),
        "active_probe_rate": totals["category:active_probe"] / decisions,
        "passive_verification_rate": totals["category:passive_verification"]
        / decisions,
        "trust_memory_operation_rate": (
            totals["category:trust"] + totals["category:memory"]
        )
        / decisions,
        "mitigation_action_rate": totals["category:mitigation"] / decisions,
        "attack_mitigation": totals["attack_mitigation_sum"] / scenarios,
        "safe_success": totals["safe_success_sum"] / scenarios,
        "probe_yield": totals["probe_yield_sum"] / scenarios,
        "invalid_operation_rate": totals["invalid_operation_sum"] / decisions,
        "safe_utility": totals["safe_utility_sum"] / scenarios,
        "business_cost": totals["business_cost_sum"] / scenarios,
        "overresponse": totals["overresponse_sum"] / scenarios,
        "poison_success": totals["poison_success_sum"] / scenarios,
        "betrayal_detection": totals["betrayal_detection_sum"] / scenarios,
        "teacher_diagnostics": has_teacher_diagnostics,
        "scenario_selection": shards[0]["scenario_selection"],
    }
    task_metrics: dict[str, dict[str, Any]] = {}
    for task_id in ("T1", "T2", "T3", "T4"):
        prefix = f"task:{task_id}:"
        task_scenarios = totals[f"{prefix}completed_scenarios"]
        if task_scenarios <= 0:
            continue
        task_decisions = max(1.0, totals[f"{prefix}decision_count"])
        task_metrics[task_id] = {
            "scenario_count": int(task_scenarios),
            "decision_count": int(totals[f"{prefix}decision_count"]),
            "action_validity": totals[f"{prefix}valid_decision_count"]
            / task_decisions,
            "active_probe_rate": totals[f"{prefix}category:active_probe"]
            / task_decisions,
            "passive_verification_rate": totals[
                f"{prefix}category:passive_verification"
            ]
            / task_decisions,
            "trust_memory_operation_rate": (
                totals[f"{prefix}category:trust"]
                + totals[f"{prefix}category:memory"]
            )
            / task_decisions,
            "mitigation_action_rate": totals[f"{prefix}category:mitigation"]
            / task_decisions,
            "attack_mitigation": totals[f"{prefix}attack_mitigation_sum"]
            / task_scenarios,
            "safe_success": totals[f"{prefix}safe_success_sum"] / task_scenarios,
            "probe_yield": totals[f"{prefix}probe_yield_sum"] / task_scenarios,
            "invalid_operation_rate": totals[f"{prefix}invalid_operation_sum"]
            / task_decisions,
            "safe_utility": totals[f"{prefix}safe_utility_sum"] / task_scenarios,
            "business_cost": totals[f"{prefix}business_cost_sum"]
            / task_scenarios,
            "overresponse": totals[f"{prefix}overresponse_sum"] / task_scenarios,
            "poison_success": totals[f"{prefix}poison_success_sum"]
            / task_scenarios,
            "betrayal_detection": totals[f"{prefix}betrayal_detection_sum"]
            / task_scenarios,
        }
    output = {
        "schema_version": 1,
        "kind": "recovery_fixed_source_k1_evaluation",
        "created_at": utc_now(),
        "variant": shards[0]["variant"],
        "metrics": metrics,
        "task_metrics": task_metrics,
        "source_sha256": shards[0]["source_sha256"],
        "adapter_sha256": shards[0]["adapter_sha256"],
        "shard_sha256": {path.name: sha256_file(path) for path in args.inputs},
        "traces": traces,
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--variant", required=True)
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--adapter", type=Path, required=True)
    run_parser.add_argument("--scenarios", type=Path, required=True)
    run_parser.add_argument("--device", required=True)
    run_parser.add_argument("--shard-index", type=int, required=True)
    run_parser.add_argument("--shard-count", type=int, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--scenario-limit", type=int)
    run_parser.add_argument("--teacher-diagnostics", action="store_true")
    run_parser.add_argument("--skip-teacher-diagnostics", action="store_true")
    run_parser.add_argument(
        "--output-format",
        choices=["full_v4", INTENT_FORMAT],
        default="full_v4",
    )
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("invalid shard index/count")
        return run(args)
    return merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
