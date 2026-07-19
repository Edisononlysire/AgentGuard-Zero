#!/usr/bin/env python3
"""Build public-state listwise candidate supervision from the robust Teacher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.generator import CandidateGenerator, DEFAULT_QUOTAS
from agentguard_zero.candidate.semantic import semantic_digest
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.canonical_scenarios import (
    canonical_recovery_group,
    canonical_recovery_suite,
)
from agentguard_zero.recovery.public_teacher import (
    PublicStateRobustTeacher,
    public_state_digest,
)
from agentguard_zero.recovery.source_counterfactuals import (
    counterfactual_groups,
    load_source_scenarios,
)
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now
from agentguard_zero.world.public_projector import assert_public, project_public


ACTIVE_PROBE_TOOLS = {
    "SourceChallenge",
    "CanaryProbe",
    "DecoyProbe",
    "ShadowActionProbe",
}


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


def _select_scored(
    options: list[CandidateOption],
    q_values: dict[str, float],
    *,
    selected_id: str,
    minimum: int,
    maximum: int,
) -> list[CandidateOption]:
    scored = [row for row in options if row.semantic_id in q_values]
    by_family: dict[str, list[CandidateOption]] = defaultdict(list)
    for row in sorted(
        scored,
        key=lambda item: (-q_values[item.semantic_id], item.semantic_id),
    ):
        by_family[row.action_family].append(row)
    selected: list[CandidateOption] = []
    for family, quota in DEFAULT_QUOTAS.items():
        selected.extend(by_family[family][:quota])
    selected_ids = {row.semantic_id for row in selected}
    target = next((row for row in scored if row.semantic_id == selected_id), None)
    if target is not None and target.semantic_id not in selected_ids:
        selected.append(target)
        selected_ids.add(target.semantic_id)
    for row in sorted(
        scored,
        key=lambda item: (-q_values[item.semantic_id], item.semantic_id),
    ):
        if len(selected) >= maximum:
            break
        if row.semantic_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row.semantic_id)
    selected = selected[:maximum]
    if len(selected) < minimum or selected_id not in {row.semantic_id for row in selected}:
        return []
    return selected


def _record_id(digest: str, candidate_id: str, index: int) -> str:
    return hashlib.sha256(f"{digest}:{candidate_id}:{index}".encode()).hexdigest()


def _hard_negatives(
    candidates: list[CandidateOption], q_values: list[float], target_index: int
) -> list[int]:
    target_q = q_values[target_index]
    selected: list[int] = []

    def add(index: int | None) -> None:
        if index is not None and index != target_index and index not in selected:
            selected.append(index)

    # Actionable Observe is the critical collapse negative.
    add(next((i for i, row in enumerate(candidates) if row.action_flags.observe_only), None))
    # Keep a low-information active probe so the ranker learns probe utility,
    # not merely the presence of an active tool token.
    probe_indices = [i for i, row in enumerate(candidates) if row.action_flags.active_probe]
    add(min(probe_indices, key=lambda i: (q_values[i], candidates[i].candidate_id)) if probe_indices else None)
    # Include an expensive or wrong-target mitigation when available.
    mitigation_indices = [i for i, row in enumerate(candidates) if row.action_flags.mitigation]
    add(
        min(mitigation_indices, key=lambda i: (q_values[i], candidates[i].candidate_id))
        if mitigation_indices
        else None
    )
    for index in sorted(
        (i for i in range(len(candidates)) if i != target_index),
        key=lambda i: (q_values[i] - target_q, candidates[i].candidate_id),
    ):
        add(index)
        if len(selected) >= 4:
            break
    return selected[:4]


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.scenario_source is not None:
        groups = counterfactual_groups(load_source_scenarios(args.scenario_source))
    elif args.task_schedule:
        task_schedule = [
            item.strip().upper()
            for item in str(args.task_schedule).split(",")
            if item.strip()
        ]
        invalid = sorted(set(task_schedule) - {"T1", "T2", "T3", "T4"})
        if not task_schedule or invalid:
            raise ValueError(f"invalid task schedule: {invalid or task_schedule}")
        groups = [
            canonical_recovery_group(task_id, args.group_offset + index)
            for index, task_id in enumerate(task_schedule)
        ]
    else:
        groups = canonical_recovery_suite(
            scenario_count=args.scenario_count,
            group_offset=args.group_offset,
        )
    if args.public_action_flip:
        for group in groups:
            for scenario in group:
                events = scenario.get("event_schedule", []) or []
                if not events:
                    continue
                index = min(1, len(events) - 1)
                semantics = events[index].get("claim_semantics", {}) or {}
                current = str(semantics.get("object", ""))
                replacement = "credential_theft" if current != "credential_theft" else "persistence"
                semantics["object"] = replacement
                events[index]["claim_semantics"] = semantics
                events[index]["objective_hint"] = replacement
                events[index]["claim"] = f"public counterfactual claim suggests {replacement}"
    teacher = PublicStateRobustTeacher(
        beam_width=args.teacher_beam_width,
        max_candidates=args.teacher_max_candidates,
    )
    generator = CandidateGenerator(
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
    )
    records: list[dict[str, Any]] = []
    skipped = Counter()
    family_counts = Counter()
    task_counts = Counter()
    decision_index = 0
    rng = random.Random(args.rollout_seed)
    effective_group_limit = int(args.max_records_per_group)
    if effective_group_limit <= 0 and args.task_schedule:
        effective_group_limit = max(1, math.ceil(args.max_records / len(groups)))
    for initial_group_index, scenarios in enumerate(groups):
        group_record_start = len(records)
        live = [instantiate_scenario(copy.deepcopy(row)) for row in scenarios]
        while (
            live
            and len(records) < args.max_records
            and (
                effective_group_limit <= 0
                or len(records) - group_record_start < effective_group_limit
            )
        ):
            public_groups: dict[str, list[Any]] = defaultdict(list)
            for env in live:
                public_groups[public_state_digest(env.observe())].append(env)
            next_live: list[Any] = []
            for digest, worlds in sorted(public_groups.items()):
                if len(worlds) < 2:
                    skipped["singleton_public_state"] += 1
                    continue
                observation = project_public(worlds[0].observe())
                assert_public(observation)
                decision = teacher.decide(worlds, horizon=3, enforce_min_worlds=True)
                options = generator.generate_all(observation)
                selected = _select_scored(
                    options,
                    decision.q_audit,
                    selected_id=decision.selected_candidate_id,
                    minimum=args.min_candidates,
                    maximum=args.max_candidates,
                )
                if not selected:
                    skipped["insufficient_scored_candidates"] += 1
                else:
                    selected = generator._remap_keys(
                        observation,
                        selected,
                        permutation_seed=args.permutation_seed + decision_index,
                    )
                    q_values = [float(decision.q_audit[row.semantic_id]) for row in selected]
                    core_q_values = [
                        float(decision.core_q_audit[row.semantic_id]) for row in selected
                    ]
                    probabilities = _softmax(q_values, args.teacher_temperature)
                    best_index = next(
                        index
                        for index, row in enumerate(selected)
                        if row.semantic_id == decision.selected_candidate_id
                    )
                    negative_indices = _hard_negatives(selected, q_values, best_index)
                    record_id = _record_id(
                        digest, decision.selected_candidate_id, decision_index
                    )
                    prior_step = worlds[0].history[-1] if worlds[0].history else {}
                    prior_packet = prior_step.get("action_packet", {}) or {}
                    prior_tool = str((prior_packet.get("tool_call") or {}).get("name", ""))
                    probe_result = observation.get("last_tool_result") or {}
                    probe_evidence_id = str(probe_result.get("evidence_id", ""))
                    target_option = selected[best_index]
                    probe_chain = {
                        "is_probe_followup_state": prior_tool in ACTIVE_PROBE_TOOLS,
                        "probe_tool": prior_tool if prior_tool in ACTIVE_PROBE_TOOLS else None,
                        "probe_evidence_id": probe_evidence_id or None,
                        "evidence_used_by_target": bool(
                            probe_evidence_id
                            and probe_evidence_id in target_option.referenced_ids
                        ),
                        "followup_flags": target_option.action_flags.to_dict(),
                        "followup_nonobserve": not target_option.action_flags.observe_only,
                    }
                    task_id = str(
                        (scenarios[0].get("metadata") or {}).get(
                            "task_id", "unknown"
                        )
                    )
                    records.append(
                        {
                            "schema_version": 1,
                            "data_source": args.data_source,
                            "task_id": task_id,
                            "record_id": record_id,
                            "public_state_digest": digest,
                            "semantic_public_state_digest": semantic_digest(observation),
                            "semantic_scenario_fingerprint": semantic_digest(
                                scenarios[0]
                            ),
                            "public_observation": observation,
                            "candidates": [
                                row.public_record(include_packet=True) for row in selected
                            ],
                            "teacher_probabilities": probabilities,
                            "teacher_q_values": q_values,
                            "teacher_core_q_values": core_q_values,
                            "auxiliary_targets": [
                                {
                                    "belief": [
                                        float(row.compiled_packet["belief"][key])
                                        for key in (
                                            "exfiltration",
                                            "sabotage",
                                            "persistence",
                                            "credential_theft",
                                        )
                                    ],
                                    "uncertainty": float(
                                        row.compiled_packet.get("uncertainty", 1.0)
                                    ),
                                    "probe_value": float(
                                        q_values[index]
                                        - next(
                                            q_values[observe_index]
                                            for observe_index, observe in enumerate(selected)
                                            if observe.action_flags.observe_only
                                        )
                                    ),
                                    "business_risk": float(
                                        (row.compiled_packet.get("safety_check") or {}).get(
                                            "business_risk", 0.0
                                        )
                                    ),
                                    "safety_risk": float(
                                        (row.compiled_packet.get("safety_check") or {}).get(
                                            "overresponse_risk", 0.0
                                        )
                                    ),
                                }
                                for index, row in enumerate(selected)
                            ],
                            "teacher_full_q_best_candidate_id": max(
                                decision.q_audit,
                                key=lambda candidate_id: (
                                    decision.q_audit[candidate_id], candidate_id
                                ),
                            ),
                            "teacher_full_q_best_value": max(decision.q_audit.values()),
                            "target_candidate_key": selected[best_index].candidate_key,
                            "target_semantic_id": decision.selected_candidate_id,
                            "target_candidate_id": selected[best_index].candidate_key,
                            "target_family": decision.selected_category,
                            "hard_negative_candidate_ids": [
                                selected[index].candidate_key for index in negative_indices
                            ],
                            "probe_chain_target": probe_chain,
                            "audit": {
                                "world_count": len(worlds),
                                "teacher_robust_value": decision.robust_value,
                                "teacher_observe_value": decision.observe_value,
                                "teacher_advantage": decision.advantage_over_observe,
                                "teacher_selected_core_value": core_q_values[best_index],
                                "teacher_observe_core_value": next(
                                    core_q_values[index]
                                    for index, row in enumerate(selected)
                                    if row.action_flags.observe_only
                                ),
                                "core_first_tolerance": teacher.core_tolerance,
                                "candidate_permutation_seed": args.permutation_seed
                                + decision_index,
                                "initial_group_index": initial_group_index,
                                "decision_index": decision_index,
                                "hidden_state_in_model_input": False,
                                "teacher_q_in_model_input": False,
                            },
                        }
                    )
                    family_counts[decision.selected_category] += 1
                    task_counts[task_id] += 1
                decision_index += 1
                rollout_packet = decision.selected_packet
                if args.trajectory_policy == "noop":
                    observe = next(
                        row for row in options if row.action_flags.observe_only
                    )
                    rollout_packet = observe.compiled_packet
                elif args.trajectory_policy in {"random", "scripted"}:
                    admitted = [
                        row for row in options if row.semantic_id in decision.q_audit
                    ]
                    if args.trajectory_policy == "scripted":
                        task_id = str(
                            (scenarios[0].get("metadata") or {}).get("task_id", "")
                        )
                        preferred = {
                            "T1": "active_probe",
                            "T2": "trust",
                            "T3": "memory",
                            "T4": "passive_verification",
                        }.get(task_id, "mitigation")
                        family_rows = [
                            row for row in admitted if row.action_family == preferred
                        ]
                        if family_rows:
                            admitted = family_rows
                    if admitted:
                        rollout_packet = copy.deepcopy(rng.choice(admitted).compiled_packet)
                for env in worlds:
                    env.step(copy.deepcopy(rollout_packet))
                    if not (
                        env.t >= env.max_steps
                        or env.attack_mitigated
                        or env.attack_success
                    ):
                        next_live.append(env)
                if len(records) >= args.max_records:
                    break
                if (
                    effective_group_limit > 0
                    and len(records) - group_record_start >= effective_group_limit
                ):
                    break
            live = next_live

    accepted = bool(records) and all(
        int(row["audit"]["world_count"]) >= 2 for row in records
    )
    manifest = {
        "schema_version": 1,
        "kind": "candidate_listwise_teacher_dataset",
        "created_at": utc_now(),
        "accepted": accepted,
        "scenario_count": sum(len(group) for group in groups),
        "data_source": args.data_source,
        "trajectory_policy": args.trajectory_policy,
        "public_action_flip": args.public_action_flip,
        "scenario_source": str(args.scenario_source.resolve()) if args.scenario_source else None,
        "scenario_source_sha256": sha256_file(args.scenario_source)
        if args.scenario_source
        else None,
        "task_schedule": (
            [item.strip().upper() for item in args.task_schedule.split(",") if item.strip()]
            if args.task_schedule
            else None
        ),
        "record_count": len(records),
        "candidate_count_min": min((len(row["candidates"]) for row in records), default=0),
        "candidate_count_max": max((len(row["candidates"]) for row in records), default=0),
        "target_family_counts": dict(sorted(family_counts.items())),
        "task_record_counts": dict(sorted(task_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "teacher_temperature": args.teacher_temperature,
        "candidate_keys_randomized": True,
        "probe_chain_record_count": sum(
            bool(row["probe_chain_target"]["is_probe_followup_state"])
            for row in records
        ),
        "formal_world_minimum": 2,
        "model_input_fields": ["public_observation", "candidate public fields"],
        "offline_label_fields": [
            "teacher_probabilities",
            "teacher_q_values",
            "teacher_core_q_values",
            "auxiliary_targets",
            "target_candidate_id",
        ],
    }
    return records, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-source", type=Path)
    parser.add_argument("--scenario-count", type=int, default=128)
    parser.add_argument("--group-offset", type=int, default=10000)
    parser.add_argument(
        "--task-schedule",
        help="Comma-separated canonical task schedule, for example T1,T2,T3,T4,T1.",
    )
    parser.add_argument("--max-records", type=int, default=512)
    parser.add_argument(
        "--max-records-per-group",
        type=int,
        default=0,
        help="Optional cap that prevents long task trajectories from crowding out T1-T4 coverage.",
    )
    parser.add_argument("--min-candidates", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-beam-width", type=int, default=20)
    parser.add_argument("--teacher-max-candidates", type=int, default=96)
    parser.add_argument("--permutation-seed", type=int, default=20260719)
    parser.add_argument(
        "--trajectory-policy",
        choices=["teacher", "random", "scripted", "noop"],
        default="teacher",
    )
    parser.add_argument("--data-source", default="teacher")
    parser.add_argument("--public-action-flip", action="store_true")
    parser.add_argument("--rollout-seed", type=int, default=20260719)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    records, manifest = build(args)
    args.output_dir.mkdir(parents=True)
    data_path = args.output_dir / "candidate_sets.jsonl"
    _atomic_jsonl(data_path, records)
    manifest["candidate_sets_sha256"] = sha256_file(data_path)
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
