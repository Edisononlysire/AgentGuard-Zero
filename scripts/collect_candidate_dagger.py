#!/usr/bin/env python3
"""Collect Teacher-ranked candidate sets on ranker-visited error states."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.generator import CandidateGenerator
from agentguard_zero.candidate.policy import CandidateRankerPolicy
from agentguard_zero.candidate.semantic import semantic_digest
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.canonical_scenarios import canonical_recovery_suite
from agentguard_zero.recovery.public_teacher import PublicStateRobustTeacher, public_state_digest
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now
from scripts.build_candidate_dataset import (
    ACTIVE_PROBE_TOOLS,
    _hard_negatives,
    _select_scored,
    _softmax,
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ranker-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-count", type=int, default=64)
    parser.add_argument("--group-offset", type=int, default=30000)
    parser.add_argument("--max-records", type=int, default=256)
    parser.add_argument("--regret-threshold", type=float, default=0.02)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    cache_root = Path(
        os.environ.get(
            "AGZ_TRITON_CACHE_ROOT",
            f"/tmp/agentguard_zero_triton_{os.environ.get('USER', 'user')}",
        )
    )
    cache = cache_root / f"candidate_dagger_{os.getpid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    ranker_manifest = json.loads(args.ranker_manifest.read_text(encoding="utf-8"))
    policy = CandidateRankerPolicy(
        model_path=args.model_path,
        adapter_path=ranker_manifest["adapter_path"],
        heads_path=ranker_manifest.get("heads_path"),
        score_head_path=(
            None if ranker_manifest.get("heads_path") else ranker_manifest["score_head_path"]
        ),
        device=args.device,
    )
    generator = CandidateGenerator()
    teacher = PublicStateRobustTeacher()
    groups = canonical_recovery_suite(
        scenario_count=args.scenario_count, group_offset=args.group_offset
    )
    records: list[dict[str, Any]] = []
    visited = 0
    skipped = Counter()
    for group_index, scenarios in enumerate(groups):
        live = [instantiate_scenario(copy.deepcopy(row)) for row in scenarios]
        while live and len(records) < args.max_records:
            grouped: dict[str, list[Any]] = defaultdict(list)
            for env in live:
                grouped[public_state_digest(env.observe())].append(env)
            next_live: list[Any] = []
            for digest, worlds in sorted(grouped.items()):
                if len(worlds) < 2:
                    skipped["singleton_public_state"] += 1
                    continue
                observation = worlds[0].observe()
                model_decision = policy.decide(observation)
                teacher_decision = teacher.decide(
                    worlds, horizon=3, enforce_min_worlds=True
                )
                visited += 1
                model_q = teacher_decision.q_audit.get(
                    str(model_decision.semantic_id), float("-inf")
                )
                regret = max(teacher_decision.q_audit.values()) - model_q
                error_state = bool(
                    model_decision.invalid_noop
                    or model_decision.semantic_id
                    != teacher_decision.selected_candidate_id
                    or regret > args.regret_threshold
                )
                if error_state:
                    options = generator.generate_all(observation)
                    selected = _select_scored(
                        options,
                        teacher_decision.q_audit,
                        selected_id=teacher_decision.selected_candidate_id,
                        minimum=8,
                        maximum=24,
                    )
                    if selected:
                        selected = generator._remap_keys(
                            observation,
                            selected,
                            permutation_seed=args.group_offset + visited,
                        )
                        q_values = [
                            float(teacher_decision.q_audit[row.semantic_id])
                            for row in selected
                        ]
                        core_q_values = [
                            float(teacher_decision.core_q_audit[row.semantic_id])
                            for row in selected
                        ]
                        target_index = next(
                            index
                            for index, row in enumerate(selected)
                            if row.semantic_id
                            == teacher_decision.selected_candidate_id
                        )
                        record_id = hashlib.sha256(
                            f"dagger:{digest}:{visited}".encode()
                        ).hexdigest()
                        records.append(
                            {
                                "schema_version": 1,
                                "data_source": "error_focused_dagger",
                                "record_id": record_id,
                                "task_id": str(
                                    (scenarios[0].get("metadata") or {}).get(
                                        "task_id", "unknown"
                                    )
                                ),
                                "public_state_digest": digest,
                                "semantic_public_state_digest": semantic_digest(
                                    observation
                                ),
                                "semantic_scenario_fingerprint": semantic_digest(
                                    scenarios[0]
                                ),
                                "public_observation": observation,
                                "candidates": [
                                    row.public_record(include_packet=True)
                                    for row in selected
                                ],
                                "teacher_probabilities": _softmax(
                                    q_values, args.teacher_temperature
                                ),
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
                                                for observe_index, observe in enumerate(
                                                    selected
                                                )
                                                if observe.action_flags.observe_only
                                            )
                                        ),
                                        "business_risk": float(
                                            (
                                                row.compiled_packet.get("safety_check")
                                                or {}
                                            ).get("business_risk", 0.0)
                                        ),
                                        "safety_risk": float(
                                            (
                                                row.compiled_packet.get("safety_check")
                                                or {}
                                            ).get("overresponse_risk", 0.0)
                                        ),
                                    }
                                    for index, row in enumerate(selected)
                                ],
                                "teacher_full_q_best_candidate_id": max(
                                    teacher_decision.q_audit,
                                    key=lambda candidate_id: (
                                        teacher_decision.q_audit[candidate_id],
                                        candidate_id,
                                    ),
                                ),
                                "teacher_full_q_best_value": max(
                                    teacher_decision.q_audit.values()
                                ),
                                "target_candidate_key": selected[
                                    target_index
                                ].candidate_key,
                                "target_candidate_id": selected[
                                    target_index
                                ].candidate_key,
                                "target_semantic_id": teacher_decision.selected_candidate_id,
                                "target_family": teacher_decision.selected_category,
                                "hard_negative_candidate_ids": [
                                    selected[index].candidate_key
                                    for index in _hard_negatives(
                                        selected, q_values, target_index
                                    )
                                ],
                                "probe_chain_target": {
                                    "is_probe_followup_state": str(
                                        (
                                            (worlds[0].history[-1] if worlds[0].history else {})
                                            .get("action_packet", {})
                                            .get("tool_call", {})
                                            .get("name", "")
                                        )
                                    )
                                    in ACTIVE_PROBE_TOOLS,
                                    "probe_tool": str(
                                        (
                                            (worlds[0].history[-1] if worlds[0].history else {})
                                            .get("action_packet", {})
                                            .get("tool_call", {})
                                            .get("name", "")
                                        )
                                    )
                                    or None,
                                    "probe_evidence_id": str(
                                        (observation.get("last_tool_result") or {}).get(
                                            "evidence_id", ""
                                        )
                                    )
                                    or None,
                                    "evidence_used_by_target": bool(
                                        str(
                                            (
                                                observation.get("last_tool_result") or {}
                                            ).get("evidence_id", "")
                                        )
                                        and str(
                                            (
                                                observation.get("last_tool_result") or {}
                                            ).get("evidence_id", "")
                                        )
                                        in selected[target_index].referenced_ids
                                    ),
                                    "followup_flags": selected[
                                        target_index
                                    ].action_flags.to_dict(),
                                    "followup_nonobserve": not selected[
                                        target_index
                                    ].action_flags.observe_only,
                                },
                                "audit": {
                                    "world_count": len(worlds),
                                    "teacher_advantage": teacher_decision.advantage_over_observe,
                                    "visited_candidate_key": model_decision.candidate_id,
                                    "visited_semantic_id": model_decision.semantic_id,
                                    "visited_candidate_regret": regret,
                                    "visited_invalid_noop": model_decision.invalid_noop,
                                    "initial_group_index": group_index,
                                    "hidden_state_in_model_input": False,
                                },
                            }
                        )
                    else:
                        skipped["insufficient_scored_candidates"] += 1
                for env in worlds:
                    env.step(copy.deepcopy(model_decision.packet))
                    if not (
                        env.t >= env.max_steps
                        or env.attack_mitigated
                        or env.attack_success
                    ):
                        next_live.append(env)
            live = next_live
    args.output_dir.mkdir(parents=True)
    data_path = args.output_dir / "candidate_sets.jsonl"
    write_jsonl(data_path, records)
    manifest = {
        "schema_version": 1,
        "kind": "candidate_error_focused_dagger",
        "created_at": utc_now(),
        "accepted": bool(records),
        "record_count": len(records),
        "visited_state_count": visited,
        "error_state_rate": len(records) / max(1, visited),
        "skipped": dict(sorted(skipped.items())),
        "candidate_sets_sha256": sha256_file(data_path),
        "source_ranker_manifest_sha256": sha256_file(args.ranker_manifest),
        "passes": 1,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
