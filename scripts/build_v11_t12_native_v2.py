#!/usr/bin/env python3
"""Generate and freeze a native, public-only, family/stage-balanced T1/T2-v2 split."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.cyber_grounding import (
    cyber_grounded_recovery_group_v2,
)
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.env.checker import full_check
from agentguard_zero.schemas.scenario_schema_v2 import public_prefix_hash
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now
from agentguard_zero.world.public_projector import assert_public
from scripts.build_candidate_dataset_v11 import (
    _atomic_jsonl,
    _public_scenario_fingerprint,
    build,
)


TASKS = ("T1", "T2")
DECISION_FAMILIES = (
    "observe",
    "passive_verification",
    "active_probe",
    "trust",
    "mitigation",
)
ATTACK_STAGES = (
    "initial_access",
    "discovery",
    "lateral_movement",
    "collection",
    "exfiltration",
)
TRAJECTORY_POLICIES = (
    "balanced_chain",
    "balanced_chain",
    "balanced_chain",
    "random",
    "noop",
)
FORBIDDEN_PUBLIC_KEYS = {
    "true_attack",
    "hidden_world",
    "oracle",
    "oracle_ledger",
    "truth_value",
    "is_fake",
    "teacher_q",
    "teacher_q_values",
    "teacher_core_q_values",
    "task_id",
}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _native_group(task: str, group_index: int) -> list[dict[str, Any]]:
    group = cyber_grounded_recovery_group_v2(task, group_index)
    for scenario in group:
        scenario["true_attack"]["phase_schedule"] = list(ATTACK_STAGES)
        constraints = scenario.setdefault("defense_constraints", {})
        constraints["horizon"] = max(7, int(constraints.get("horizon", 0) or 0))
        constraints["require_passive_check_before_active_probe"] = True
        metadata = dict(scenario.get("metadata") or {})
        metadata["native_t12_v2_protocol"] = "v11_t12_native_v2_20260723"
        metadata["generation_attack_stage_count"] = len(ATTACK_STAGES)
        scenario["metadata"] = metadata
        if task == "T2":
            scenario["prefix_hash"] = public_prefix_hash(scenario)
        checks = full_check(scenario)
        if not checks.get("all_ok", False):
            raise ValueError(
                f"invalid native T1/T2 scenario {scenario.get('scenario_id')}: {checks}"
            )
    return group


def _attack_stage(row: dict[str, Any]) -> str:
    step = max(0, int(row.get("trajectory_step", 0)))
    return ATTACK_STAGES[min(step, len(ATTACK_STAGES) - 1)]


def _candidate_set_hash(row: dict[str, Any]) -> str:
    public_candidates = [
        {
            "candidate_key": str(item.get("candidate_key", "")),
            "semantic_id": str(item.get("semantic_id", "")),
            "action_family": str(item.get("action_family", "")),
            "compiled_packet": item.get("compiled_packet") or {},
        }
        for item in row.get("candidates", [])
    ]
    return _stable_hash(public_candidates)


def _apply_lifecycle_supervision(row: dict[str, Any]) -> bool:
    """Use the Teacher-best candidate inside the public lifecycle family.

    The robust Teacher's unconstrained Top-1 is retained for regret and
    acceptable-set losses.  The primary candidate/family target is the
    candidate already chosen by ``_causal_target``: a public-state lifecycle
    family followed by Teacher-Q maximization within that family.
    """

    causal_id = str(row.get("causal_target_candidate_id", ""))
    causal_family = str(row.get("causal_target_family", ""))
    candidates = list(row.get("candidates") or [])
    causal = next(
        (
            candidate
            for candidate in candidates
            if str(candidate.get("candidate_key", "")) == causal_id
        ),
        None,
    )
    if causal is None or causal_family not in DECISION_FAMILIES:
        # The upstream candidate builder may legitimately produce a public
        # state for which no member of the required lifecycle family survives
        # legality/Teacher admission.  Such a state is ineligible for this
        # lifecycle-balanced split; it must not be relabelled with the
        # unconstrained Teacher target.
        return False
    audit = dict(row.get("audit") or {})
    audit["unconstrained_teacher_target_candidate_id"] = str(
        row.get("target_candidate_id", "")
    )
    audit["unconstrained_teacher_target_family"] = str(row.get("target_family", ""))
    audit["unconstrained_teacher_target_semantic_id"] = str(
        row.get("target_semantic_id", "")
    )
    audit["supervision_target_override"] = (
        "public_lifecycle_family_then_teacher_q_v1"
    )
    row["audit"] = audit
    row["target_candidate_key"] = causal_id
    row["target_candidate_id"] = causal_id
    row["target_semantic_id"] = str(causal.get("semantic_id", ""))
    row["target_family"] = causal_family
    row["hard_negative_candidate_ids"] = [
        str(candidate_id)
        for candidate_id in row.get("hard_negative_candidate_ids", [])
        if str(candidate_id) != causal_id
    ]
    row["trajectory_negative_candidate_ids"] = [
        str(candidate_id)
        for candidate_id in row.get("trajectory_negative_candidate_ids", [])
        if str(candidate_id) != causal_id
    ]
    return True


def _annotate_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    shard_group_offset: int,
    trajectory_policy: str,
) -> list[dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    for row in rows:
        if not _apply_lifecycle_supervision(row):
            continue
        audit = dict(row.get("audit") or {})
        local_group_index = int(audit.get("initial_group_index", -1))
        row["generation_audit"] = {
            "schema_version": 1,
            "split": split,
            "generator_group_index": shard_group_offset + local_group_index,
            "attack_stage": _attack_stage(row),
            "trajectory_policy": trajectory_policy,
            "hidden_generation_stratum_in_model_input": False,
        }
        row["candidate_set_hash"] = _candidate_set_hash(row)
        admitted.append(row)
    return admitted


def _build_shard(
    split: str,
    task: str,
    shard_index: int,
    *,
    group_offset: int,
    groups_per_shard: int,
    seed: int,
    teacher_beam_width: int,
    teacher_max_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_index = TASKS.index(task)
    split_offset = 0 if split == "train" else 500_000
    shard_group_offset = (
        group_offset
        + split_offset
        + task_index * 100_000
        + shard_index * groups_per_shard
    )
    shard_seed = seed + split_offset + task_index * 100_000 + shard_index * 101
    trajectory_policy = TRAJECTORY_POLICIES[shard_index % len(TRAJECTORY_POLICIES)]
    groups = [
        _native_group(task, shard_group_offset + index)
        for index in range(groups_per_shard)
    ]
    args = SimpleNamespace(
        scenario_source=None,
        scenario_groups=groups,
        task_schedule=None,
        scenario_count=2 * groups_per_shard,
        group_offset=shard_group_offset,
        robust_world_multiplicity=2,
        public_action_flip=False,
        teacher_beam_width=teacher_beam_width,
        teacher_max_candidates=teacher_max_candidates,
        min_candidates=8,
        max_candidates=24,
        max_records=groups_per_shard * 14,
        max_records_per_group=14,
        permutation_seed=shard_seed,
        rollout_seed=shard_seed,
        trajectory_policy=trajectory_policy,
        allow_post_probe_singletons=True,
        post_mitigation_audit_steps=1,
        teacher_temperature=0.1,
        data_source=f"v11_t12_native_v2_{split}",
        protocol_role="supervised_train" if split == "train" else "heldout_dev",
    )
    raw_rows, manifest = build(args)
    rows = _annotate_rows(
        raw_rows,
        split=split,
        shard_group_offset=shard_group_offset,
        trajectory_policy=trajectory_policy,
    )
    manifest = copy.deepcopy(manifest)
    manifest["native_t12_v2"] = True
    manifest["trajectory_policy"] = trajectory_policy
    manifest["post_mitigation_audit_steps"] = 1
    manifest["raw_record_count"] = len(raw_rows)
    manifest["lifecycle_admitted_record_count"] = len(rows)
    manifest["lifecycle_rejected_record_count"] = len(raw_rows) - len(rows)
    manifest["lifecycle_admission_rate"] = len(rows) / max(1, len(raw_rows))
    if not rows:
        manifest["accepted"] = False
        manifest.setdefault("failures", []).append("no_lifecycle_admitted_records")
    return rows, manifest


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _public_contract(row: dict[str, Any]) -> bool:
    observation = copy.deepcopy(row.get("public_observation") or {})
    try:
        assert_public(observation)
    except Exception:
        return False
    lowered = {key.lower() for key in _walk_keys(observation)}
    return not any(
        forbidden in key
        for key in lowered
        for forbidden in FORBIDDEN_PUBLIC_KEYS
    )


def _candidate_contract(row: dict[str, Any]) -> bool:
    candidates = [CandidateOption.from_record(item) for item in row.get("candidates", [])]
    ids = {candidate.candidate_id for candidate in candidates}
    count = len(candidates)
    return bool(
        count >= 8
        and len(row.get("teacher_probabilities") or []) == count
        and len(row.get("teacher_policy_scores") or []) == count
        and len(row.get("teacher_q_values") or []) == count
        and len(row.get("teacher_core_q_values") or []) == count
        and len(row.get("teacher_acceptable_mask") or []) == count
        and len(row.get("outcome_targets") or []) == count
        and str(row.get("target_candidate_id", "")) in ids
        and str(row.get("causal_target_candidate_id", "")) in ids
        and str(row.get("target_family", "")) in DECISION_FAMILIES
        and str(row.get("candidate_set_hash", "")) == _candidate_set_hash(row)
    )


def _blocked_fingerprints(
    *,
    scenario_paths: list[Path],
    record_paths: list[Path],
) -> tuple[set[str], dict[str, str]]:
    blocked: set[str] = set()
    source_hashes: dict[str, str] = {}
    for path in scenario_paths:
        source_hashes[str(path)] = sha256_file(path)
        for value in _read_jsonl(path):
            scenarios = value if isinstance(value, list) else [value]
            blocked.update(
                _public_scenario_fingerprint(scenario)
                for scenario in scenarios
                if isinstance(scenario, dict)
            )
    for path in record_paths:
        source_hashes[str(path)] = sha256_file(path)
        blocked.update(
            str(row.get("semantic_scenario_fingerprint", ""))
            for row in _read_jsonl(path)
            if str(row.get("semantic_scenario_fingerprint", ""))
        )
    return blocked, source_hashes


def _has_ordered_subsequence(values: list[str], required: tuple[str, ...]) -> bool:
    cursor = 0
    for value in values:
        if cursor < len(required) and value == required[cursor]:
            cursor += 1
    return cursor == len(required)


def _complete_chain_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("trajectory_id", ""))].append(row)
    chains: list[list[dict[str, Any]]] = []
    required = (
        "passive_verification",
        "active_probe",
        "trust",
        "mitigation",
        "observe",
    )
    for trajectory_rows in grouped.values():
        ordered = sorted(
            trajectory_rows,
            key=lambda row: (
                int(row.get("trajectory_step", -1)),
                str(row.get("record_id", "")),
            ),
        )
        families = [str(row.get("causal_target_family", "")) for row in ordered]
        if _has_ordered_subsequence(families, required):
            picked: list[dict[str, Any]] = []
            cursor = 0
            for row, family in zip(ordered, families, strict=True):
                if cursor < len(required) and family == required[cursor]:
                    picked.append(row)
                    cursor += 1
            if len(picked) == len(required):
                chains.append(picked)
    return sorted(
        chains,
        key=lambda chain: (
            str(chain[0].get("semantic_scenario_fingerprint", "")),
            str(chain[0].get("trajectory_id", "")),
        ),
    )


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    original: int


def _add_edge(graph: list[list[_Edge]], source: int, target: int, capacity: int) -> None:
    forward = _Edge(target, len(graph[target]), capacity, capacity)
    reverse = _Edge(source, len(graph[source]), 0, 0)
    graph[source].append(forward)
    graph[target].append(reverse)


def _max_flow_cell_quotas(
    capacities: dict[tuple[str, str], int],
    family_remaining: dict[str, int],
    stage_remaining: dict[str, int],
) -> dict[tuple[str, str], int]:
    source = 0
    family_node = {family: 1 + index for index, family in enumerate(DECISION_FAMILIES)}
    stage_node = {
        stage: 1 + len(DECISION_FAMILIES) + index
        for index, stage in enumerate(ATTACK_STAGES)
    }
    sink = 1 + len(DECISION_FAMILIES) + len(ATTACK_STAGES)
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    for family in DECISION_FAMILIES:
        _add_edge(graph, source, family_node[family], family_remaining[family])
        for stage in ATTACK_STAGES:
            _add_edge(
                graph,
                family_node[family],
                stage_node[stage],
                capacities.get((family, stage), 0),
            )
    for stage in ATTACK_STAGES:
        _add_edge(graph, stage_node[stage], sink, stage_remaining[stage])

    total = 0
    while True:
        parent: list[tuple[int, int] | None] = [None] * len(graph)
        queue = deque([source])
        parent[source] = (-1, -1)
        while queue and parent[sink] is None:
            node = queue.popleft()
            for edge_index, edge in enumerate(graph[node]):
                if edge.cap > 0 and parent[edge.to] is None:
                    parent[edge.to] = (node, edge_index)
                    queue.append(edge.to)
        if parent[sink] is None:
            break
        amount = 10**9
        node = sink
        while node != source:
            previous, edge_index = parent[node]  # type: ignore[misc]
            amount = min(amount, graph[previous][edge_index].cap)
            node = previous
        node = sink
        while node != source:
            previous, edge_index = parent[node]  # type: ignore[misc]
            edge = graph[previous][edge_index]
            edge.cap -= amount
            graph[node][edge.rev].cap += amount
            node = previous
        total += amount

    expected = sum(family_remaining.values())
    if total != expected or total != sum(stage_remaining.values()):
        raise RuntimeError(
            f"family/stage balance infeasible: max_flow={total}, expected={expected}"
        )
    quotas: dict[tuple[str, str], int] = {}
    for family in DECISION_FAMILIES:
        node = family_node[family]
        for edge in graph[node]:
            stage = next(
                (name for name, index in stage_node.items() if index == edge.to),
                None,
            )
            if stage is not None:
                quotas[(family, stage)] = edge.original - edge.cap
    return quotas


def _round_robin_select(
    rows: list[dict[str, Any]],
    quota: int,
    used_states: set[str],
    used_records: set[str],
) -> list[dict[str, Any]]:
    by_scenario: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("semantic_scenario_fingerprint", "")),
            int(item.get("trajectory_step", -1)),
            str(item.get("record_id", "")),
        ),
    ):
        state = str(row.get("semantic_public_state_digest", ""))
        record_id = str(row.get("record_id", ""))
        if state and record_id and state not in used_states and record_id not in used_records:
            by_scenario[str(row.get("semantic_scenario_fingerprint", ""))].append(row)
    selected: list[dict[str, Any]] = []
    while by_scenario and len(selected) < quota:
        for scenario in sorted(tuple(by_scenario)):
            bucket = by_scenario[scenario]
            while bucket:
                row = bucket[0]
                state = str(row.get("semantic_public_state_digest", ""))
                record_id = str(row.get("record_id", ""))
                if state in used_states or record_id in used_records:
                    bucket.popleft()
                else:
                    break
            if bucket and len(selected) < quota:
                row = bucket.popleft()
                used_states.add(str(row["semantic_public_state_digest"]))
                used_records.add(str(row["record_id"]))
                selected.append(row)
            if not bucket:
                by_scenario.pop(scenario, None)
    if len(selected) != quota:
        raise RuntimeError(f"insufficient selectable rows: {len(selected)} < {quota}")
    return selected


def _select_balanced_task(
    rows: list[dict[str, Any]],
    *,
    task: str,
    total: int,
    minimum_complete_chains: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_quota = total // len(DECISION_FAMILIES)
    stage_quota = total // len(ATTACK_STAGES)
    if family_quota * len(DECISION_FAMILIES) != total:
        raise ValueError("total must divide evenly across decision families")
    task_rows = [
        row
        for row in rows
        if str(row.get("task_id", "")) == task
        and str(row.get("target_family", "")) in DECISION_FAMILIES
        and str((row.get("generation_audit") or {}).get("attack_stage", ""))
        in ATTACK_STAGES
    ]
    used_states: set[str] = set()
    used_records: set[str] = set()
    selected: list[dict[str, Any]] = []
    reserved_chain_ids: list[str] = []
    family_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    for chain in _complete_chain_rows(task_rows):
        if len(reserved_chain_ids) >= minimum_complete_chains:
            break
        additions = [
            row
            for row in chain
            if str(row.get("semantic_public_state_digest", "")) not in used_states
            and str(row.get("record_id", "")) not in used_records
        ]
        add_family = Counter(str(row["target_family"]) for row in additions)
        add_stage = Counter(
            str((row["generation_audit"] or {})["attack_stage"]) for row in additions
        )
        if any(
            family_counts[family] + add_family[family] > family_quota
            for family in DECISION_FAMILIES
        ) or any(
            stage_counts[stage] + add_stage[stage] > stage_quota
            for stage in ATTACK_STAGES
        ):
            continue
        for row in additions:
            selected.append(row)
            used_states.add(str(row["semantic_public_state_digest"]))
            used_records.add(str(row["record_id"]))
            family_counts[str(row["target_family"])] += 1
            stage_counts[str((row["generation_audit"] or {})["attack_stage"])] += 1
        reserved_chain_ids.append(str(chain[0].get("trajectory_id", "")))
    if len(reserved_chain_ids) != minimum_complete_chains:
        raise RuntimeError(
            f"insufficient complete {task} chains: "
            f"{len(reserved_chain_ids)} < {minimum_complete_chains}"
        )

    family_remaining = {
        family: family_quota - family_counts[family] for family in DECISION_FAMILIES
    }
    stage_remaining = {
        stage: stage_quota - stage_counts[stage] for stage in ATTACK_STAGES
    }
    cell_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        state = str(row.get("semantic_public_state_digest", ""))
        record_id = str(row.get("record_id", ""))
        if state in used_states or record_id in used_records:
            continue
        cell_rows[
            (
                str(row["target_family"]),
                str((row["generation_audit"] or {})["attack_stage"]),
            )
        ].append(row)
    quotas = _max_flow_cell_quotas(
        {key: len(value) for key, value in cell_rows.items()},
        family_remaining,
        stage_remaining,
    )
    for family in DECISION_FAMILIES:
        for stage in ATTACK_STAGES:
            selected.extend(
                _round_robin_select(
                    cell_rows.get((family, stage), []),
                    quotas.get((family, stage), 0),
                    used_states,
                    used_records,
                )
            )
    selected.sort(
        key=lambda row: (
            str(row.get("task_id", "")),
            str(row.get("target_family", "")),
            str((row.get("generation_audit") or {}).get("attack_stage", "")),
            str(row.get("record_id", "")),
        )
    )
    selected_family = Counter(str(row["target_family"]) for row in selected)
    selected_stage = Counter(
        str((row["generation_audit"] or {})["attack_stage"]) for row in selected
    )
    if len(selected) != total:
        raise RuntimeError(f"{task} selection count mismatch: {len(selected)} != {total}")
    if selected_family != Counter({family: family_quota for family in DECISION_FAMILIES}):
        raise RuntimeError(f"{task} family balance mismatch: {selected_family}")
    if selected_stage != Counter({stage: stage_quota for stage in ATTACK_STAGES}):
        raise RuntimeError(f"{task} attack-stage balance mismatch: {selected_stage}")
    return selected, {
        "reserved_complete_chain_count": len(reserved_chain_ids),
        "reserved_complete_chain_ids": reserved_chain_ids,
        "lifecycle_target_family_counts": dict(sorted(selected_family.items())),
        "generation_attack_stage_counts": dict(sorted(selected_stage.items())),
        "family_stage_cell_counts": {
            f"{family}:{stage}": count
            for (family, stage), count in sorted(
                Counter(
                    (
                        str(row["target_family"]),
                        str((row["generation_audit"] or {})["attack_stage"]),
                    )
                    for row in selected
                ).items()
            )
        },
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(rows),
        "task_record_counts": dict(
            sorted(Counter(str(row["task_id"]) for row in rows).items())
        ),
        "lifecycle_target_family_counts": {
            f"{task}:{family}": count
            for (task, family), count in sorted(
                Counter(
                    (str(row["task_id"]), str(row["target_family"])) for row in rows
                ).items()
            )
        },
        "generation_attack_stage_counts": {
            f"{task}:{stage}": count
            for (task, stage), count in sorted(
                Counter(
                    (
                        str(row["task_id"]),
                        str((row.get("generation_audit") or {}).get("attack_stage", "")),
                    )
                    for row in rows
                ).items()
            )
        },
        "causal_target_family_counts": dict(
            sorted(Counter(str(row.get("causal_target_family", "")) for row in rows).items())
        ),
        "unique_record_count": len({str(row["record_id"]) for row in rows}),
        "unique_public_state_count": len(
            {str(row["semantic_public_state_digest"]) for row in rows}
        ),
        "semantic_scenario_count": len(
            {str(row["semantic_scenario_fingerprint"]) for row in rows}
        ),
        "public_contract_coverage": sum(_public_contract(row) for row in rows)
        / max(1, len(rows)),
        "candidate_contract_coverage": sum(_candidate_contract(row) for row in rows)
        / max(1, len(rows)),
        "credibility_metadata_coverage": sum(
            isinstance(row.get("scenario_credibility"), dict) for row in rows
        )
        / max(1, len(rows)),
        "candidate_set_hash_unique_count": len(
            {str(row.get("candidate_set_hash", "")) for row in rows}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument(
        "--blocked-scenario-jsonl", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--blocked-record-jsonl", type=Path, action="append", default=[]
    )
    parser.add_argument("--group-offset", type=int, default=3_100_000)
    parser.add_argument("--seed", type=int, default=2026072301)
    parser.add_argument("--groups-per-shard", type=int, default=12)
    parser.add_argument("--train-shards-per-task", type=int, default=60)
    parser.add_argument("--dev-shards-per-task", type=int, default=12)
    parser.add_argument("--train-records-per-task", type=int, default=2000)
    parser.add_argument("--dev-records-per-task", type=int, default=200)
    parser.add_argument("--train-complete-chains-per-task", type=int, default=20)
    parser.add_argument("--dev-complete-chains-per-task", type=int, default=4)
    parser.add_argument("--teacher-beam-width", type=int, default=20)
    parser.add_argument("--teacher-max-candidates", type=int, default=96)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.audit_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.audit_output}")
    if not 1 <= args.workers <= 35:
        raise ValueError("workers must be in [1, 35]")

    protocol = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    if protocol.get("kind") != "v11_three_expert_router_preregistration":
        raise RuntimeError("invalid three-expert Router preregistration")
    if protocol.get("status") != "frozen_before_data_generation":
        raise RuntimeError("protocol was not frozen before data generation")
    if (protocol.get("forbidden") or {}).get("formal_three_round_coevolution") is not True:
        raise RuntimeError("formal three-round stop condition is missing")

    work = [
        (split, task, shard)
        for split, shard_count in (
            ("train", args.train_shards_per_task),
            ("dev", args.dev_shards_per_task),
        )
        for task in TASKS
        for shard in range(shard_count)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _build_shard,
                split,
                task,
                shard,
                group_offset=args.group_offset,
                groups_per_shard=args.groups_per_shard,
                seed=args.seed,
                teacher_beam_width=args.teacher_beam_width,
                teacher_max_candidates=args.teacher_max_candidates,
            )
            for split, task, shard in work
        ]
        components = [future.result() for future in futures]
    if not all(manifest.get("accepted") is True for _, manifest in components):
        raise RuntimeError("at least one native T1/T2-v2 source shard failed acceptance")

    blocked, blocked_hashes = _blocked_fingerprints(
        scenario_paths=args.blocked_scenario_jsonl,
        record_paths=args.blocked_record_jsonl,
    )
    pools: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for (split, _, _), (rows, _) in zip(work, components, strict=True):
        pools[split].extend(
            row
            for row in rows
            if str(row.get("semantic_scenario_fingerprint", "")) not in blocked
        )

    train: list[dict[str, Any]] = []
    train_selection: dict[str, Any] = {}
    for task in TASKS:
        selected, selection = _select_balanced_task(
            pools["train"],
            task=task,
            total=args.train_records_per_task,
            minimum_complete_chains=args.train_complete_chains_per_task,
        )
        train.extend(selected)
        train_selection[task] = selection
    train_scenarios = {str(row["semantic_scenario_fingerprint"]) for row in train}
    train_states = {str(row["semantic_public_state_digest"]) for row in train}
    train_records = {str(row["record_id"]) for row in train}

    dev_pool = [
        row
        for row in pools["dev"]
        if str(row.get("semantic_scenario_fingerprint", "")) not in train_scenarios
        and str(row.get("semantic_public_state_digest", "")) not in train_states
        and str(row.get("record_id", "")) not in train_records
    ]
    dev: list[dict[str, Any]] = []
    dev_selection: dict[str, Any] = {}
    for task in TASKS:
        selected, selection = _select_balanced_task(
            dev_pool,
            task=task,
            total=args.dev_records_per_task,
            minimum_complete_chains=args.dev_complete_chains_per_task,
        )
        dev.extend(selected)
        dev_selection[task] = selection
    train.sort(key=lambda row: (str(row["task_id"]), str(row["record_id"])))
    dev.sort(key=lambda row: (str(row["task_id"]), str(row["record_id"])))

    dev_scenarios = {str(row["semantic_scenario_fingerprint"]) for row in dev}
    dev_states = {str(row["semantic_public_state_digest"]) for row in dev}
    dev_records = {str(row["record_id"]) for row in dev}
    train_coverage = _coverage(train)
    dev_coverage = _coverage(dev)
    expected_train = len(TASKS) * args.train_records_per_task
    expected_dev = len(TASKS) * args.dev_records_per_task
    expected_train_family = args.train_records_per_task // len(DECISION_FAMILIES)
    expected_dev_family = args.dev_records_per_task // len(DECISION_FAMILIES)
    checks = {
        "train_exact_4000": len(train) == expected_train == 4000,
        "dev_exact_400": len(dev) == expected_dev == 400,
        "train_task_exact": train_coverage["task_record_counts"]
        == {"T1": 2000, "T2": 2000},
        "dev_task_exact": dev_coverage["task_record_counts"]
        == {"T1": 200, "T2": 200},
        "train_family_exact": all(
            train_coverage["lifecycle_target_family_counts"].get(f"{task}:{family}") == expected_train_family
            for task in TASKS
            for family in DECISION_FAMILIES
        ),
        "dev_family_exact": all(
            dev_coverage["lifecycle_target_family_counts"].get(f"{task}:{family}") == expected_dev_family
            for task in TASKS
            for family in DECISION_FAMILIES
        ),
        "train_attack_stage_exact": all(
            train_coverage["generation_attack_stage_counts"].get(f"{task}:{stage}") == expected_train_family
            for task in TASKS
            for stage in ATTACK_STAGES
        ),
        "dev_attack_stage_exact": all(
            dev_coverage["generation_attack_stage_counts"].get(f"{task}:{stage}") == expected_dev_family
            for task in TASKS
            for stage in ATTACK_STAGES
        ),
        "train_complete_chains": all(
            train_selection[task]["reserved_complete_chain_count"]
            >= args.train_complete_chains_per_task
            for task in TASKS
        ),
        "dev_complete_chains": all(
            dev_selection[task]["reserved_complete_chain_count"]
            >= args.dev_complete_chains_per_task
            for task in TASKS
        ),
        "train_records_unique": train_coverage["unique_record_count"] == expected_train,
        "dev_records_unique": dev_coverage["unique_record_count"] == expected_dev,
        "train_public_states_unique": train_coverage["unique_public_state_count"]
        == expected_train,
        "dev_public_states_unique": dev_coverage["unique_public_state_count"]
        == expected_dev,
        "train_dev_record_disjoint": not (train_records & dev_records),
        "train_dev_state_disjoint": not (train_states & dev_states),
        "train_dev_scenario_disjoint": not (train_scenarios & dev_scenarios),
        "frozen_or_expert_data_scenario_disjoint": not (
            (train_scenarios | dev_scenarios) & blocked
        ),
        "public_only_model_inputs": train_coverage["public_contract_coverage"] == 1.0
        and dev_coverage["public_contract_coverage"] == 1.0,
        "candidate_contract": train_coverage["candidate_contract_coverage"] == 1.0
        and dev_coverage["candidate_contract_coverage"] == 1.0,
        "candidate_set_hashes_present": train_coverage[
            "candidate_set_hash_unique_count"
        ]
        == expected_train
        and dev_coverage["candidate_set_hash_unique_count"] == expected_dev,
        "credibility_metadata_complete": train_coverage[
            "credibility_metadata_coverage"
        ]
        == 1.0
        and dev_coverage["credibility_metadata_coverage"] == 1.0,
    }
    accepted = all(checks.values())

    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "train.jsonl"
    dev_path = args.output_dir / "dev.jsonl"
    _atomic_jsonl(train_path, train)
    _atomic_jsonl(dev_path, dev)
    common = {
        "schema_version": 1,
        "kind": "v11_native_t12_v2_candidate_dataset",
        "created_at": utc_now(),
        "accepted": accepted,
        "checks": checks,
        "failures": [key for key, passed in checks.items() if not passed],
        "protocol_manifest_sha256": sha256_file(args.protocol_manifest),
        "generation_seed": args.seed,
        "group_offset": args.group_offset,
        "decision_families": list(DECISION_FAMILIES),
        "attack_stage_strata": list(ATTACK_STAGES),
        "attack_stage_is_generation_audit_only": True,
        "task_label_in_model_input": False,
        "hidden_state_in_model_input": False,
        "teacher_q_in_model_input": False,
        "primary_supervision_target": "public_lifecycle_family_then_teacher_q_v1",
        "unconstrained_teacher_scores_retained": True,
        "candidate_set_hash_algorithm": "sha256-canonical-json-v1",
        "blocked_source_hashes": blocked_hashes,
        "full_scale_router_auto_start": False,
        "formal_three_rounds_auto_start": False,
    }
    train_manifest = {
        **common,
        "protocol_role": "supervised_train",
        "data_source": "v11_t12_native_v2_train",
        "candidate_sets_sha256": sha256_file(train_path),
        "selection": train_selection,
        **train_coverage,
    }
    dev_manifest = {
        **common,
        "protocol_role": "heldout_dev",
        "data_source": "v11_t12_native_v2_dev",
        "candidate_sets_sha256": sha256_file(dev_path),
        "selection": dev_selection,
        **dev_coverage,
    }
    atomic_write_json(args.output_dir / "train_manifest.json", train_manifest)
    atomic_write_json(args.output_dir / "dev_manifest.json", dev_manifest)
    split_manifest = {
        **common,
        "kind": "v11_native_t12_v2_frozen_split",
        "train_records": len(train),
        "dev_records": len(dev),
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "train_manifest_sha256": sha256_file(args.output_dir / "train_manifest.json"),
        "dev_manifest_sha256": sha256_file(args.output_dir / "dev_manifest.json"),
        "semantic_train_dev_overlap": len(train_scenarios & dev_scenarios),
        "blocked_scenario_overlap": len(
            (train_scenarios | dev_scenarios) & blocked
        ),
        "component_count": len(components),
        "component_manifest_hashes": [
            _stable_hash(manifest) for _, manifest in components
        ],
    }
    atomic_write_json(args.output_dir / "manifest.json", split_manifest)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.audit_output,
        {
            "schema_version": 1,
            "kind": "v11_t12_native_v2_data_audit",
            "created_at": utc_now(),
            "accepted": accepted,
            "checks": checks,
            "failures": [key for key, passed in checks.items() if not passed],
            "split_manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
            "protocol_manifest_sha256": sha256_file(args.protocol_manifest),
            "teacher_directory_modified": False,
            "ecrg_enabled": False,
            "dagger_enabled": False,
            "formal_three_rounds_started": False,
        },
    )
    print(json.dumps(split_manifest, ensure_ascii=False, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
