#!/usr/bin/env python3
"""Build Gate-A bootstrap SFT records; this script never updates a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.bootstrap_data import build_bootstrap_records
from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION, RecoveryConfig
from agentguard_zero.recovery.public_teacher import (
    PublicStateRobustTeacher,
    public_state_digest,
)
from agentguard_zero.recovery.source_counterfactuals import counterfactual_groups


def load_accepted_stage0(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = payload.get("verdict", {}) if isinstance(payload, dict) else {}
    if payload.get("kind") != "recovery_stage0_audit":
        raise RuntimeError("Stage-0 parent has the wrong artifact kind")
    if payload.get("protocol_version") != RECOVERY_PROTOCOL_VERSION:
        raise RuntimeError("Stage-0 parent has the wrong recovery protocol")
    if payload.get("accepted") is not True or payload.get("status") != "accepted":
        raise RuntimeError("Stage-0 parent is not an accepted artifact")
    if (
        verdict.get("gate") != "stage0_fixed_policy"
        or verdict.get("accepted") is not True
    ):
        raise RuntimeError("Stage-0 parent did not pass")
    if verdict.get("next_stage") != "bootstrap_data_build_and_audit":
        raise RuntimeError("Stage-0 parent does not unlock Bootstrap data build")
    if payload.get("next_stage") != verdict.get("next_stage"):
        raise RuntimeError("Stage-0 parent next-stage fields disagree")
    if (
        int(payload.get("model_calls", -1)) != 0
        or int(payload.get("parameter_updates", -1)) != 0
    ):
        raise RuntimeError("Stage-0 parent is not model-free")
    return payload


def _scenario_from_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("scenario", "scenario_json"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    extra = row.get("extra_info")
    if isinstance(extra, str) and extra.strip():
        extra = json.loads(extra)
    if isinstance(extra, dict):
        value = extra.get("scenario")
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    if row.get("protocol_version") == "tmcd-v2":
        return row
    raise ValueError("row does not contain a TMCD scenario")


def load_scenarios(path: Path) -> list[dict[str, Any]]:
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
            rows = [scenario for group in payload["groups"] for scenario in group]
        else:
            rows = payload.get("scenarios", [])
    return [_scenario_from_row(dict(row)) for row in rows]


def group_public_worlds(
    scenarios: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        env = instantiate_scenario(scenario)
        grouped[public_state_digest(env.observe())].append(scenario)
    singletons = [key for key, values in grouped.items() if len(values) < 2]
    if singletons:
        raise ValueError(
            f"{len(singletons)} initial public states lack counterfactual worlds"
        )
    return [grouped[key] for key in sorted(grouped)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--stage0-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-scenarios", type=int, default=400)
    parser.add_argument("--min-records", type=int, default=2_000)
    parser.add_argument("--max-records", type=int, default=3_000)
    parser.add_argument("--derive-counterfactual-worlds", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--defer-global-distribution-gates", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    stage0_parent = (
        load_accepted_stage0(args.stage0_audit)
        if args.stage0_audit is not None
        else None
    )
    source_sha256 = _sha256(args.scenarios)
    if (
        stage0_parent is not None
        and source_sha256 == stage0_parent.get("scenario_source_sha256")
    ):
        raise RuntimeError("Bootstrap scenarios must be disjoint from Stage 0")
    scenarios = load_scenarios(args.scenarios)
    if len(scenarios) != args.expected_scenarios:
        raise ValueError(
            f"expected {args.expected_scenarios} scenarios, got {len(scenarios)}"
        )
    all_groups = (
        counterfactual_groups(scenarios)
        if args.derive_counterfactual_worlds
        else group_public_worlds(scenarios)
    )
    selected_groups = all_groups[args.shard_index :: args.shard_count]
    config = RecoveryConfig()
    teacher = PublicStateRobustTeacher(
        advantage_delta=config.teacher.advantage_delta,
        min_worlds_per_public_state=config.teacher.min_worlds_per_public_state,
        beam_width=config.teacher.beam_width,
        max_candidates=config.teacher.max_candidates,
    )
    result = build_bootstrap_records(
        selected_groups,
        teacher=teacher,
        max_records=args.max_records,
    )
    record_count_ok = args.min_records <= len(result.train_records) <= args.max_records
    unique_ratio_ok = (
        float(result.manifest.get("unique_prompt_target_ratio", 0.0)) + 1.0e-12
        >= config.bootstrap_sft.unique_prompt_target_ratio_min
    )
    rank_state_count = int(
        result.manifest.get("teacher_core_rank_correlation_state_count", 0)
    )
    rank_correlation = result.manifest.get("teacher_core_rank_correlation_mean")
    rank_gate_ok = (
        rank_state_count >= 200
        and isinstance(rank_correlation, (int, float))
        and float(rank_correlation) > config.teacher.core_rank_correlation_min_exclusive
    )
    result.manifest["record_count_gate"] = {
        "minimum": args.min_records,
        "maximum": args.max_records,
        "accepted": record_count_ok,
    }
    result.manifest["unique_prompt_target_gate"] = {
        "minimum": config.bootstrap_sft.unique_prompt_target_ratio_min,
        "actual": result.manifest.get("unique_prompt_target_ratio"),
        "accepted": unique_ratio_ok,
    }
    result.manifest["teacher_core_rank_correlation_gate"] = {
        "minimum_states": 200,
        "minimum_exclusive": config.teacher.core_rank_correlation_min_exclusive,
        "actual_states": rank_state_count,
        "actual": rank_correlation,
        "accepted": rank_gate_ok,
    }
    result.manifest["protocol_version"] = RECOVERY_PROTOCOL_VERSION
    result.manifest["source_scenarios_sha256"] = source_sha256
    result.manifest["source_scenario_count"] = len(selected_groups)
    result.manifest["audit_world_count"] = sum(
        len(group) for group in selected_groups
    )
    result.manifest["counterfactual_worlds_training_visible"] = False
    result.manifest["derived_counterfactual_worlds"] = bool(
        args.derive_counterfactual_worlds
    )
    result.manifest["shard_index"] = args.shard_index
    result.manifest["shard_count"] = args.shard_count
    result.manifest["stage0_parent"] = (
        str(args.stage0_audit.resolve()) if args.stage0_audit else None
    )
    result.manifest["stage0_parent_sha256"] = (
        _sha256(args.stage0_audit) if args.stage0_audit else None
    )
    result.manifest["stage0_parent_snapshot"] = stage0_parent
    result.manifest["execution_authorization"] = (
        "accepted_stage0" if stage0_parent is not None else "direct_user_fixed500_vda"
    )
    result.manifest["recovery_config"] = config.to_dict()
    if args.defer_global_distribution_gates:
        deferred_prefixes = (
            "single_action_category_above_40pct",
            "missing_action_support:",
        )
        remaining_failures = [
            failure
            for failure in result.manifest.get("failures", [])
            if not failure.startswith(deferred_prefixes)
        ]
        result.manifest["shard_deferred_distribution_failures"] = [
            failure
            for failure in result.manifest.get("failures", [])
            if failure.startswith(deferred_prefixes)
        ]
        base_accepted = not remaining_failures
    else:
        base_accepted = bool(result.manifest.get("accepted"))
    result.manifest["accepted"] = bool(
        base_accepted and record_count_ok and unique_ratio_ok and rank_gate_ok
    )
    result.manifest["status"] = (
        "accepted" if result.manifest["accepted"] else "rejected"
    )
    result.manifest["next_stage"] = (
        "await_explicit_gate_a_sft_review"
        if result.manifest["accepted"]
        else "stop_and_repair_bootstrap_data"
    )

    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "bootstrap_sft.parquet"
    audit_path = args.output_dir / "teacher_selection_audit.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    pd.DataFrame(result.train_records).to_parquet(train_path, index=False)
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in result.audit_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path.write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: _sha256(path) for path in (train_path, audit_path, manifest_path)
    }
    (args.output_dir / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.manifest, ensure_ascii=False, sort_keys=True))
    return 0 if result.manifest["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
