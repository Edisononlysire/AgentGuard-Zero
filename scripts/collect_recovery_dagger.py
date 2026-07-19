#!/usr/bin/env python3
"""Collect the single frozen-policy DAgger correction dataset."""

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
from agentguard_zero.recovery.dagger import collect_dagger_records
from agentguard_zero.recovery.model_policy import RecoveryModelPolicy
from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION, RecoveryConfig
from agentguard_zero.recovery.public_teacher import public_state_digest
from agentguard_zero.recovery.source_counterfactuals import (
    counterfactual_groups,
    load_source_scenarios,
)
from agentguard_zero.training.coevolution import sha256_file, sha256_tree


def _scenario_from_row(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("scenario", row)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("DAgger input row does not contain a scenario")
    return value


def _load_groups(path: Path) -> list[list[dict[str, Any]]]:
    if path.suffix == ".parquet":
        rows = [
            _scenario_from_row(row)
            for row in pd.read_parquet(path).to_dict(orient="records")
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("groups"), list):
            return [[dict(row) for row in group] for group in payload["groups"]]
        source = payload if isinstance(payload, list) else payload.get("scenarios", [])
        rows = [_scenario_from_row(dict(row)) for row in source]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[public_state_digest(instantiate_scenario(row).observe())].append(row)
    if any(len(group) < 2 for group in grouped.values()):
        raise RuntimeError("DAgger input contains unmatched public states")
    return [grouped[key] for key in sorted(grouped)]


def _load_adapter_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != RECOVERY_PROTOCOL_VERSION:
        raise RuntimeError("adapter manifest has the wrong recovery protocol")
    if payload.get("status") not in {
        "trained_pending_gate_a",
        "trained_pending_k1_evaluation",
    }:
        raise RuntimeError("adapter is not a frozen trained recovery checkpoint")
    adapter = Path(str(payload.get("adapter_path", "")))
    if not adapter.is_dir() or sha256_tree(adapter) != payload.get("adapter_sha256"):
        raise RuntimeError("adapter hash verification failed")
    return payload, adapter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_review_approval(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("review approval must be a JSON object")
    if payload.get("kind") != "recovery_execution_approval":
        raise RuntimeError("review approval has the wrong kind")
    if payload.get("protocol_version") != RECOVERY_PROTOCOL_VERSION:
        raise RuntimeError("review approval has the wrong recovery protocol")
    stages = payload.get("approved_stages")
    if (
        payload.get("status") != "approved"
        or not isinstance(stages, list)
        or "single_dagger_collection" not in stages
    ):
        raise RuntimeError("single DAgger collection remains review-locked")
    if not str(payload.get("reviewer", "")).strip():
        raise RuntimeError("review approval is missing reviewer identity")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--adapter-manifest", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--review-approval", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--derive-counterfactual-worlds", action="store_true")
    parser.add_argument("--expected-source-scenarios", type=int, default=500)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--min-records", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--defer-global-distribution-gates", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if (args.selection_manifest is None) == (args.adapter_manifest is None):
        raise ValueError("provide exactly one selection or adapter manifest")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    review_approval = (
        _load_review_approval(args.review_approval)
        if args.review_approval is not None
        else {"reviewer": "direct_user_fixed500_vda"}
    )
    if args.adapter_manifest is not None:
        selection, adapter = _load_adapter_manifest(args.adapter_manifest)
        selected_adapter_sha256 = str(selection["adapter_sha256"])
        parent_manifest = args.adapter_manifest
    else:
        selection = json.loads(
            args.selection_manifest.read_text(encoding="utf-8")
        )
        if (
            selection.get("protocol_version") != RECOVERY_PROTOCOL_VERSION
            or selection.get("status") != "selected_pending_single_dagger"
            or selection.get("accepted") is not True
        ):
            raise RuntimeError("Gate-A selection manifest does not unlock DAgger")
        adapter = Path(str(selection["selected_adapter_path"]))
        selected_adapter_sha256 = str(selection["selected_adapter_sha256"])
        if sha256_tree(adapter) != selected_adapter_sha256:
            raise RuntimeError("selected Gate-A adapter hash changed before DAgger")
        parent_manifest = args.selection_manifest
    all_groups = (
        counterfactual_groups(load_source_scenarios(args.scenarios))
        if args.derive_counterfactual_worlds
        else _load_groups(args.scenarios)
    )
    if len(all_groups) != args.expected_source_scenarios:
        raise RuntimeError(
            f"DAgger requires {args.expected_source_scenarios} fixed source "
            f"scenarios, got {len(all_groups)}"
        )
    groups = all_groups[args.shard_index :: args.shard_count]
    cfg = RecoveryConfig().dagger
    policy = RecoveryModelPolicy(
        model_path=args.model_path,
        adapter_path=adapter,
        device=args.device,
        max_new_tokens=320,
    )
    result = collect_dagger_records(
        groups,
        model_policy=policy,
        max_records=args.max_records or cfg.correction_records_max,
    )
    record_min = args.min_records or cfg.correction_records_min
    record_max = args.max_records or cfg.correction_records_max
    record_gate = (
        record_min
        <= len(result.train_records)
        <= record_max
    )
    result.manifest.update(
        {
            "protocol_version": RECOVERY_PROTOCOL_VERSION,
            "selection_manifest": str(parent_manifest.resolve()),
            "selection_manifest_sha256": sha256_file(parent_manifest),
            "selected_adapter_sha256": selected_adapter_sha256,
            "source_scenarios_sha256": sha256_file(args.scenarios),
            "source_scenario_count": len(groups),
            "audit_world_count": sum(len(group) for group in groups),
            "counterfactual_worlds_training_visible": False,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "review_approval": (
                str(args.review_approval.resolve()) if args.review_approval else None
            ),
            "review_approval_sha256": (
                sha256_file(args.review_approval) if args.review_approval else None
            ),
            "reviewer": review_approval["reviewer"],
            "record_count_gate": {
                "minimum": record_min,
                "maximum": record_max,
                "accepted": record_gate,
            },
        }
    )
    if args.defer_global_distribution_gates:
        deferred_prefixes = (
            "single_action_category_above_40pct",
            "missing_action_support:",
        )
        remaining = [
            failure
            for failure in result.manifest.get("failures", [])
            if not failure.startswith(deferred_prefixes)
        ]
        result.manifest["shard_deferred_distribution_failures"] = [
            failure
            for failure in result.manifest.get("failures", [])
            if failure.startswith(deferred_prefixes)
        ]
        result.manifest["accepted"] = bool(not remaining and record_gate)
    else:
        result.manifest["accepted"] = bool(
            result.manifest.get("accepted") and record_gate
        )
    result.manifest["status"] = (
        "accepted_pending_short_sft" if result.manifest["accepted"] else "rejected"
    )
    args.output_dir.mkdir(parents=True)
    train = args.output_dir / "dagger_correction.parquet"
    audit = args.output_dir / "teacher_relabel_audit.jsonl"
    manifest = args.output_dir / "manifest.json"
    pd.DataFrame(result.train_records).to_parquet(train, index=False)
    with audit.open("w", encoding="utf-8") as handle:
        for row in result.audit_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest.write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    hashes = {path.name: _sha256(path) for path in (train, audit, manifest)}
    (args.output_dir / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.manifest, ensure_ascii=False, sort_keys=True))
    return 0 if result.manifest["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
