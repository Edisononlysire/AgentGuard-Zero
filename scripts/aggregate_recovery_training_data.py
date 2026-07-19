#!/usr/bin/env python3
"""Aggregate accepted bootstrap and DAgger records for one corrective SFT pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.bootstrap_data import audit_bootstrap_records
from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION, RecoveryConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--dagger-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    parents = [args.bootstrap_dir, args.dagger_dir]
    manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in parents
    ]
    if any(row.get("accepted") is not True for row in manifests):
        raise RuntimeError("aggregation parent is not accepted")
    source_hashes = {row.get("source_scenarios_sha256") for row in manifests}
    if len(source_hashes) != 1:
        raise RuntimeError("bootstrap and DAgger source hashes differ")
    # DAgger deliberately revisits some bootstrap public states. A duplicated
    # prompt with two targets is ambiguous SFT supervision, so the fresh DAgger
    # correction replaces (rather than appends to) its bootstrap counterpart.
    # Unrevisited bootstrap states remain as replay support.
    by_prompt: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {}
    replacement_count = 0
    for parent_kind, path in zip(("bootstrap", "dagger"), parents):
        rows = pd.read_parquet(path / "bootstrap_sft.parquet").to_dict(
            orient="records"
        )
        audits = read_jsonl(path / "teacher_selection_audit.jsonl")
        audit_by_id = {str(row.get("record_id", "")): row for row in audits}
        if len(audit_by_id) != len(audits):
            raise RuntimeError(f"duplicate audit record IDs in {path}")
        for row in rows:
            record_id = str(row.get("record_id", ""))
            if record_id not in audit_by_id:
                raise RuntimeError(f"training row has no matching audit in {path}")
            prompt = str(row.get("prompt", ""))
            if parent_kind == "dagger" and prompt in by_prompt:
                replacement_count += 1
            by_prompt[prompt] = (row, audit_by_id[record_id], parent_kind)
    ordered = sorted(by_prompt.values(), key=lambda item: str(item[0]["record_id"]))
    train_rows = [item[0] for item in ordered]
    audit_rows = [item[1] for item in ordered]
    manifest = audit_bootstrap_records(train_rows, audit_rows)
    manifest.update(
        {
            "schema_version": 1,
            "kind": "recovery_bootstrap_plus_single_dagger_dataset",
            "protocol_version": RECOVERY_PROTOCOL_VERSION,
            "source_scenarios_sha256": next(iter(source_hashes)),
            "source_scenario_count": 500,
            "counterfactual_worlds_training_visible": False,
            "human_action_labels": 0,
            "parents": {
                str(path): sha256(path / "manifest.json") for path in parents
            },
            "aggregation_policy": {
                "name": "single_dagger_correction_replaces_same_public_prompt",
                "one_target_per_unique_public_prompt": True,
                "dagger_replacement_count": replacement_count,
                "output_record_count": len(train_rows),
            },
            "recovery_config": RecoveryConfig().to_dict(),
        }
    )
    manifest["accepted"] = bool(manifest.get("accepted"))
    manifest["status"] = "accepted" if manifest["accepted"] else "rejected"
    manifest["teacher_core_rank_correlation_gate"] = {
        "accepted": bool(
            manifest.get("teacher_core_rank_correlation_state_count", 0) >= 200
            and float(manifest.get("teacher_core_rank_correlation_mean", 0.0))
            > 0.5
        )
    }
    args.output_dir.mkdir(parents=True)
    train = args.output_dir / "bootstrap_sft.parquet"
    audit = args.output_dir / "teacher_selection_audit.jsonl"
    out_manifest = args.output_dir / "manifest.json"
    pd.DataFrame(train_rows).to_parquet(train, index=False)
    with audit.open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    out_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256(path) for path in (train, audit, out_manifest)}
    (args.output_dir / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
