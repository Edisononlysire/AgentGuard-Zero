#!/usr/bin/env python3
"""Build hidden-free Teacher-vs-Observe action preferences from fixed data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.public_teacher import (
    compact_wire_json,
    enumerate_public_candidates,
)
from agentguard_zero.schemas.action_schema_v4 import parse_action_json_v4
from agentguard_zero.world.public_projector import assert_public


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def public_observation(prompt: str) -> dict:
    marker = "\nCurrent decision instance:"
    if marker not in prompt:
        raise ValueError("preference prompt is missing its public context")
    context = json.loads(prompt.split(marker, 1)[1])
    assert_public(context)
    return dict(context["observation"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=500)
    parser.add_argument("--min-advantage-over-observe", type=float, default=0.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    parent_path = args.teacher_dir / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("accepted") is not True:
        raise RuntimeError("Teacher dataset is not accepted")
    rows = pd.read_parquet(args.teacher_dir / "bootstrap_sft.parquet")
    audits = read_jsonl(args.teacher_dir / "teacher_selection_audit.jsonl")
    audit_by_id = {str(row.get("record_id", "")): row for row in audits}
    if len(audit_by_id) != len(audits):
        raise RuntimeError("Teacher audit contains duplicate record IDs")
    pairs = []
    category_counts: Counter[str] = Counter()
    for row in rows.to_dict(orient="records"):
        if str(row.get("action_category")) == "observe":
            continue
        record_id = str(row["record_id"])
        audit = audit_by_id.get(record_id)
        if audit is None:
            raise RuntimeError("Teacher preference row has no matching audit")
        advantage = float(audit.get("advantage_over_observe", 0.0))
        if advantage <= args.min_advantage_over_observe + 1.0e-12:
            continue
        chosen = str(row["target"])
        observe_candidates = [
            candidate
            for candidate in enumerate_public_candidates(
                public_observation(str(row["prompt"])), max_candidates=96
            )
            if candidate.category == "observe"
        ]
        if len(observe_candidates) != 1:
            raise RuntimeError("public state does not have exactly one Observe candidate")
        observe = compact_wire_json(observe_candidates[0].packet)
        if chosen == observe:
            continue
        _, chosen_valid, chosen_reason = parse_action_json_v4(chosen)
        _, rejected_valid, rejected_reason = parse_action_json_v4(observe)
        if not chosen_valid or not rejected_valid:
            raise RuntimeError(
                f"invalid preference pair: {chosen_reason}/{rejected_reason}"
            )
        record_id = hashlib.sha256(
            f"preference:{row['record_id']}:{chosen}:{observe}".encode("utf-8")
        ).hexdigest()
        pairs.append(
            {
                "record_id": record_id,
                "source_record_id": str(row["record_id"]),
                "prompt": str(row["prompt"]),
                "chosen": chosen,
                "rejected": observe,
                "chosen_action_category": str(row["action_category"]),
                "rejected_action_category": "observe",
            }
        )
        category_counts[str(row["action_category"])] += 1
    unique = len({row["record_id"] for row in pairs}) == len(pairs)
    accepted = len(pairs) >= args.min_pairs and unique
    manifest = {
        "schema_version": 1,
        "kind": "recovery_teacher_vs_observe_preferences",
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "pair_count": len(pairs),
        "minimum_pair_count": args.min_pairs,
        "minimum_advantage_over_observe": args.min_advantage_over_observe,
        "chosen_action_category_counts": dict(category_counts),
        "unique_record_ids": unique,
        "source_scenario_count": parent.get("source_scenario_count"),
        "source_scenarios_sha256": parent.get("source_scenarios_sha256"),
        "teacher_manifest_sha256": sha256(parent_path),
        "counterfactual_worlds_training_visible": False,
        "hidden_state_in_prompt_or_response": False,
        "preference_semantics": (
            "positive_robust_teacher_advantage_action_preferred_to_public_observe"
        ),
    }
    args.output_dir.mkdir(parents=True)
    parquet = args.output_dir / "preferences.parquet"
    manifest_path = args.output_dir / "manifest.json"
    pd.DataFrame(pairs).to_parquet(parquet, index=False)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256(path) for path in (parquet, manifest_path)}
    (args.output_dir / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
