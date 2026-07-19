#!/usr/bin/env python3
"""Expose a ranker checkpoint through the shared DCA/VDA round lineage contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import (
    SCHEMA_VERSION,
    PROTOCOL_VERSION,
    atomic_write_json,
    model_identity,
    sha256_file,
    sha256_tree,
    utc_now,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--training-data-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    ranker = json.loads(args.ranker_manifest.read_text(encoding="utf-8"))
    adapter = Path(ranker["adapter_path"])
    score_head = Path(ranker["score_head_path"])
    heads = Path(ranker.get("heads_path", ranker["score_head_path"]))
    if sha256_tree(adapter) != ranker["adapter_sha256"]:
        raise RuntimeError("ranker adapter hash mismatch")
    if sha256_file(score_head) != ranker["score_head_sha256"]:
        raise RuntimeError("ranker score head hash mismatch")
    if sha256_file(heads) != ranker.get("heads_sha256", ranker["score_head_sha256"]):
        raise RuntimeError("ranker multi-head hash mismatch")
    if args.round_index > 0 and args.parent_manifest is None:
        raise ValueError("nonzero ranker round requires a parent manifest")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "kind": "checkpoint",
        "role": "vda",
        "policy_architecture": "direct_candidate_ranker_v1",
        "backbone": "qwen3.5-4b",
        "round": args.round_index,
        "created_at": utc_now(),
        "seed": args.seed,
        "base_model": model_identity(args.model_path),
        "parent_manifest": str(args.parent_manifest.resolve())
        if args.parent_manifest
        else None,
        "parent_manifest_sha256": sha256_file(args.parent_manifest)
        if args.parent_manifest
        else None,
        "training_data_manifest": str(args.training_data_manifest.resolve())
        if args.training_data_manifest
        else None,
        "training_data_manifest_sha256": sha256_file(args.training_data_manifest)
        if args.training_data_manifest
        else None,
        "checkpoint_path": str(Path(ranker["adapter_path"]).resolve().parent),
        "adapter_path": str(adapter.resolve()),
        "adapter_sha256": sha256_tree(adapter),
        "score_head_path": str(score_head.resolve()),
        "score_head_sha256": sha256_file(score_head),
        "heads_path": str(heads.resolve()),
        "heads_sha256": sha256_file(heads),
        "ranker_manifest": str(args.ranker_manifest.resolve()),
        "ranker_manifest_sha256": sha256_file(args.ranker_manifest),
        "status": "trained" if args.round_index else "candidate_bootstrap",
        "training_config": {
            "policy_architecture": "direct_candidate_ranker_v1",
            "format_generated_by_model": False,
            "active_probe_required": True,
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
