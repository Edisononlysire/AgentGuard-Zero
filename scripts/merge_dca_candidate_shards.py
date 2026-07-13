#!/usr/bin/env python3
"""Merge and validate data-parallel DCA candidate shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, utc_now


def merge_candidate_shards(paths: list[Path], expected_count: int) -> dict[str, Any]:
    if not paths or expected_count <= 0:
        raise ValueError("candidate shards and a positive expected count are required")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    shard_count = len(shards)
    source_hashes = {
        str(shard.get("source_dca_checkpoint_manifest_sha256", "")) for shard in shards
    }
    if "" in source_hashes or len(source_hashes) != 1:
        raise ValueError("candidate shards do not share one DCA checkpoint manifest hash")
    indices = {int(shard.get("shard_index", -1)) for shard in shards}
    declared_counts = {int(shard.get("num_shards", -1)) for shard in shards}
    if indices != set(range(shard_count)) or declared_counts != {shard_count}:
        raise ValueError("candidate shard indices or declared shard count are incomplete")

    records: list[dict[str, Any]] = []
    for shard in shards:
        if int(shard.get("num_candidates_requested", -1)) != expected_count:
            raise ValueError("candidate shard global request count mismatch")
        records.extend(shard.get("candidates", []) or [])
    records.sort(key=lambda item: int(item.get("candidate_index", -1)))
    record_indices = [int(item.get("candidate_index", -1)) for item in records]
    if record_indices != list(range(expected_count)):
        raise ValueError("candidate shards do not cover every global candidate index exactly once")

    seen: set[str] = set()
    duplicates = 0
    for record in records:
        fingerprint = str(record.get("scenario_fingerprint", ""))
        duplicate = not fingerprint or fingerprint in seen
        record["duplicate"] = duplicate
        duplicates += int(duplicate)
        seen.add(fingerprint)

    first = shards[0]
    return {
        "schema_version": 1,
        "kind": "dca_candidate_pool",
        "created_at": utc_now(),
        "seed": first.get("seed"),
        "backbone": first.get("backbone"),
        "source_dca_round": first.get("source_dca_round"),
        "source_dca_checkpoint_manifest": first.get("source_dca_checkpoint_manifest"),
        "source_dca_checkpoint_manifest_sha256": next(iter(source_hashes)),
        "num_candidates_requested": expected_count,
        "num_candidates_generated": len(records),
        "num_parse_ok": sum(bool(item.get("parse_ok", False)) for item in records),
        "num_all_checks_ok": sum(
            bool((item.get("checks", {}) or {}).get("all_ok", False)) for item in records
        ),
        "num_duplicates": duplicates,
        "generation_num_shards": shard_count,
        "candidate_shards": [str(path.resolve()) for path in paths],
        "candidates": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = merge_candidate_shards(
        [Path(value).resolve() for value in args.shards], args.expected_count
    )
    atomic_write_json(args.output, result)
    print(
        json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
