#!/usr/bin/env python3
"""Fail closed when a TMCD-v2 VDA pool violates public-data or split invariants."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import read_json, scenario_fingerprint
from agentguard_zero.protocol import TMCD_RELEASE_REVISION
from agentguard_zero.world.public_projector import forbidden_public_paths


TASK_IDS = ("T1", "T2", "T3", "T4")
INSTANCE_ID = re.compile(r"^tmcd-[0-9a-f]{16}$")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _public_context(problem: Any) -> dict[str, Any]:
    messages = list(problem) if not isinstance(problem, list) else problem
    text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )
    marker = "Current decision instance:"
    if marker not in text:
        raise ValueError("VDA prompt is missing Current decision instance")
    loaded = json.loads(text.split(marker, 1)[1])
    if not isinstance(loaded, dict):
        raise ValueError("VDA public context is not an object")
    return loaded


def audit_pool(
    manifest_path: Path,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    import pandas as pd

    manifest = read_json(manifest_path)
    if manifest.get("kind") != "vda_pool":
        raise ValueError("manifest kind must be vda_pool")
    if manifest.get("protocol_version") != "tmcd-v2":
        raise ValueError("release audit requires TMCD-v2")
    if int(manifest.get("candidate_normalization_version", 0)) < 2:
        raise ValueError("candidate normalization version must be at least 2")
    if str(manifest.get("tmcd_release_revision", "")) != TMCD_RELEASE_REVISION:
        raise ValueError("pool does not match the active TMCD release revision")

    paths = manifest.get("paths", {}) or {}
    fingerprints: set[str] = set()
    pair_splits: dict[str, set[str]] = defaultdict(set)
    pair_types: dict[str, list[str]] = defaultdict(list)
    actual_counts: dict[str, int] = {}
    actual_tasks: dict[str, dict[str, int]] = {}
    prompt_rows = 0

    for split, expected in expected_counts.items():
        path = Path(str(paths.get(split, ""))).resolve()
        if not path.is_file():
            raise ValueError(f"missing {split} parquet: {path}")
        frame = pd.read_parquet(path)
        actual_counts[split] = len(frame)
        if len(frame) != expected:
            raise ValueError(f"{split} has {len(frame)} rows, expected {expected}")
        task_counts = Counter(str(value) for value in frame["task_id"])
        expected_tasks = {
            str(task_id): int(count)
            for task_id, count in (
                (manifest.get("split_task_counts", {}) or {}).get(split, {}) or {}
            ).items()
        }
        if set(expected_tasks) != set(TASK_IDS) or sum(expected_tasks.values()) != expected:
            raise ValueError(f"{split} manifest task quotas are invalid: {expected_tasks}")
        if max(expected_tasks.values()) - min(expected_tasks.values()) > 2:
            raise ValueError(f"{split} task quotas are not approximately balanced")
        if expected_tasks["T2"] % 2:
            raise ValueError(f"{split} T2 quota must be even")
        if dict(task_counts) != expected_tasks:
            raise ValueError(
                f"{split} task counts mismatch: {dict(task_counts)} != {expected_tasks}"
            )
        actual_tasks[split] = dict(task_counts)

        for row in frame.to_dict(orient="records"):
            context = _public_context(row.get("problem"))
            leaks = forbidden_public_paths(context)
            if leaks:
                raise ValueError(f"hidden public path in {split}: {leaks[0]}")
            if not INSTANCE_ID.fullmatch(str(context.get("instance_id", ""))):
                raise ValueError(f"invalid opaque instance_id in {split}")
            prompt_rows += 1

            scenario = _as_dict(row.get("scenario"))
            extra = _as_dict(row.get("extra_info"))
            fingerprint = str(
                extra.get("scenario_fingerprint") or scenario_fingerprint(scenario)
            )
            if not fingerprint or fingerprint in fingerprints:
                raise ValueError("duplicate VDA scenario fingerprint")
            fingerprints.add(fingerprint)
            if str(row.get("task_id")) == "T2":
                pair_id = str(scenario.get("pair_id", "")).strip()
                if not pair_id:
                    raise ValueError("T2 scenario is missing pair_id")
                pair_splits[pair_id].add(split)
                pair_types[pair_id].append(str(scenario.get("trajectory_type", "")))

    for pair_id, branches in pair_types.items():
        if sorted(branches) != ["betrayal", "legitimate_change"]:
            raise ValueError(f"invalid T2 pair {pair_id}: {branches}")
        if len(pair_splits[pair_id]) != 1:
            raise ValueError(f"T2 pair crosses splits: {pair_id}")

    if actual_counts != {
        str(split): int(count)
        for split, count in (manifest.get("split_counts", {}) or {}).items()
    }:
        raise ValueError("manifest split_counts do not match release expectations")
    if actual_tasks != {
        str(split): {str(task): int(count) for task, count in counts.items()}
        for split, counts in (manifest.get("split_task_counts", {}) or {}).items()
    }:
        raise ValueError("manifest split_task_counts do not match parquet data")

    return {
        "ok": True,
        "manifest": str(manifest_path),
        "split_counts": actual_counts,
        "split_task_counts": actual_tasks,
        "prompt_rows_audited": prompt_rows,
        "unique_fingerprints": len(fingerprints),
        "t2_pairs": len(pair_types),
        "public_hidden_path_violations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--dev-size", type=int, required=True)
    parser.add_argument("--xplay-size", type=int, required=True)
    args = parser.parse_args()
    result = audit_pool(
        Path(args.pool_manifest).resolve(),
        {
            "train": args.train_size,
            "dev": args.dev_size,
            "xplay": args.xplay_size,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
