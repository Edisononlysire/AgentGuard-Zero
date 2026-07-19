#!/usr/bin/env python3
"""Mix six warm-start sources with the frozen candidate curriculum ratios."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


RATIOS = {
    "teacher": 0.20,
    "random_legal": 0.10,
    "scripted_skill": 0.15,
    "visited": 0.15,
    # The source table is advisory, while the protocol separately requires
    # 30%-40% minimal public counterfactual states. Freeze the pilot at 35%.
    "counterfactual_flip": 0.35,
    "delayed_noop_error": 0.05,
}


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in RATIOS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--record-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if args.record_count < 20 or args.record_count % 20:
        raise ValueError("record-count must be a positive multiple of 20")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    selected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    for name, ratio in RATIOS.items():
        path = Path(getattr(args, name))
        rows = read(path)
        count = round(args.record_count * ratio)
        if len(rows) < count:
            raise RuntimeError(f"{name} has {len(rows)} records; requires {count}")
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{args.seed}:{name}:{row['record_id']}".encode("utf-8")
            ).hexdigest(),
        )
        chosen = ranked[:count]
        for row in chosen:
            row["data_source"] = name
        selected.extend(chosen)
        source_counts[name] = count
        source_hashes[name] = sha256_file(path)

    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}:mix:{row['record_id']}".encode("utf-8")
        ).hexdigest()
    )
    target_by_state: dict[str, set[str]] = {}
    for row in selected:
        target_by_state.setdefault(row["semantic_public_state_digest"], set()).add(
            row["target_semantic_id"]
        )
    conflict_rate = sum(len(values) > 1 for values in target_by_state.values()) / max(
        1, len(target_by_state)
    )
    scenario_by_source = {
        name: {
            row["semantic_scenario_fingerprint"]
            for row in selected
            if row["data_source"] == name
        }
        for name in RATIOS
    }
    overlap = 0
    names = list(RATIOS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap += len(scenario_by_source[left] & scenario_by_source[right])
    # Multiple trajectory policies intentionally visit the same training
    # scenario. Leakage is enforced when splitting train/dev/test, not between
    # source buckets inside the same training pool.
    accepted = conflict_rate == 0.0

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "candidate_sets.jsonl"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    manifest = {
        "schema_version": 1,
        "kind": "candidate_warmstart_six_source_mix",
        "created_at": utc_now(),
        "accepted": accepted,
        "record_count": len(selected),
        "source_ratios": RATIOS,
        "source_counts": source_counts,
        "source_hashes": source_hashes,
        "semantic_source_overlap": overlap,
        "semantic_target_conflict_rate": conflict_rate,
        "counterfactual_state_share": RATIOS["counterfactual_flip"],
        "candidate_sets_sha256": sha256_file(output),
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
