#!/usr/bin/env python3
"""Build one deterministic fresh/canonical/past/DAgger candidate replay mix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ranked(values: list[dict], seed: int, label: str) -> list[dict]:
    return sorted(
        values,
        key=lambda row: hashlib.sha256(
            f"{seed}:{label}:{row['record_id']}".encode()
        ).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--past", type=Path, nargs="*")
    parser.add_argument("--dagger", type=Path)
    parser.add_argument("--probe-chain", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-records", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    canonical_rows = rows(args.canonical)
    sources = {
        "fresh_frontier": rows(args.fresh),
        "canonical_replay": canonical_rows,
        "past_round_replay": [row for path in (args.past or []) for row in rows(path)],
        "probe_chain_states": (
            rows(args.probe_chain)
            if args.probe_chain
            else [
                row
                for row in canonical_rows
                if bool(
                    (row.get("probe_chain_target") or {}).get(
                        "is_probe_followup_state", False
                    )
                )
            ]
        ),
        "hard_state_replay": rows(args.dagger),
    }
    ratios = {
        "fresh_frontier": 0.45,
        "canonical_replay": 0.20,
        "past_round_replay": 0.15,
        "probe_chain_states": 0.10,
        "hard_state_replay": 0.10,
    }
    selected: list[dict] = []
    source_counts = Counter()
    seen: set[str] = set()
    for label, ratio in ratios.items():
        target = round(args.total_records * ratio)
        for row in ranked(sources[label], args.seed, label):
            if source_counts[label] >= target:
                break
            key = str(row["record_id"])
            if key in seen:
                continue
            copy_row = dict(row)
            copy_row["replay_source"] = label
            selected.append(copy_row)
            source_counts[label] += 1
            seen.add(key)
    # Round 1 has no past/DAgger yet. Fill missing capacity from canonical,
    # then fresh, while retaining exact provenance counts.
    for label in (
        "canonical_replay",
        "fresh_frontier",
        "probe_chain_states",
        "past_round_replay",
        "hard_state_replay",
    ):
        for row in ranked(sources[label], args.seed + 1, label):
            if len(selected) >= args.total_records:
                break
            key = str(row["record_id"])
            if key in seen:
                continue
            copy_row = dict(row)
            copy_row["replay_source"] = label
            selected.append(copy_row)
            source_counts[label] += 1
            seen.add(key)
        if len(selected) >= args.total_records:
            break
    if len(selected) < args.total_records:
        raise RuntimeError(
            f"replay sources provide {len(selected)} unique rows, need {args.total_records}"
        )
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
        "kind": "candidate_round_replay_mix",
        "created_at": utc_now(),
        "accepted": True,
        "record_count": len(selected),
        "source_counts": dict(sorted(source_counts.items())),
        "candidate_sets_sha256": sha256_file(output),
        "seed": args.seed,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
