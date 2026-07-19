#!/usr/bin/env python3
"""Build, mix, audit, and split the six-source candidate warm start."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--scenario-count", type=int, default=8)
    parser.add_argument("--records-per-source", type=int, default=64)
    parser.add_argument("--mixed-record-count", type=int, default=20)
    parser.add_argument("--group-offset", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--skill-gate", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    skill_gate = json.loads(args.skill_gate.read_text(encoding="utf-8"))
    if skill_gate.get("accepted") is not True:
        raise RuntimeError("skill identifiability gate is not accepted")
    args.output_dir.mkdir(parents=True)

    source_specs = {
        "teacher": ("teacher", False),
        "random_legal": ("random", False),
        "scripted_skill": ("scripted", False),
        "visited": ("random", False),
        "counterfactual_flip": ("teacher", True),
        "delayed_noop_error": ("noop", False),
    }
    source_paths: dict[str, Path] = {}
    for index, (name, (policy, flipped)) in enumerate(source_specs.items()):
        output = args.output_dir / "sources" / name
        command = [
            args.python,
            "-s",
            str(ROOT / "scripts/build_candidate_dataset.py"),
            "--output-dir",
            str(output),
            "--scenario-count",
            str(args.scenario_count),
            "--group-offset",
            str(args.group_offset + index * 1000),
            "--max-records",
            str(args.records_per_source),
            "--trajectory-policy",
            policy,
            "--data-source",
            name,
            "--rollout-seed",
            str(args.seed + index),
            "--permutation-seed",
            str(args.seed + index * 100),
        ]
        if flipped:
            command.append("--public-action-flip")
        run(command)
        source_paths[name] = output / "candidate_sets.jsonl"

    mixed = args.output_dir / "mixed"
    command = [
        args.python,
        "-s",
        str(ROOT / "scripts/mix_candidate_warmstart.py"),
        "--record-count",
        str(args.mixed_record_count),
        "--output-dir",
        str(mixed),
        "--seed",
        str(args.seed),
    ]
    for name, path in source_paths.items():
        command.extend([f"--{name.replace('_', '-')}", str(path)])
    run(command)
    run(
        [
            args.python,
            "-s",
            str(ROOT / "scripts/audit_candidate_dataset.py"),
            "--candidate-sets",
            str(mixed / "candidate_sets.jsonl"),
            "--output",
            str(mixed / "candidate_audit.json"),
        ]
    )
    run(
        [
            args.python,
            "-s",
            str(ROOT / "scripts/audit_candidate_learnability.py"),
            "--candidate-sets",
            str(mixed / "candidate_sets.jsonl"),
            "--output",
            str(mixed / "learnability_audit.json"),
            "--minimum-records",
            str(args.mixed_record_count),
        ]
    )
    run(
        [
            args.python,
            "-s",
            str(ROOT / "scripts/split_candidate_dataset.py"),
            "--input",
            str(mixed / "candidate_sets.jsonl"),
            "--output-dir",
            str(args.output_dir / "split"),
            "--seed",
            str(args.seed),
        ]
    )
    print(
        json.dumps(
            {
                "accepted": True,
                "train": str(args.output_dir / "split/train.jsonl"),
                "dev": str(args.output_dir / "split/dev.jsonl"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
