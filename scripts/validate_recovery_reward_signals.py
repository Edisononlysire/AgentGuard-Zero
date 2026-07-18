#!/usr/bin/env python3
"""Validate 32+ real Gate-B trajectory signal payloads before any update."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.rewards.recovery_vda_reward import (
    validate_recovery_signal_batch,
)
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-trajectories", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = [
        json.loads(line)
        for line in args.signals.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verdict = validate_recovery_signal_batch(
        rows,
        minimum_trajectories=args.minimum_trajectories,
    )
    verdict["signal_source"] = str(args.signals.resolve())
    verdict["signal_source_sha256"] = sha256_file(args.signals)
    atomic_write_json(args.output, verdict)
    print(json.dumps(verdict, ensure_ascii=False, sort_keys=True))
    return 0 if verdict["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
