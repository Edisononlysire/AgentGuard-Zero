#!/usr/bin/env python3
"""Evaluate Gate A or a round-over-round candidate policy gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.gates import evaluate_round_gate, evaluate_vda_gate_a
from agentguard_zero.training.coevolution import atomic_write_json, utc_now


def metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("metrics", payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate_a = commands.add_parser("gate-a")
    gate_a.add_argument("--evaluation", type=Path, required=True)
    gate_a.add_argument("--output", type=Path, required=True)
    round_gate = commands.add_parser("round")
    round_gate.add_argument("--start", type=Path, required=True)
    round_gate.add_argument("--end", type=Path, required=True)
    round_gate.add_argument("--round-index", type=int, required=True)
    round_gate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    verdict = (
        evaluate_vda_gate_a(metrics(args.evaluation))
        if args.command == "gate-a"
        else evaluate_round_gate(
            metrics(args.start), metrics(args.end), round_index=args.round_index
        )
    )
    payload = {"created_at": utc_now(), **verdict.to_dict()}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if verdict.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
