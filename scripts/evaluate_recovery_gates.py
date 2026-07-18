#!/usr/bin/env python3
"""Evaluate recovery gates from frozen metric JSON; never launches training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.gates import (
    choose_gate_a_arm,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_stage0_gate,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="gate", required=True)

    stage0 = sub.add_parser("stage0")
    stage0.add_argument("--metrics", type=Path, required=True)

    gate_a = sub.add_parser("gate-a")
    gate_a.add_argument("--base-metrics", type=Path, required=True)
    gate_a.add_argument("--vda1-metrics", type=Path, required=True)

    gate_b = sub.add_parser("gate-b")
    gate_b.add_argument("--metrics", type=Path, required=True)
    gate_b.add_argument("--baseline", type=Path, required=True)

    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.gate == "stage0":
        result: dict[str, Any] = evaluate_stage0_gate(
            _load(args.metrics)
        ).to_dict()
    elif args.gate == "gate-a":
        base = evaluate_gate_a(
            _load(args.base_metrics),
            arm="qwen3.5_base",
        )
        vda1 = evaluate_gate_a(
            _load(args.vda1_metrics),
            arm="vda_1",
        )
        result = {
            "base": base.to_dict(),
            "vda_1": vda1.to_dict(),
            "selection": choose_gate_a_arm([base, vda1]),
        }
        result["accepted"] = bool(result["selection"]["accepted"])
    else:
        result = evaluate_gate_b(
            _load(args.metrics),
            _load(args.baseline),
        ).to_dict()

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if bool(result.get("accepted", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
