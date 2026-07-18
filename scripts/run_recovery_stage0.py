#!/usr/bin/env python3
"""Run the model-free Stage-0 learnability gate on 200 canonical scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.fixed_policies import run_stage0_suite
from agentguard_zero.recovery.gates import evaluate_stage0_gate
from agentguard_zero.recovery.protocol import RecoveryConfig
from agentguard_zero.recovery.public_teacher import PublicStateRobustTeacher


def _load_groups(path: Path) -> list[list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups")
    if not isinstance(groups, list) or any(not isinstance(item, list) for item in groups):
        raise ValueError("canonical input must contain public-world groups")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    groups = _load_groups(args.scenarios)
    scenario_count = sum(len(group) for group in groups)
    config = RecoveryConfig()
    if scenario_count != config.stage0.scenario_count:
        raise ValueError(
            f"Stage-0 requires {config.stage0.scenario_count} scenarios, got {scenario_count}"
        )
    teacher = PublicStateRobustTeacher(
        advantage_delta=config.teacher.advantage_delta,
        min_worlds_per_public_state=config.teacher.min_worlds_per_public_state,
        beam_width=config.teacher.beam_width,
        max_candidates=config.teacher.max_candidates,
    )
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    metrics = run_stage0_suite(groups, teacher=teacher, workers=args.workers)
    verdict = evaluate_stage0_gate(metrics, config)
    output = {
        "verdict": verdict.to_dict(),
        "scenario_source_sha256": hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
        "model_calls": 0,
        "parameter_updates": 0,
        "workers": args.workers,
        "hidden_state_usage": "fixed_policy_or_offline_teacher_utility_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if verdict.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
