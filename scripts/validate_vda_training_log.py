#!/usr/bin/env python3
"""Validate that a VDA gate completed one real trajectory-reward LoRA update."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


FATAL_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "Traceback (most recent call last)",
)

# VerL's environment metric includes Qwen's two-token turn terminator, while
# the rollout response budget only counts generated action tokens.
ACTION_PROTOCOL_OVERHEAD_TOKENS = 2


def parse_training_metrics(
    text: str,
    *,
    expected_step: int,
    action_budget: int,
    observation_budget: int,
) -> dict[str, float]:
    if any(marker in text for marker in FATAL_MARKERS):
        raise ValueError("VDA gate log contains a fatal error")
    metric_lines = [
        line
        for line in text.splitlines()
        if f"training/global_step:{expected_step}" in line
    ]
    if not metric_lines:
        raise ValueError(f"VDA gate did not emit the step-{expected_step} metric")
    line = metric_lines[-1]

    def metric(name: str) -> float:
        match = re.search(rf"(?:^| - ){re.escape(name)}:([^ ]+)", line)
        if not match:
            raise ValueError(f"missing VDA gate metric: {name}")
        return float(match.group(1))

    report = {
        "training_global_step": metric("training/global_step"),
        "trajectory_reward_available": metric(
            "reward_extra_info/level1_trajectory_reward_available"
        ),
        "single_step_reward_fallback": metric(
            "reward_extra_info/single_step_reward_fallback"
        ),
        "actor_grad_norm": metric("actor/grad_norm"),
        "actor_lr": metric("actor/lr"),
        "valid_action_ratio": metric("env/ratio_of_valid_action"),
        "max_action_tokens": metric("env/action_length/max"),
        "max_observation_tokens": metric("env/obs_length/max"),
        "max_memory_allocated_gb": metric("perf/max_memory_allocated_gb"),
        "max_memory_reserved_gb": metric("perf/max_memory_reserved_gb"),
        "step_seconds": metric("timing_s/step"),
    }
    if report["training_global_step"] != float(expected_step):
        raise ValueError(f"unexpected global step: {report}")
    if report["trajectory_reward_available"] != 1.0:
        raise ValueError(f"trajectory reward was unavailable: {report}")
    if report["single_step_reward_fallback"] != 0.0:
        raise ValueError(f"single-step reward fallback was used: {report}")
    if not math.isfinite(report["actor_grad_norm"]) or report["actor_grad_norm"] <= 0.0:
        raise ValueError(f"invalid actor gradient norm: {report}")
    if not math.isfinite(report["actor_lr"]) or report["actor_lr"] <= 0.0:
        raise ValueError(f"the gate did not perform a non-zero-LR optimizer step: {report}")
    if report["max_action_tokens"] > action_budget + ACTION_PROTOCOL_OVERHEAD_TOKENS:
        raise ValueError(f"action wire budget was exceeded: {report}")
    if report["max_observation_tokens"] > observation_budget:
        raise ValueError(f"observation wire budget was exceeded: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-step", type=int, default=1)
    parser.add_argument("--action-budget", type=int, required=True)
    parser.add_argument("--observation-budget", type=int, required=True)
    args = parser.parse_args()

    report = parse_training_metrics(
        Path(args.log).read_text(encoding="utf-8", errors="replace"),
        expected_step=args.expected_step,
        action_budget=args.action_budget,
        observation_budget=args.observation_budget,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
