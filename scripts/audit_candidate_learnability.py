#!/usr/bin/env python3
"""Verify that Teacher labels contain defense utility, not only valid formatting."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-records", type=int, default=64)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.candidate_sets.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target_families = Counter()
    q_variation = 0
    nonobserve_over_observe = 0
    probe_contrast = 0
    probe_gaps: list[float] = []
    active_advantages: list[float] = []
    actionable_states = 0
    for row in rows:
        options = [CandidateOption.from_record(item) for item in row["candidates"]]
        q_values = list(map(float, row["teacher_q_values"]))
        target_families[str(row["target_family"])] += 1
        q_variation += int(max(q_values) - min(q_values) > 1.0e-6)
        observe = [q for option, q in zip(options, q_values) if option.action_flags.observe_only]
        active = [q for option, q in zip(options, q_values) if not option.action_flags.observe_only]
        if observe and active:
            gap = max(active) - max(observe)
            nonobserve_over_observe += int(gap > 0.05 + 1.0e-12)
            active_advantages.append(gap)
        probes = [q for option, q in zip(options, q_values) if option.action_flags.active_probe]
        if len(probes) >= 2:
            gap = max(probes) - min(probes)
            probe_gaps.append(gap)
            probe_contrast += int(gap > 0.01 + 1.0e-12)
        actionable_states += int(float((row.get("audit") or {}).get("teacher_advantage", 0.0)) > 0.05)
    count = max(1, len(rows))
    active_targets = sum(
        count_value
        for family, count_value in target_families.items()
        if family != "observe"
    )
    metrics = {
        "record_count": len(rows),
        "target_family_counts": dict(sorted(target_families.items())),
        "nonzero_q_variation_rate": q_variation / count,
        "active_target_rate": active_targets / count,
        "teacher_actionable_state_rate": actionable_states / count,
        "nonobserve_advantage_state_rate": nonobserve_over_observe / count,
        "mean_nonobserve_advantage": statistics.mean(active_advantages)
        if active_advantages
        else 0.0,
        "probe_contrast_state_rate": probe_contrast / count,
        "mean_probe_q_gap": statistics.mean(probe_gaps) if probe_gaps else 0.0,
    }
    failures = []
    if len(rows) < args.minimum_records:
        failures.append("insufficient_records")
    if metrics["nonzero_q_variation_rate"] < 0.30:
        failures.append("insufficient_q_variation")
    if metrics["active_target_rate"] < 0.10:
        failures.append("insufficient_active_targets")
    if metrics["nonobserve_advantage_state_rate"] < 0.10:
        failures.append("insufficient_defense_advantage")
    if metrics["probe_contrast_state_rate"] < 0.05:
        failures.append("active_probes_have_no_utility_contrast")
    for family in ("active_probe", "trust", "memory", "mitigation"):
        if target_families[family] == 0:
            failures.append(f"missing_teacher_target:{family}")
    payload = {
        "schema_version": 1,
        "kind": "candidate_defense_learnability_audit",
        "created_at": utc_now(),
        "candidate_sets_sha256": sha256_file(args.candidate_sets),
        "accepted": not failures,
        "failures": failures,
        "metrics": metrics,
        "interpretation": (
            "accepted means labels contain utility-separated defensive choices; "
            "compiler validity is intentionally not a learning target"
        ),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
