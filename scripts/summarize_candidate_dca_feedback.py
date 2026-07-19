#!/usr/bin/env python3
"""Summarize and gate candidate-level DCA feedback logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.gates import evaluate_dca_feedback_gate
from agentguard_zero.rewards.candidate_dca_reward import summarize_candidate_dca_feedback
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = [
        json.loads(line)
        for line in args.feedback_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        {
            **dict(item.get("feedback") or {}),
            "vda_action_validity": float(
                (item.get("feedback") or {}).get("vda_action_validity", 0.0)
            ),
        }
        for item in raw
    ]
    metrics = summarize_candidate_dca_feedback(rows)
    verdict = evaluate_dca_feedback_gate(metrics)
    payload = {
        "schema_version": 1,
        "kind": "candidate_dca_feedback_gate",
        "created_at": utc_now(),
        "feedback_log_sha256": sha256_file(args.feedback_log),
        "metrics": metrics,
        "verdict": verdict.to_dict(),
        "accepted": verdict.accepted,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if verdict.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
