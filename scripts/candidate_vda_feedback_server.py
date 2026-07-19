#!/usr/bin/env python3
"""Serve G-way candidate-ranker rollouts for frontier DCA rewards."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.policy import CandidateRankerPolicy
from agentguard_zero.env.checker import full_check
from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.utility import recovery_core_utility
from agentguard_zero.rewards.candidate_dca_reward import compute_candidate_dca_reward


class FeedbackEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        manifest = json.loads(Path(args.ranker_manifest).read_text(encoding="utf-8"))
        self.policy = CandidateRankerPolicy(
            model_path=args.model_path,
            adapter_path=manifest["adapter_path"],
            heads_path=manifest.get("heads_path"),
            score_head_path=(None if manifest.get("heads_path") else manifest["score_head_path"]),
            device=args.device,
            max_length=args.max_length,
            score_batch_size=args.score_batch_size,
        )
        self.lock = threading.Lock()
        self.request_count = 0
        self.skill_gaps = (
            json.loads(Path(args.skill_gap_json).read_text(encoding="utf-8"))
            if args.skill_gap_json
            else {}
        )

    def _rollout(self, scenario: dict[str, Any], rollout_index: int) -> dict[str, Any]:
        env = instantiate_scenario(copy.deepcopy(scenario))
        invalid = 0
        decisions = 0
        while not (
            env.t >= env.max_steps or env.attack_mitigated or env.attack_success
        ):
            decision = self.policy.decide(
                env.observe(),
                sample=True,
                temperature=self.args.temperature,
                seed=self.args.seed + self.request_count * 101 + rollout_index * 1009 + env.t,
            )
            decisions += 1
            invalid += int(decision.invalid_noop)
            env.step(copy.deepcopy(decision.packet))
        score = score_trajectory_v2(env)
        return {
            "safe_success": bool(score.get("safe_success", False)),
            "attack_mitigated": bool(score.get("attack_mitigated", False)),
            "safe_utility": recovery_core_utility(env, score),
            "action_validity": 1.0 - invalid / max(1, decisions),
            "decision_count": decisions,
            "invalid_noop_count": invalid,
        }

    def evaluate(self, scenario: dict[str, Any]) -> dict[str, Any]:
        checks = full_check(scenario)
        solvable = bool(checks.get("all_ok", False))
        if not solvable:
            feedback = {
                "teacher_solvable": False,
                "safe_success_samples": [],
                "candidate_generation_failure": False,
            }
            return {**feedback, **compute_candidate_dca_reward(feedback), "checks": checks}
        with self.lock:
            rollouts = [self._rollout(scenario, index) for index in range(self.args.rollouts)]
            self.request_count += 1
        successes = [row["safe_success"] for row in rollouts]
        task = str((scenario.get("metadata") or {}).get("task_id", "unknown"))
        mean_utility = sum(row["safe_utility"] for row in rollouts) / max(1, len(rollouts))
        feedback = {
            "teacher_solvable": True,
            "safe_success_samples": successes,
            "novelty": float((scenario.get("metadata") or {}).get("novelty", 0.5)),
            "skill_gap": float(self.skill_gaps.get(task, 0.5)),
            "vda_regret": min(1.0, max(0.0, (1.5 - mean_utility) / 3.0)),
            "candidate_generation_failure": any(
                row["action_validity"] < 1.0 for row in rollouts
            ),
        }
        reward = compute_candidate_dca_reward(feedback)
        return {
            **feedback,
            **reward,
            "checks": checks,
            "rollouts": rollouts,
            "vda_action_validity": sum(row["action_validity"] for row in rollouts)
            / len(rollouts),
        }

    def evaluate_many(self, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.evaluate(row) for row in scenarios]


class Handler(BaseHTTPRequestHandler):
    engine: FeedbackEngine

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write(200, {"ok": True, "requests": self.engine.request_count})
        else:
            self._write(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4 * 1024 * 1024:
                raise ValueError("invalid request length")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/evaluate_batch":
                scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
                if not isinstance(scenarios, list) or not scenarios:
                    raise ValueError("scenarios must be a non-empty list")
                self._write(200, {"ok": True, "results": self.engine.evaluate_many(scenarios)})
            elif self.path == "/evaluate":
                scenario = payload.get("scenario") if isinstance(payload, dict) else None
                if not isinstance(scenario, dict):
                    raise ValueError("scenario must be an object")
                self._write(200, {"ok": True, "result": self.engine.evaluate(scenario)})
            else:
                self._write(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._write(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[candidate-feedback] " + fmt % args + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ranker-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--skill-gap-json", type=Path)
    args = parser.parse_args()
    if args.rollouts != 4:
        parser.error("minimum co-evolution protocol requires exactly G=4 rollouts")
    Handler.engine = FeedbackEngine(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"event": "ready", "host": args.host, "port": args.port}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
