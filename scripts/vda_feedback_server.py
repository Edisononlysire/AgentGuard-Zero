#!/usr/bin/env python3
"""Serve real current-VDA rollouts as DCA training feedback."""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentguard_zero.training.vda_dataset import scenario_to_training_row
from eval_level1_select import (
    ACTIVE_PROBE_ACTIONS,
    ACTIVE_PROBE_TOOLS,
    PASSIVE_VERIFY_TOOLS,
    HFBackend,
    as_messages,
    compute_safe_utility,
    next_user_message,
    sanitize_initial_messages,
    scenario_extra_from_row,
    select_candidate,
)
from generate_level1_frontier import compute_cfc_metrics
from level1_rollout_server import Level1RolloutStore


def _small_event(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    return {
        key: event[key]
        for key in ("event_id", "time", "type", "source", "objective_hint", "spoofability")
        if key in event
    }


def _history_summary(
    turn: int,
    public_context: Any,
    selected_packet: dict[str, Any],
) -> dict[str, Any]:
    context = public_context if isinstance(public_context, dict) else {}
    observation = context.get("observation", context)
    observation = observation if isinstance(observation, dict) else {}
    packet = selected_packet if isinstance(selected_packet, dict) else {}
    belief = packet.get("belief", {}) if isinstance(packet.get("belief"), dict) else {}
    def belief_value(key: str) -> float:
        try:
            return float(belief.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    top_belief = max(belief, key=belief_value) if belief else "unknown"
    tool = packet.get("tool_call", {}) if isinstance(packet.get("tool_call"), dict) else {}
    response = packet.get("response", {}) if isinstance(packet.get("response"), dict) else {}
    memory = packet.get("memory_update", {}) if isinstance(packet.get("memory_update"), dict) else {}
    trust = packet.get("trust_update", {}) if isinstance(packet.get("trust_update"), dict) else {}
    tool_result = observation.get("last_tool_result")
    if isinstance(tool_result, dict):
        tool_result = {
            key: tool_result[key]
            for key in (
                "tool",
                "event_id",
                "source",
                "verdict",
                "challenge_consistency",
                "contradiction_risk",
                "canary_triggered",
                "overresponse_risk",
                "estimated_cost",
            )
            if key in tool_result
        }
    return {
        "t": turn,
        "events": [_small_event(event) for event in observation.get("events", []) or []],
        "tool_result": tool_result,
        "decision": {
            "belief": top_belief,
            "uncertainty": packet.get("uncertainty"),
            "tool": tool.get("name", "None"),
            "action": response.get("action", "Observe"),
            "target": response.get("target", "none"),
            "source_reliability": trust.get("source_reliability", {}),
            "quarantine": memory.get("profile_quarantine", []),
            "confirmed": memory.get("confirmed_profile", []),
            "rejected": memory.get("rejected_profile", []),
        },
    }


def _generation_messages(state: dict[str, Any]) -> list[dict[str, str]]:
    initial_messages = state["initial_messages"]
    if not state["history"]:
        return initial_messages
    continuation = {
        "history": state["history"],
        "current_public_state": state["public_context"],
    }
    return initial_messages + [
        {
            "role": "user",
            "content": (
                "Compact trajectory state (history is chronological):"
                + json.dumps(continuation, ensure_ascii=False, separators=(",", ":"), default=str)
                + "\nReturn the next compact strict VDA JSON action only."
            ),
        }
    ]


def _ambiguity_penalty(scenario: dict[str, Any]) -> float:
    assets = scenario.get("network_context", {}).get("assets", []) or []
    fake = scenario.get("poisoning_plan", {}).get("fake_evidence", []) or []
    claims = [str(item.get("claim", "")).strip() for item in fake]
    penalty = 0.0
    if len(assets) < 2:
        penalty += 0.5
    if not fake:
        penalty += 0.35
    if claims and max(map(len, claims)) < 12:
        penalty += 0.25
    return min(1.0, penalty)


def _action_metrics(result: dict[str, Any]) -> dict[str, Any]:
    active = 0
    passive = 0
    quarantined = 0
    confirmed = 0
    summaries = []
    for selected in result.get("selected_actions", []) or []:
        packet = selected.get("selected_packet", {}) or {}
        tool = str((packet.get("tool_call", {}) or {}).get("name", "None"))
        action = str((packet.get("response", {}) or {}).get("action", "Observe"))
        memory = packet.get("memory_update", {}) or {}
        active += int(tool in ACTIVE_PROBE_TOOLS or action in ACTIVE_PROBE_ACTIONS)
        passive += int(tool in PASSIVE_VERIFY_TOOLS)
        quarantined += len(memory.get("profile_quarantine", []) or [])
        confirmed += len(memory.get("confirmed_profile", []) or [])
        summaries.append(
            {
                "turn": selected.get("turn"),
                "tool": tool,
                "action": action,
                "parse_ok": bool(selected.get("selected_ok", False)),
            }
        )
    return {
        "current_vda_active_probe_count": active,
        "current_vda_passive_verify_count": passive,
        "current_vda_quarantine_count": quarantined,
        "current_vda_confirmed_count": confirmed,
        "current_vda_action_summaries": summaries,
    }


class FeedbackEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend: HFBackend | None = None
        self.lock = threading.Lock()
        self.request_count = 0
        if not args.lazy_load:
            self._ensure_backend()

    @property
    def model_loaded(self) -> bool:
        return self.backend is not None

    @property
    def model_resident(self) -> bool:
        if self.backend is None:
            return False
        return next(self.backend.model.parameters()).device.type != "cpu"

    def _ensure_backend(self) -> HFBackend:
        if self.backend is None:
            self.backend = HFBackend(self.args)
            self.backend.tokenizer.truncation_side = "left"
        return self.backend

    def _activate_backend(self) -> HFBackend:
        backend = self._ensure_backend()
        if next(backend.model.parameters()).device != backend.device:
            backend.model.to(backend.device)
            backend.model.eval()
        return backend

    def _offload_backend(self) -> None:
        if self.backend is None or not self.model_resident:
            return
        self.backend.model.to("cpu")
        if self.backend.torch.cuda.is_available():
            self.backend.torch.cuda.empty_cache()

    def _generate_batch(self, message_batches: list[list[dict[str, str]]]) -> list[str]:
        backend = self._activate_backend()
        torch = backend.torch
        prompts = [backend.format_prompt(messages) for messages in message_batches]
        encoded = backend.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_input_tokens,
        )
        encoded = {key: value.to(backend.device) for key, value in encoded.items()}
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": bool(self.args.do_sample),
            "pad_token_id": backend.tokenizer.pad_token_id,
            "eos_token_id": backend.tokenizer.eos_token_id,
        }
        if self.args.do_sample:
            kwargs.update(
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=max(0, int(self.args.top_k)),
            )
        if self.args.stop_on_complete_json:
            from transformers import StoppingCriteriaList
            from agentguard_zero.json_stopping import CompleteJSONObjectCriteria

            kwargs["stopping_criteria"] = StoppingCriteriaList(
                [CompleteJSONObjectCriteria(backend.tokenizer, batch_size=len(message_batches))]
            )
        with torch.inference_mode():
            output = backend.model.generate(**encoded, **kwargs)
        input_length = encoded["input_ids"].shape[-1]
        return [
            backend.tokenizer.decode(row[input_length:], skip_special_tokens=True).strip()
            for row in output
        ]

    def _run_many(self, scenarios: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
        store = Level1RolloutStore(invalid_penalty=self.args.invalid_penalty)
        states: list[dict[str, Any]] = []
        for offset, scenario in enumerate(scenarios):
            row = scenario_to_training_row(scenario, split="dca_feedback")
            messages, public_context = sanitize_initial_messages(as_messages(row.get("problem", "")))
            extra = scenario_extra_from_row(row)
            max_env_steps = int(extra.get("max_env_steps", self.args.max_turns))
            states.append(
                {
                    "scenario": scenario,
                    "initial_messages": messages,
                    "public_context": public_context,
                    "history": [],
                    "extra": extra,
                    "max_turns": min(self.args.max_turns, max_env_steps),
                    "trajectory_id": (
                        f"dca-feedback-{start_index + offset}-{scenario.get('scenario_id', offset)}"
                    ),
                    "selected_actions": [],
                    "final_observation": None,
                    "done": False,
                }
            )

        for turn in range(max((state["max_turns"] for state in states), default=0)):
            active = [
                index
                for index, state in enumerate(states)
                if not state["done"] and turn < state["max_turns"]
            ]
            if not active:
                break
            raw_outputs = self._generate_batch(
                [_generation_messages(states[index]) for index in active]
            )
            selected_values = []
            for index, raw in zip(active, raw_outputs):
                state = states[index]
                selected = select_candidate(
                    state["public_context"],
                    [raw],
                    self.args.policy,
                    selector_mode=self.args.selector_mode,
                )
                selected_values.append(selected)
                state["selected_actions"].append(
                    {
                        "turn": turn,
                        "selected_text": selected.text,
                        "selected_packet": selected.packet,
                        "selected_ok": selected.ok,
                        "parse_msg": selected.parse_msg,
                    }
                )
            response = store.handle(
                {
                    "trajectory_ids": [states[index]["trajectory_id"] for index in active],
                    "actions": [selected.text for selected in selected_values],
                    "finish": [False for _ in active],
                    "is_last_step": [turn + 1 >= states[index]["max_turns"] for index in active],
                    "extra_fields": [states[index]["extra"] for index in active],
                }
            )
            for position, index in enumerate(active):
                state = states[index]
                observation = response["observations"][position]
                state["final_observation"] = observation
                state["done"] = bool(response["dones"][position])
                if not state["done"]:
                    user_message, public_context = next_user_message(observation)
                    del user_message
                    state["history"].append(
                        _history_summary(
                            turn,
                            public_context,
                            selected_values[position].packet,
                        )
                    )
                    state["public_context"] = public_context

        results = []
        for state in states:
            score = dict((state["final_observation"] or {}).get("score", {}))
            steps = int(score.get("steps", len(state["selected_actions"])))
            budget = float(
                state["scenario"].get("defense_constraints", {}).get("business_budget", 5.0)
            )
            score["safe_utility"] = compute_safe_utility(
                score,
                steps=steps,
                max_steps=int(state["extra"].get("max_env_steps", state["max_turns"])),
                budget=budget,
            )
            results.append(
                {
                    "done": state["done"],
                    "steps": steps,
                    "score": score,
                    "selected_actions": state["selected_actions"],
                }
            )
        return results

    def _combine(
        self,
        scenario: dict[str, Any],
        cfc: dict[str, Any],
        rollout: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if rollout is None:
            return {
                **cfc,
                "checks": cfc.get("checks", {}),
                "oracle_solvable": False,
                "ambiguity_penalty": _ambiguity_penalty(scenario),
                "current_vda_safe_success": False,
                "current_vda_safe_utility": -1.0,
            }
        score = rollout.get("score", {}) or {}
        metrics = _action_metrics(rollout)
        return {
            **cfc,
            **metrics,
            "checks": cfc.get("checks", {}),
            "oracle_solvable": bool(cfc.get("oracle_solvable", False)),
            "ambiguity_penalty": _ambiguity_penalty(scenario),
            "current_vda_safe_success": bool(score.get("safe_success", False)),
            "current_vda_safe_utility": float(score.get("safe_utility", 0.0)),
            "current_vda_reward": float(score.get("reward", 0.0)),
            "current_vda_correct_intent": bool(score.get("correct_intent", False)),
            "current_vda_attack_mitigated": bool(score.get("attack_mitigated", False)),
            "current_vda_attack_success": bool(score.get("attack_success", False)),
            "current_vda_fake_confirmed": int(score.get("fake_confirmed", 0)),
            "current_vda_overresponse": bool(score.get("overresponse", False)),
            "current_vda_business_cost": float(score.get("business_cost", 0.0)),
            "current_vda_verification_cost": float(score.get("verification_cost", 0.0)),
            "current_vda_steps": int(rollout.get("steps", 0)),
        }

    def evaluate_many(self, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfcs = [compute_cfc_metrics(scenario) for scenario in scenarios]
        solvable_indices = [
            index for index, cfc in enumerate(cfcs) if cfc.get("oracle_solvable", False)
        ]
        rollout_by_index: dict[int, dict[str, Any]] = {}
        if solvable_indices:
            with self.lock:
                start_index = self.request_count
                self.request_count += len(scenarios)
                try:
                    rollouts = self._run_many(
                        [scenarios[index] for index in solvable_indices],
                        start_index,
                    )
                finally:
                    if self.args.offload_after_request:
                        self._offload_backend()
            rollout_by_index = dict(zip(solvable_indices, rollouts))
        else:
            self.request_count += len(scenarios)
        return [
            self._combine(scenario, cfcs[index], rollout_by_index.get(index))
            for index, scenario in enumerate(scenarios)
        ]

    def evaluate(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return self.evaluate_many([scenario])[0]


class Handler(BaseHTTPRequestHandler):
    engine: FeedbackEngine

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write(
                200,
                {
                    "ok": True,
                    "requests": self.engine.request_count,
                    "model_loaded": self.engine.model_loaded,
                    "model_resident": self.engine.model_resident,
                },
            )
        else:
            self._write(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/evaluate", "/evaluate_batch"}:
            self._write(404, {"ok": False, "error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4 * 1024 * 1024:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/evaluate_batch":
                scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
                if not isinstance(scenarios, list) or not scenarios or not all(
                    isinstance(item, dict) for item in scenarios
                ):
                    raise ValueError("scenarios must be a non-empty object list")
                results = self.engine.evaluate_many(scenarios)
                self._write(200, {"ok": True, "results": results})
            else:
                scenario = payload.get("scenario") if isinstance(payload, dict) else None
                if not isinstance(scenario, dict):
                    raise ValueError("scenario must be an object")
                result = self.engine.evaluate(scenario)
                self._write(200, {"ok": True, "result": result})
        except Exception as exc:
            self._write(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[vda-feedback] " + (fmt % args) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--stop-on-complete-json",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Bind the service before loading model weights; load on the first evaluation request.",
    )
    parser.add_argument(
        "--offload-after-request",
        action="store_true",
        help="Move VDA weights to CPU after each feedback batch so DCA update owns GPU memory.",
    )
    args = parser.parse_args()

    # Namespace fields shared with scripts/eval_level1_select.py.
    args.model_path = args.model_path
    args.adapter_path = args.adapter_path
    args.device_map = ""
    args.max_input_tokens = args.max_input_tokens
    args.max_new_tokens = args.max_new_tokens
    args.attn_implementation = args.attn_implementation
    args.policy = "zero_shot_vda"
    args.candidate_count = 1
    args.selector_mode = "v5_c_frontier_minimax"
    args.run_name = f"dca-feedback-{args.port}"
    args.invalid_penalty = 0.5
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    engine = FeedbackEngine(args)
    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "status": "ready",
                "host": args.host,
                "port": args.port,
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
