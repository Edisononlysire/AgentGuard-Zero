#!/usr/bin/env python3
"""Stateful Level-1 VDA rollout server for AgentGuard-Zero.

The server implements the verl-tool `/get_observation` contract while keeping
one symbolic cyber environment per trajectory id. It never executes payloads,
network activity, exploit logic, malware, or real cyber operations; all effects
are symbolic updates inside the Level-1 simulator.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agentguard_zero.env.oracle import score_trajectory
from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.schemas.action_schema import OBJECTIVE_KEYS, parse_action_json
from agentguard_zero.schemas.action_schema_v4 import parse_action_json_v4
from agentguard_zero.schemas.scenario_schema import minimal_example


LOGGER = logging.getLogger("level1_rollout_server")


def _as_list(value: Any, n: int, default: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return [copy.deepcopy(default) for _ in range(n)]
    return [value for _ in range(n)]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _maybe_json_obj(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def _scenario_from_extra(extra: dict[str, Any]) -> dict[str, Any]:
    for key in ("scenario", "scenario_json", "ground_truth", "answer"):
        scenario = _maybe_json_obj(extra.get(key))
        if scenario is not None:
            return scenario
    scenario = minimal_example()
    scenario_id = extra.get("scenario_id")
    if scenario_id:
        scenario["scenario_id"] = str(scenario_id)
    return scenario


def _belief_for(objective: str) -> dict[str, float]:
    belief = {key: 0.05 for key in OBJECTIVE_KEYS}
    if objective in belief:
        belief[objective] = 0.85
    else:
        belief = {key: 1.0 / len(OBJECTIVE_KEYS) for key in OBJECTIVE_KEYS}
    return belief


def _self_test_action(objective: str, response_action: str) -> str:
    target = "database"
    packet = {
        "belief": _belief_for(objective),
        "evidence_assessment": [],
        "trust_update": {"source_reliability": {}, "rationale": "self-test"},
        "uncertainty": 0.2,
        "tool_call": {"name": "None", "args": {}},
        "memory_update": {
            "profile_quarantine": [],
            "confirmed_profile": ["true-0"] if response_action != "Observe" else [],
            "rejected_profile": [],
        },
        "safety_check": {"business_risk": 0.1, "overresponse_risk": 0.1, "justification": "self-test"},
        "response": {"tier": "L1", "action": response_action, "target": target},
    }
    return _json_dumps(packet)


@dataclass
class TrajectoryState:
    trajectory_id: str
    scenario: dict[str, Any]
    env: Any
    steps: int = 0
    invalid_count: int = 0
    done: bool = False
    final_score: dict[str, Any] | None = None
    presented_evidence_ids: set[str] = field(default_factory=set)
    last_defender_state: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class Level1RolloutStore:
    def __init__(
        self,
        invalid_penalty: float = 0.5,
        max_states: int = 4096,
        defender_checkpoint_interval: int = 8,
        max_parallel_trajectories: int = 8,
        oracle_mode: bool = False,
    ):
        self.invalid_penalty = invalid_penalty
        self.max_states = max_states
        if defender_checkpoint_interval <= 0:
            raise ValueError("defender checkpoint interval must be positive")
        self.defender_checkpoint_interval = defender_checkpoint_interval
        if max_parallel_trajectories <= 0:
            raise ValueError("max_parallel_trajectories must be positive")
        self.max_parallel_trajectories = int(max_parallel_trajectories)
        self.oracle_mode = bool(oracle_mode)
        self._lock = threading.Lock()
        self._states: dict[str, TrajectoryState] = {}

    @property
    def state_count(self) -> int:
        with self._lock:
            return len(self._states)

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        trajectory_ids = payload.get("trajectory_ids") or []
        actions = payload.get("actions") or []
        n = max(len(trajectory_ids), len(actions), 1)
        trajectory_ids = _as_list(trajectory_ids, n, "")
        actions = _as_list(actions, n, "")
        finish = _as_list(payload.get("finish"), n, False)
        is_last_step = _as_list(payload.get("is_last_step"), n, False)
        extra_fields = _as_list(payload.get("extra_fields"), n, {})

        grouped: dict[str, list[tuple[int, TrajectoryState, str, bool, bool]]] = {}
        with self._lock:
            for idx in range(n):
                extra = extra_fields[idx] if isinstance(extra_fields[idx], dict) else {}
                trajectory_id = str(trajectory_ids[idx] or f"trajectory-{idx}")
                action = actions[idx]
                action_text = action if isinstance(action, str) else _json_dumps(action)
                state = self._get_or_create_state(trajectory_id, extra)
                grouped.setdefault(trajectory_id, []).append(
                    (
                        idx,
                        state,
                        action_text,
                        bool(finish[idx]),
                        bool(is_last_step[idx]),
                    )
                )

        def process_group(
            entries: list[tuple[int, TrajectoryState, str, bool, bool]],
        ) -> list[tuple[int, dict[str, Any], bool, bool]]:
            results: list[tuple[int, dict[str, Any], bool, bool]] = []
            state = entries[0][1]
            with state.lock:
                for idx, _, action_text, finish_flag, last_flag in entries:
                    obs, done, valid = self._step_state(
                        state=state,
                        action_text=action_text,
                        finish=finish_flag,
                        is_last_step=last_flag,
                    )
                    results.append((idx, obs, done, valid))
            return results

        ordered: list[tuple[dict[str, Any], bool, bool] | None] = [None] * n
        groups = list(grouped.values())
        if len(groups) <= 1:
            completed_groups = [process_group(group) for group in groups]
        else:
            workers = min(self.max_parallel_trajectories, len(groups))
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="agz-trajectory",
            ) as executor:
                completed_groups = list(executor.map(process_group, groups))
        for completed in completed_groups:
            for idx, obs, done, valid in completed:
                ordered[idx] = (obs, done, valid)

        if any(item is None for item in ordered):
            raise RuntimeError("rollout batch did not produce one result per input")
        observations = [item[0] for item in ordered if item is not None]
        dones = [1 if item[1] else 0 for item in ordered if item is not None]
        valids = [1 if item[2] else 0 for item in ordered if item is not None]

        with self._lock:
            self._trim_completed()

        return {"observations": observations, "dones": dones, "valids": valids}

    def _get_or_create_state(self, trajectory_id: str, extra: dict[str, Any]) -> TrajectoryState:
        existing = self._states.get(trajectory_id)
        if existing is not None:
            return existing

        scenario = _scenario_from_extra(extra)
        max_steps = extra.get("max_env_steps")
        try:
            max_steps_int = int(max_steps) if max_steps is not None else None
        except Exception:
            max_steps_int = None
        env = instantiate_scenario(
            scenario,
            max_steps=max_steps_int,
            oracle_mode=self.oracle_mode,
        )
        state = TrajectoryState(trajectory_id=trajectory_id, scenario=scenario, env=env)
        initial_observation = env.observe()
        state.presented_evidence_ids.update(
            str(row.get("evidence_id", ""))
            for row in initial_observation.get("available_evidence", [])
            if isinstance(row, dict) and row.get("evidence_id")
        )
        state.last_defender_state = copy.deepcopy(
            initial_observation.get("defender_state", {})
        )
        self._states[trajectory_id] = state
        return state

    @staticmethod
    def _comparable_defender_section(name: str, value: Any) -> Any:
        comparable = copy.deepcopy(value)
        if name == "memory" and isinstance(comparable, dict):
            comparable.pop("retrieval_id", None)
        return comparable

    def _step_state(
        self,
        state: TrajectoryState,
        action_text: str,
        finish: bool,
        is_last_step: bool,
    ) -> tuple[dict[str, Any], bool, bool]:
        if state.done:
            return self._done_observation(state, valid=True, invalid_reason=None, finish=finish), True, True

        if getattr(state.env, "protocol_version", "") == "tmcd-v2":
            action_packet, ok, parse_msg = parse_action_json_v4(action_text)
        else:
            action_packet, ok, parse_msg = parse_action_json(action_text)
        if not ok:
            state.invalid_count += 1

        next_obs, tool_result, env_done = state.env.step(action_packet)
        state.steps += 1

        done = bool(env_done or is_last_step)
        if done and not bool(getattr(state.env, "attack_mitigated", False)):
            state.env.attack_success = True

        if done:
            state.done = True
            self._ensure_score(state)
            return (
                self._done_observation(
                    state=state,
                    valid=ok,
                    invalid_reason=None if ok else parse_msg,
                    finish=finish,
                ),
                True,
                ok,
            )

        return (
            self._continue_observation(
                state=state,
                next_obs=next_obs,
                tool_result=tool_result,
                valid=ok,
                invalid_reason=None if ok else parse_msg,
                finish=finish,
            ),
            False,
            ok,
        )

    def _ensure_score(self, state: TrajectoryState) -> dict[str, Any]:
        if state.final_score is not None:
            return state.final_score

        if getattr(state.env, "protocol_version", "") == "tmcd-v2":
            raw_score = score_trajectory_v2(state.env)
        else:
            raw_score = score_trajectory(
                state.scenario,
                state.env.history,
                state.env.memory,
                bool(state.env.attack_mitigated),
                bool(state.env.attack_success),
                float(state.env.business_cost),
                float(state.env.verification_cost),
                int(state.env.high_impact_count),
            )
        raw_reward = float(raw_score.get("reward", 0.0))
        trajectory_reward = raw_reward - self.invalid_penalty * float(state.invalid_count)
        score = dict(raw_score)
        score.update(
            {
                "raw_reward": raw_reward,
                "reward": trajectory_reward,
                "invalid_json_count": int(state.invalid_count),
                "steps": int(state.steps),
                "trajectory_id": state.trajectory_id,
                "scenario_id": state.scenario.get("scenario_id", "unknown"),
            }
        )
        state.final_score = score
        return score

    def _continue_observation(
        self,
        state: TrajectoryState,
        next_obs: dict[str, Any],
        tool_result: dict[str, Any],
        valid: bool,
        invalid_reason: str | None,
        finish: bool,
    ) -> dict[str, Any]:
        env = state.env
        continuation = copy.deepcopy(next_obs)
        evidence = continuation.get("available_evidence", []) or []
        new_evidence = [
            row
            for row in evidence
            if isinstance(row, dict)
            and str(row.get("evidence_id", "")) not in state.presented_evidence_ids
        ]
        state.presented_evidence_ids.update(
            str(row.get("evidence_id", ""))
            for row in new_evidence
            if row.get("evidence_id")
        )
        continuation["available_evidence"] = new_evidence
        current_defender = copy.deepcopy(continuation.get("defender_state", {}))
        force_checkpoint = state.steps % self.defender_checkpoint_interval == 0
        delivered_defender: dict[str, Any] = {}
        retained_sections: list[str] = []
        for name, value in current_defender.items():
            previous = state.last_defender_state.get(name)
            changed = self._comparable_defender_section(
                name, value
            ) != self._comparable_defender_section(name, previous)
            if force_checkpoint or changed:
                delivered_defender[name] = value
            else:
                retained_sections.append(name)
        continuation["defender_state"] = delivered_defender
        continuation["observation_mode"] = (
            "continuation_checkpoint" if force_checkpoint else "continuation_delta"
        )
        if retained_sections:
            continuation["retained_defender_sections"] = retained_sections
        state.last_defender_state = current_defender
        obs_payload = {
            "time": getattr(env, "t", state.steps),
            "observation": continuation,
            "costs": {
                "business_cost": float(getattr(env, "business_cost", 0.0)),
                "verification_cost": float(getattr(env, "verification_cost", 0.0)),
                "high_impact_count": int(getattr(env, "high_impact_count", 0)),
                "invalid_json_count": int(state.invalid_count),
            },
            "last_action_valid": bool(valid),
        }
        if invalid_reason:
            obs_payload["invalid_reason"] = invalid_reason
        if finish:
            obs_payload["manager_finish_flag"] = True
        return {
            "obs": _json_dumps(obs_payload),
            "reward": None,
            "level1_done": False,
            "valid_action": bool(valid),
            "invalid_reason": invalid_reason,
            "steps": int(state.steps),
        }

    def _done_observation(
        self,
        state: TrajectoryState,
        valid: bool,
        invalid_reason: str | None,
        finish: bool,
    ) -> dict[str, Any]:
        score = self._ensure_score(state)
        env = state.env
        if bool(getattr(env, "attack_mitigated", False)):
            done_reason = "attack_mitigated"
        elif int(getattr(env, "t", state.steps)) >= int(getattr(env, "max_steps", state.steps)):
            done_reason = "env_max_steps"
        else:
            done_reason = "rollout_last_step"
        return {
            "obs": "",
            "reward": float(score.get("reward", 0.0)),
            "level1_done": True,
            "done_reason": done_reason,
            "score": score,
            "valid_action": bool(valid),
            "invalid_reason": invalid_reason,
            "manager_finish_flag": bool(finish),
            "steps": int(state.steps),
        }

    def _trim_completed(self) -> None:
        if len(self._states) <= self.max_states:
            return
        completed = [key for key, state in self._states.items() if state.done]
        for key in completed[: max(0, len(self._states) - self.max_states)]:
            self._states.pop(key, None)


class Handler(BaseHTTPRequestHandler):
    server_version = "Level1RolloutServer/0.1"

    @property
    def store(self) -> Level1RolloutStore:
        return self.server.store  # type: ignore[attr-defined]

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"ok": True, "states": self.store.state_count})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/get_observation":
            self._send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            response = self.store.handle(payload)
            LOGGER.info(
                "handled batch trajectories=%s active_states=%s dones=%s valids=%s",
                len(payload.get("trajectory_ids") or []),
                self.store.state_count,
                response.get("dones"),
                response.get("valids"),
            )
            self._send_json(200, response)
        except Exception as exc:  # pragma: no cover - defensive server path
            LOGGER.exception("request failed")
            self._send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


class Level1HTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_cls: type[BaseHTTPRequestHandler], store: Level1RolloutStore):
        super().__init__(server_address, handler_cls)
        self.store = store


def run_self_test() -> None:
    scenario = minimal_example()
    true_objective = scenario["oracle"]["true_objective"]
    store = Level1RolloutStore()
    first = store.handle(
        {
            "trajectory_ids": ["self-test"],
            "actions": [_self_test_action(true_objective, "Observe")],
            "finish": [True],
            "is_last_step": [False],
            "extra_fields": [{"scenario": scenario, "scenario_id": scenario["scenario_id"]}],
        }
    )
    assert first["dones"] == [0], first
    assert first["valids"] == [1], first
    assert first["observations"][0]["obs"], first
    second = store.handle(
        {
            "trajectory_ids": ["self-test"],
            "actions": [_self_test_action(true_objective, "LimitSession")],
            "finish": [True],
            "is_last_step": [True],
            "extra_fields": [{"scenario": scenario, "scenario_id": scenario["scenario_id"]}],
        }
    )
    assert second["dones"] == [1], second
    assert second["valids"] == [1], second
    assert second["observations"][0]["reward"] is not None, second
    assert second["observations"][0]["score"]["attack_mitigated"] is True, second
    print(json.dumps({"ok": True, "first": first, "second": second}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30150)
    parser.add_argument("--invalid-penalty", type=float, default=0.5)
    parser.add_argument("--max-states", type=int, default=512)
    parser.add_argument("--max-parallel-trajectories", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        run_self_test()
        return

    store = Level1RolloutStore(
        invalid_penalty=args.invalid_penalty,
        max_states=args.max_states,
        max_parallel_trajectories=args.max_parallel_trajectories,
    )
    server = Level1HTTPServer((args.host, args.port), Handler, store)
    LOGGER.info("Level-1 rollout server listening on http://%s:%s/get_observation", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
