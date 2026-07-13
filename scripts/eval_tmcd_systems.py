#!/usr/bin/env python3
"""Evaluate the fixed AgentGuard-Zero TMCD system suite.

This runner intentionally follows the final paper protocol:

  - Rule-based SOC and Oracle Defender are deterministic, training-free systems.
  - ReAct/Base+Tools, Memory Agent, Trust-score Agent, Qwen Zero-shot VDA,
    Cyber LLM VDA, AgentGuard-Zero-Select, and AgentGuard-Zero-Train share the
    same Level-1 simulator, VDA JSON schema, and tool interface.
  - AgentGuard-Zero-Train is evaluated by loading a LoRA adapter when provided;
    this script does not train parameters.

The script is safe by construction: it runs only symbolic cyber scenarios and
never emits payloads, exploits, malware logic, real IPs, or real organizations.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_level1_select as base
from level1_rollout_server import Level1RolloutStore
from vda_feedback_server import _generation_messages, _history_summary


SYSTEMS = [
    "rule_based_soc",
    "react_base_tools",
    "memory_agent",
    "trust_score_agent",
    "cyber_llm_vda",
    "lily_cybersecurity_vda",
    "qwen_zero_shot_vda",
    "agentguard_zero_select",
    "agentguard_zero_train",
    "agentguard_zero_train_select",
    "oracle_defender",
]

INTERNAL_SYSTEMS = {"random_policy"}

SYSTEM_DISPLAY = {
    "rule_based_soc": "Rule-based SOC",
    "react_base_tools": "ReAct / Base+Tools",
    "memory_agent": "Memory Agent",
    "trust_score_agent": "Trust-score Agent",
    "cyber_llm_vda": "Cyber LLM VDA",
    "lily_cybersecurity_vda": "Lily-Cybersecurity-7B VDA",
    "qwen_zero_shot_vda": "Qwen Zero-shot VDA",
    "agentguard_zero_select": "AgentGuard-Zero-Select",
    "agentguard_zero_train": "AgentGuard-Zero-Train",
    "agentguard_zero_train_select": "AgentGuard-Zero-Train + V5-C",
    "oracle_defender": "Oracle Defender",
    "random_policy": "Random Policy",
}

TASK_MAP = {
    "Evidence-Level Profile Poisoning": "T3 Profile / Memory Poisoning",
    "Long-Horizon Trust-Building Betrayal": "T2 Trust-Building Betrayal",
    "Objective / Strategy Switching": "T1 Active Probing Defense",
    "Active Probing Defense": "T1 Active Probing Defense",
    "Business-Constrained Overreaction Induction": "T4 Business-Constrained Overreaction",
    "Business-Constrained Overreaction": "T4 Business-Constrained Overreaction",
    "CAGE-style Evidence-Level Profile Poisoning": "CAGE-T3 Profile / Memory Poisoning",
    "CAGE-style Business-Constrained Overreaction": "CAGE-T4 Business-Constrained Overreaction",
}

MODEL_SYSTEMS = {
    "react_base_tools",
    "memory_agent",
    "trust_score_agent",
    "cyber_llm_vda",
    "lily_cybersecurity_vda",
    "qwen_zero_shot_vda",
    "agentguard_zero_select",
    "agentguard_zero_train",
    "agentguard_zero_train_select",
}


def json_dumps(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, indent=indent, default=str)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def task_name(row: dict[str, Any]) -> str:
    split = str(row.get("cfc_split") or row.get("split") or "")
    extra = base.as_dict(row.get("extra_info", {}))
    split = str(extra.get("cfc_split") or split)
    return TASK_MAP.get(split, split or "unknown")


def first_event_id(public_context: Any) -> str:
    event_ids = base.public_signals(public_context).get("event_ids") or ["event-0"]
    return str(event_ids[0])


def scenario_target(scenario: dict[str, Any]) -> str:
    return str(scenario.get("true_attack", {}).get("target", "none"))


def scenario_true_objective(scenario: dict[str, Any]) -> str:
    return str(
        scenario.get("oracle", {}).get("true_objective")
        or scenario.get("true_attack", {}).get("objective")
        or "exfiltration"
    )


def normalize_belief(objective: str) -> dict[str, float]:
    belief = {key: 0.05 for key in base.OBJECTIVE_KEYS}
    if objective in belief:
        belief[objective] = 0.85
    else:
        belief = {key: 1.0 / len(base.OBJECTIVE_KEYS) for key in base.OBJECTIVE_KEYS}
    return belief


def packet(
    *,
    belief: dict[str, float],
    event_id: str,
    action: str,
    tool: str = "None",
    target: str | None = None,
    quarantine: list[str] | None = None,
    confirmed: list[str] | None = None,
    rejected: list[str] | None = None,
    rationale: str = "policy decision",
    business_risk: float = 0.25,
    overresponse_risk: float = 0.25,
    uncertainty: float = 0.45,
) -> dict[str, Any]:
    tool_args: dict[str, Any] = {}
    if tool in {"CrossCheck", "ProvenanceCheck"}:
        tool_args = {"event_id": event_id}
    elif tool in {"SourceChallenge", "CanaryProbe"}:
        tool_args = {"event_id": event_id}
    elif tool in {"DecoyProbe", "DeployDecoy"}:
        tool_args = {"event_id": event_id, "decoy_type": "symbolic_probe_decoy", "zone": "server"}
    elif tool in {"BusinessImpactEstimator", "ShadowResponse", "ShadowActionProbe"}:
        tool_args = {"action": {"tier": "L1", "action": action, "target": target or event_id}}
    elif tool == "GraphQuery":
        tool_args = {"node": target or event_id}

    status = "unverified"
    if rejected:
        status = "rejected"
    elif confirmed:
        status = "confirmed"
    return {
        "belief": belief,
        "evidence_assessment": [
            {
                "event_id": event_id,
                "status": status,
                "suspected_poisoning": bool(quarantine or rejected),
            }
        ],
        "trust_update": {"source_reliability": {}, "rationale": rationale},
        "uncertainty": float(uncertainty),
        "tool_call": {"name": tool, "args": tool_args},
        "memory_update": {
            "profile_quarantine": list(quarantine or []),
            "confirmed_profile": list(confirmed or []),
            "rejected_profile": list(rejected or []),
        },
        "safety_check": {
            "business_risk": float(business_risk),
            "overresponse_risk": float(overresponse_risk),
            "justification": "bounded symbolic response",
        },
        "response": {"tier": "L1", "action": action, "target": target or event_id},
    }


def rule_based_soc_action(public_context: Any, scenario: dict[str, Any] | None, turn: int) -> dict[str, Any]:
    signals = base.public_signals(public_context)
    event_id = first_event_id(public_context)
    belief = base.public_belief_guess(public_context)
    suspicious = bool(signals.get("suspicious"))
    high_risk = bool(signals.get("high_public_risk"))
    critical = bool(signals.get("critical_asset"))
    verified = bool(signals.get("verified_evidence"))
    verification_spent = base.safe_float(signals.get("verification_cost", 0.0), 0.0)

    if suspicious and verification_spent < 1.0:
        return packet(
            belief=belief,
            event_id=event_id,
            tool="CrossCheck",
            action="CrossCheck",
            quarantine=[event_id],
            rationale="SOC rule: verify suspicious event",
            uncertainty=0.60,
        )
    if critical and high_risk and verification_spent < 2.0 and not verified:
        return packet(
            belief=belief,
            event_id=event_id,
            tool="BusinessImpactEstimator",
            action="Observe",
            quarantine=[event_id],
            rationale="SOC rule: estimate critical impact",
            business_risk=0.55,
            overresponse_risk=0.55,
            uncertainty=0.55,
        )
    if high_risk and (verified or verification_spent >= 1.0 or turn >= 2):
        return packet(
            belief=belief,
            event_id=event_id,
            tool="None",
            action="LimitSession",
            confirmed=[event_id] if verified else [],
            rationale="SOC rule: low-impact containment",
            uncertainty=0.35,
        )
    return packet(belief=belief, event_id=event_id, action="Observe", rationale="SOC rule: observe")


def oracle_action(public_context: Any, scenario: dict[str, Any] | None, turn: int) -> dict[str, Any]:
    scenario = scenario or {}
    event_id = first_event_id(public_context)
    target = scenario_target(scenario)
    true_objective = scenario_true_objective(scenario)
    belief = normalize_belief(true_objective)
    if turn == 0:
        return packet(
            belief=belief,
            event_id=event_id,
            tool="ProvenanceCheck",
            action="QueryProvenance",
            target=target,
            rejected=[],
            quarantine=[event_id],
            rationale="oracle checks decisive evidence",
            uncertainty=0.20,
        )
    return packet(
        belief=belief,
        event_id=event_id,
        tool="None",
        action="ShadowBlock",
        target=event_id,
        confirmed=[true_objective],
        rationale=f"oracle bounded containment for {target}",
        business_risk=0.15,
        overresponse_risk=0.10,
        uncertainty=0.10,
    )


def random_action(public_context: Any, scenario: dict[str, Any] | None, turn: int) -> dict[str, Any]:
    rng = random.Random(str(public_context)[:1000] + str(turn))
    event_id = first_event_id(public_context)
    belief = {key: rng.random() for key in base.OBJECTIVE_KEYS}
    total = sum(belief.values()) or 1.0
    belief = {key: value / total for key, value in belief.items()}
    action = rng.choice(["Observe", "CrossCheck", "LimitSession", "ShadowBlock", "Isolate"])
    tool = "CrossCheck" if action == "CrossCheck" else "None"
    return packet(
        belief=belief,
        event_id=event_id,
        tool=tool,
        action=action,
        quarantine=[event_id] if rng.random() < 0.35 else [],
        confirmed=[event_id] if rng.random() < 0.35 else [],
        business_risk=0.5,
        overresponse_risk=0.5,
        uncertainty=0.7,
    )


STATIC_POLICIES: dict[str, Callable[[Any, dict[str, Any] | None, int], dict[str, Any]]] = {
    "rule_based_soc": rule_based_soc_action,
    "oracle_defender": oracle_action,
    "random_policy": random_action,
}


def prepend_system_prompt(messages: list[dict[str, str]], content: str) -> list[dict[str, str]]:
    if not content:
        return messages
    return [{"role": "system", "content": content}] + messages


def system_prompt(system: str) -> str:
    common = (
        "Return compact strict JSON only. Do not include markdown, prose, code, payloads, "
        "exploit steps, malware logic, real IPs, or real organizations."
    )
    prompts = {
        "react_base_tools": (
            common
            + " Use a ReAct-style tool-first policy: inspect the current observation, call one useful tool when needed, "
            "then choose a bounded response. Do not use candidate selection."
        ),
        "memory_agent": (
            common
            + " You are a standard persistent-memory cyber defense agent. Maintain confirmed_profile memory for plausible "
            "evidence and reuse it in later decisions. You do not have protected quarantine memory."
        ),
        "trust_score_agent": (
            common
            + " You are a trust-score cyber defense agent. Weight claims primarily by source reliability and historical "
            "trust. Verify low-trust sources and act on high-trust sources when risk is high."
        ),
        "cyber_llm_vda": common + " Use the VDA schema and cyber-security knowledge to defend safely.",
        "lily_cybersecurity_vda": common + " Use the VDA schema and cyber-security knowledge to defend safely.",
        "qwen_zero_shot_vda": common + " Use the VDA schema and tools to defend safely.",
        "agentguard_zero_select": (
            common
            + " Generate one diverse candidate JSON action. The external AgentGuard selector will choose among candidates. "
            "When trust is uncertain, prefer safe active probes such as SourceChallenge, CanaryProbe, DecoyProbe, "
            "or ShadowActionProbe before confirmed memory updates or high-impact responses."
        ),
        "agentguard_zero_train": common + " Use the trained AgentGuard-Zero VDA policy.",
        "agentguard_zero_train_select": (
            common
            + " Use the trained AgentGuard-Zero VDA policy to generate diverse candidate actions. "
            "The external public-state V5-C selector will choose among them."
        ),
    }
    return prompts.get(system, common)


def postprocess_candidate(system: str, text: str, public_context: Any) -> str:
    if system not in {"memory_agent", "trust_score_agent"}:
        return text
    pkt, ok, _ = base.parse_action_json(text)
    if not ok:
        return text

    memory = pkt.setdefault("memory_update", {})
    if not isinstance(memory, dict):
        memory = {}
        pkt["memory_update"] = memory

    quarantine = memory.get("profile_quarantine", [])
    confirmed = memory.get("confirmed_profile", [])
    rejected = memory.get("rejected_profile", [])
    if not isinstance(quarantine, list):
        quarantine = []
    if not isinstance(confirmed, list):
        confirmed = []
    if not isinstance(rejected, list):
        rejected = []

    if system == "memory_agent":
        # Ordinary memory baseline: it has persistence, but no protected profile
        # quarantine. This makes T3 a direct test of memory poisoning.
        event_id = first_event_id(public_context)
        merged = list(dict.fromkeys([*confirmed, *quarantine, event_id]))
        memory["profile_quarantine"] = []
        memory["confirmed_profile"] = merged[:6]
        memory["rejected_profile"] = rejected[:4]
        pkt["trust_update"] = pkt.get("trust_update") or {"source_reliability": {}, "rationale": ""}
        if isinstance(pkt["trust_update"], dict):
            pkt["trust_update"]["rationale"] = "ordinary memory update without quarantine"

    if system == "trust_score_agent":
        # Simple trust-score baseline: preserve source-trust intuition, but do
        # not apply AgentGuard quarantine or trajectory-level safe governor.
        signals = base.public_signals(public_context)
        event_id = first_event_id(public_context)
        if signals.get("weak_source") or signals.get("suspicious"):
            memory["profile_quarantine"] = []
            memory["confirmed_profile"] = confirmed[:4]
            memory["rejected_profile"] = list(dict.fromkeys([*rejected, event_id]))[:6]
        else:
            memory["profile_quarantine"] = []
            memory["confirmed_profile"] = list(dict.fromkeys([*confirmed, event_id]))[:6]
            memory["rejected_profile"] = rejected[:4]
        pkt["trust_update"] = pkt.get("trust_update") or {"source_reliability": {}, "rationale": ""}
        if isinstance(pkt["trust_update"], dict):
            pkt["trust_update"]["rationale"] = "simple source trust score"
    return json_dumps(pkt)


def model_policy(system: str) -> str:
    if system == "react_base_tools":
        return "base_tools"
    if system in {"agentguard_zero_select", "agentguard_zero_train_select"}:
        return "agentguard_zero_select"
    return "zero_shot_vda"


def default_candidate_count(system: str, requested: int) -> int:
    if system in {"agentguard_zero_select", "agentguard_zero_train_select"}:
        return max(2, requested)
    return 1


def build_backend(args: argparse.Namespace) -> Any:
    if args.model_backend == "mock":
        return base.MockBackend(args.seed)
    if args.model_backend == "api":
        return base.APIBackend(args)
    return base.HFBackend(args)


def row_context(row: dict[str, Any], row_index: int, args: argparse.Namespace) -> tuple[list[dict[str, str]], Any, dict[str, Any], dict[str, Any], str, int, float]:
    messages, public_context = base.sanitize_initial_messages(base.as_messages(row.get("problem", "")))
    messages = prepend_system_prompt(messages, system_prompt(args.system))
    extra = base.scenario_extra_from_row(row)
    scenario = base.as_dict(extra.get("scenario"))
    scenario_id = str(extra.get("scenario_id", row.get("scenario_id", f"row-{row_index}")))
    max_env_steps = int(extra.get("max_env_steps", args.max_turns))
    budget = base.safe_float(scenario.get("defense_constraints", {}).get("business_budget", 5.0), 5.0)
    return messages, public_context, extra, scenario, scenario_id, max_env_steps, budget


def run_static_one(row: dict[str, Any], row_index: int, args: argparse.Namespace) -> dict[str, Any]:
    messages, public_context, extra, scenario, scenario_id, max_env_steps, budget = row_context(row, row_index, args)
    max_turns = min(args.max_turns, max_env_steps)
    trajectory_id = f"{args.run_name}-{args.system}-{row_index}-{scenario_id}"
    store = Level1RolloutStore(invalid_penalty=args.invalid_penalty)
    action_fn = STATIC_POLICIES[args.system]
    selected_actions: list[dict[str, Any]] = []
    final_observation: dict[str, Any] | None = None
    done = False

    for turn in range(max_turns):
        pkt = action_fn(public_context, scenario, turn)
        text = json_dumps(pkt)
        selected = base.score_candidate(public_context, text, selector_mode=args.selector_mode)
        selected_actions.append(
            {
                "turn": turn,
                "selected_text": text,
                "selected_packet": pkt,
                "selected_ok": selected.ok,
                "parse_msg": selected.parse_msg,
                "selector_score": selected.selector_score,
                "diagnostics": selected.diagnostics,
                "candidate_count": 1,
                "candidate_diagnostics": [selected.diagnostics],
            }
        )
        response = store.handle(
            {
                "trajectory_ids": [trajectory_id],
                "actions": [text],
                "finish": [False],
                "is_last_step": [turn == max_turns - 1],
                "extra_fields": [extra],
            }
        )
        final_observation = response["observations"][0]
        done = bool(response["dones"][0])
        messages.append({"role": "assistant", "content": text})
        if done:
            break
        user_msg, public_context = base.next_user_message(final_observation)
        messages.append(user_msg)

    return finalize_result(row, row_index, args, scenario_id, trajectory_id, done, selected_actions, final_observation, max_env_steps, budget)


def run_model_one(row: dict[str, Any], row_index: int, backend: Any, args: argparse.Namespace) -> dict[str, Any]:
    policy = model_policy(args.system)
    candidate_count = default_candidate_count(args.system, args.candidate_count)
    messages, public_context, extra, scenario, scenario_id, max_env_steps, budget = row_context(row, row_index, args)
    max_turns = min(args.max_turns, max_env_steps)
    trajectory_id = f"{args.run_name}-{args.system}-{row_index}-{scenario_id}"
    store = Level1RolloutStore(invalid_penalty=args.invalid_penalty)
    selected_actions: list[dict[str, Any]] = []
    final_observation: dict[str, Any] | None = None
    done = False

    for turn in range(max_turns):
        raw_candidates = backend.generate(messages, public_context, candidate_count)
        raw_candidates = [postprocess_candidate(args.system, item, public_context) for item in raw_candidates]
        selected = base.select_candidate(public_context, raw_candidates, policy, selector_mode=args.selector_mode)
        selected_actions.append(
            {
                "turn": turn,
                "selected_text": selected.text,
                "selected_packet": selected.packet,
                "selected_ok": selected.ok,
                "parse_msg": selected.parse_msg,
                "selector_score": selected.selector_score,
                "diagnostics": selected.diagnostics,
                "candidate_count": len(raw_candidates),
                "candidate_diagnostics": [
                    base.score_candidate(public_context, cand, selector_mode=args.selector_mode).diagnostics
                    for cand in raw_candidates
                ],
            }
        )
        response = store.handle(
            {
                "trajectory_ids": [trajectory_id],
                "actions": [selected.text],
                "finish": [False],
                "is_last_step": [turn == max_turns - 1],
                "extra_fields": [extra],
            }
        )
        final_observation = response["observations"][0]
        done = bool(response["dones"][0])
        messages.append({"role": "assistant", "content": selected.text})
        if done:
            break
        user_msg, public_context = base.next_user_message(final_observation)
        messages.append(user_msg)

    return finalize_result(row, row_index, args, scenario_id, trajectory_id, done, selected_actions, final_observation, max_env_steps, budget)


def run_model_many(
    indexed_rows: list[tuple[int, dict[str, Any]]], backend: Any, args: argparse.Namespace
) -> list[dict[str, Any]]:
    policy = model_policy(args.system)
    candidate_count = default_candidate_count(args.system, args.candidate_count)
    store = Level1RolloutStore(invalid_penalty=args.invalid_penalty)
    states: list[dict[str, Any]] = []
    for row_index, row in indexed_rows:
        messages, public_context, extra, scenario, scenario_id, max_env_steps, budget = row_context(
            row, row_index, args
        )
        states.append(
            {
                "row": row,
                "row_index": row_index,
                "initial_messages": messages,
                "public_context": public_context,
                "history": [],
                "extra": extra,
                "scenario": scenario,
                "scenario_id": scenario_id,
                "max_env_steps": max_env_steps,
                "max_turns": min(args.max_turns, max_env_steps),
                "budget": budget,
                "trajectory_id": f"{args.run_name}-{args.system}-{row_index}-{scenario_id}",
                "selected_actions": [],
                "final_observation": None,
                "done": False,
            }
        )

    max_turns = max((state["max_turns"] for state in states), default=0)
    for turn in range(max_turns):
        active = [
            index
            for index, state in enumerate(states)
            if not state["done"] and turn < state["max_turns"]
        ]
        if not active:
            break
        message_batches = [_generation_messages(states[index]) for index in active]
        contexts = [states[index]["public_context"] for index in active]
        if hasattr(backend, "generate_batch"):
            candidate_batches = backend.generate_batch(message_batches, contexts, candidate_count)
        else:
            candidate_batches = [
                backend.generate(messages, context, candidate_count)
                for messages, context in zip(message_batches, contexts)
            ]

        selected_values = []
        for index, raw_candidates in zip(active, candidate_batches):
            state = states[index]
            raw_candidates = [
                postprocess_candidate(args.system, item, state["public_context"])
                for item in raw_candidates
            ]
            selected = base.select_candidate(
                state["public_context"], raw_candidates, policy, selector_mode=args.selector_mode
            )
            selected_values.append(selected)
            state["selected_actions"].append(
                {
                    "turn": turn,
                    "selected_text": selected.text,
                    "selected_packet": selected.packet,
                    "selected_ok": selected.ok,
                    "parse_msg": selected.parse_msg,
                    "selector_score": selected.selector_score,
                    "diagnostics": selected.diagnostics,
                    "candidate_count": len(raw_candidates),
                    "candidate_diagnostics": [
                        base.score_candidate(
                            state["public_context"], candidate, selector_mode=args.selector_mode
                        ).diagnostics
                        for candidate in raw_candidates
                    ],
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
                user_message, public_context = base.next_user_message(observation)
                del user_message
                state["history"].append(
                    _history_summary(
                        turn,
                        public_context,
                        selected_values[position].packet,
                    )
                )
                state["public_context"] = public_context

    return [
        finalize_result(
            state["row"],
            state["row_index"],
            args,
            state["scenario_id"],
            state["trajectory_id"],
            state["done"],
            state["selected_actions"],
            state["final_observation"],
            state["max_env_steps"],
            state["budget"],
        )
        for state in states
    ]


def load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenario_id = item.get("scenario_id") if isinstance(item, dict) else None
            if scenario_id:
                results[str(scenario_id)] = item
    return results


def append_progress(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json_dumps(result) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def betrayal_detected(actions: list[dict[str, Any]], score: dict[str, Any], task: str) -> float:
    if "Betrayal" not in task:
        return math.nan
    if bool(score.get("overresponse", False)) or int(base.safe_float(score.get("fake_confirmed", 0), 0)) > 0:
        return 0.0
    for item in actions:
        packet = item.get("selected_packet", {}) if isinstance(item, dict) else {}
        tool = str((packet.get("tool_call", {}) or {}).get("name", "None")) if isinstance(packet, dict) else "None"
        memory = packet.get("memory_update", {}) if isinstance(packet, dict) else {}
        if tool in {"CrossCheck", "ProvenanceCheck", "GraphQuery"}:
            return 1.0
        if isinstance(memory, dict) and (memory.get("profile_quarantine") or memory.get("rejected_profile")):
            return 1.0
    return 0.0


def finalize_result(
    row: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    scenario_id: str,
    trajectory_id: str,
    done: bool,
    selected_actions: list[dict[str, Any]],
    final_observation: dict[str, Any] | None,
    max_env_steps: int,
    budget: float,
) -> dict[str, Any]:
    score = dict((final_observation or {}).get("score", {}))
    steps = int(score.get("steps", len(selected_actions)))
    safe_utility = base.compute_safe_utility(score, steps=steps, max_steps=max_env_steps, budget=budget)
    score["safe_utility"] = safe_utility
    task = task_name(row)
    selected_parse_failures = sum(1 for item in selected_actions if str(item["parse_msg"]).startswith("json_parse_error"))
    selected_invalid_tools = sum(1 for item in selected_actions if item["parse_msg"] == "invalid_tool")
    selected_invalid_actions = sum(1 for item in selected_actions if item["parse_msg"] == "invalid_response_action")
    poison_success = float(int(base.safe_float(score.get("fake_confirmed", 0), 0.0) > 0.0))
    result = {
        "row_index": row_index,
        "scenario_id": scenario_id,
        "split": row.get("split", "unknown"),
        "cfc_split": row.get("cfc_split", ""),
        "task": task,
        "system": args.system,
        "system_display": SYSTEM_DISPLAY.get(args.system, args.system),
        "policy": model_policy(args.system) if args.system in MODEL_SYSTEMS else args.system,
        "trajectory_id": trajectory_id,
        "done": done,
        "steps": steps,
        "score": score,
        "tmcd_metrics": {
            "safe_utility": safe_utility,
            "attack_mitigation": float(bool(score.get("attack_mitigated", False))),
            "betrayal_detection": betrayal_detected(selected_actions, score, task),
            "poison_success": poison_success,
            "overresponse": float(bool(score.get("overresponse", False))),
            "business_cost": base.safe_float(score.get("business_cost", 0.0), 0.0),
            "verification_cost": base.safe_float(score.get("verification_cost", 0.0), 0.0),
        },
        "selected_json_parse_failures": selected_parse_failures,
        "selected_invalid_tool_calls": selected_invalid_tools,
        "selected_invalid_response_actions": selected_invalid_actions,
        "selected_actions": selected_actions,
    }
    return result


def load_rows(path: str, split: str, limit: int | None, seed: int, offset: int = 0, task_filter: str = "") -> list[dict[str, Any]]:
    rows = base.load_rows(path, split=split, limit=None, seed=seed, offset=offset)
    if task_filter:
        wanted = {item.strip() for item in task_filter.split(",") if item.strip()}
        rows = [row for row in rows if task_name(row) in wanted or str(row.get("cfc_split", "")) in wanted]
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = len(results)
    action_count = sum(max(1, len(item.get("selected_actions", []))) for item in results)
    raw_candidate_count = 0
    raw_candidate_json_failures = 0
    selector_fallbacks = 0
    selector_governor_overrides = 0
    for item in results:
        for action in item.get("selected_actions", []):
            diagnostics = action.get("diagnostics", {}) if isinstance(action, dict) else {}
            selector_fallbacks += int(bool(diagnostics.get("fallback")))
            selector_governor_overrides += int(bool(diagnostics.get("governor_override")))
            for diag in action.get("candidate_diagnostics", []):
                if isinstance(diag, dict):
                    raw_candidate_count += 1
                    raw_candidate_json_failures += int(bool(diag.get("json_parse_failure")))

    betrayal_values = [
        value
        for item in results
        for value in [item.get("tmcd_metrics", {}).get("betrayal_detection", math.nan)]
        if not math.isnan(float(value))
    ]
    summary = {
        "run_name": args.run_name,
        "system": args.system,
        "system_display": SYSTEM_DISPLAY.get(args.system, args.system),
        "model_backend": args.model_backend if args.system in MODEL_SYSTEMS else "static",
        "model_path": args.model_path if args.system in MODEL_SYSTEMS else "",
        "adapter_path": (
            args.adapter_path
            if args.system in {"agentguard_zero_train", "agentguard_zero_train_select"}
            else ""
        ),
        "api_model": args.api_model if args.model_backend == "api" else "",
        "num_scenarios": n,
        "candidate_count": default_candidate_count(args.system, args.candidate_count) if args.system in MODEL_SYSTEMS else 1,
        "selector_mode": (
            args.selector_mode
            if args.system in {"agentguard_zero_select", "agentguard_zero_train_select"}
            else ""
        ),
        "offset": args.offset,
        "safe_utility": mean([item["tmcd_metrics"]["safe_utility"] for item in results]),
        "trajectory_reward": mean([base.safe_float(item["score"].get("reward", 0.0), 0.0) for item in results]),
        "safe_success_rate": mean([float(bool(item["score"].get("safe_success", False))) for item in results]),
        "attack_mitigation": mean([item["tmcd_metrics"]["attack_mitigation"] for item in results]),
        "attack_success": mean([float(bool(item["score"].get("attack_success", False))) for item in results]),
        "intent_accuracy": mean([float(bool(item["score"].get("correct_intent", False))) for item in results]),
        "betrayal_detection": mean([float(v) for v in betrayal_values]) if betrayal_values else math.nan,
        "poison_success": mean([item["tmcd_metrics"]["poison_success"] for item in results]),
        "business_cost": mean([item["tmcd_metrics"]["business_cost"] for item in results]),
        "verification_cost": mean([item["tmcd_metrics"]["verification_cost"] for item in results]),
        "overresponse_rate": mean([item["tmcd_metrics"]["overresponse"] for item in results]),
        "json_parse_failure_rate": sum(item["selected_json_parse_failures"] for item in results) / max(1, action_count),
        "raw_candidate_json_parse_failure_rate": raw_candidate_json_failures / max(1, raw_candidate_count),
        "selector_fallback_rate": selector_fallbacks / max(1, action_count),
        "selector_governor_override_rate": selector_governor_overrides / max(1, action_count),
        "invalid_tool_call_rate": sum(item["selected_invalid_tool_calls"] for item in results) / max(1, action_count),
        "invalid_response_action_rate": sum(item["selected_invalid_response_actions"] for item in results) / max(1, action_count),
        "avg_steps": mean([float(item.get("steps", 0)) for item in results]),
    }
    task_groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        task_groups.setdefault(str(item.get("task", "unknown")), []).append(item)
    summary["task_safe_utility"] = {
        task: mean([entry["tmcd_metrics"]["safe_utility"] for entry in entries])
        for task, entries in sorted(task_groups.items())
    }
    return summary


def write_outputs(results: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json_dumps(item) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        f.write(json_dumps(summary, indent=2) + "\n")
    with (output_dir / "tmcd_table_row.md").open("w", encoding="utf-8") as f:
        f.write("| System | Safe Utility | Attack Mitigation | Betrayal Detection | Poison Success | Overresponse | Business Cost |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        betrayal = summary.get("betrayal_detection")
        betrayal_text = "" if isinstance(betrayal, float) and math.isnan(betrayal) else f"{float(betrayal):.6f}"
        f.write(
            "| {system} | {safe:.6f} | {mit:.6f} | {betrayal} | {poison:.6f} | {over:.6f} | {cost:.6f} |\n".format(
                system=summary.get("system_display", summary.get("system", "")),
                safe=summary.get("safe_utility", 0.0),
                mit=summary.get("attack_mitigation", 0.0),
                betrayal=betrayal_text,
                poison=summary.get("poison_success", 0.0),
                over=summary.get("overresponse_rate", 0.0),
                cost=summary.get("business_cost", 0.0),
            )
        )


def resolve_model_path(args: argparse.Namespace) -> None:
    if args.system not in MODEL_SYSTEMS:
        return
    if args.model_path:
        return
    if args.system == "cyber_llm_vda":
        args.model_path = os.environ.get("AGZ_CYBER_LLM_MODEL_PATH", "")
    elif args.system == "lily_cybersecurity_vda":
        args.model_path = os.environ.get("AGZ_LILY_CYBER_MODEL_PATH", "")
    else:
        args.model_path = os.environ.get("AGZ_QWEN35_9B_PATH", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--system", choices=SYSTEMS + sorted(INTERNAL_SYSTEMS), required=True)
    parser.add_argument("--model_path", default=os.environ.get("AGZ_MODEL_PATH", ""))
    parser.add_argument("--adapter_path", default=os.environ.get("AGZ_ADAPTER_PATH", ""))
    parser.add_argument("--model_backend", choices=["hf", "mock", "api"], default=os.environ.get("AGZ_MODEL_BACKEND", "hf"))
    parser.add_argument("--candidate_count", type=int, default=int(os.environ.get("AGZ_CANDIDATE_COUNT", "4")))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("AGZ_EVAL_LIMIT", "16")))
    parser.add_argument("--offset", type=int, default=int(os.environ.get("AGZ_EVAL_OFFSET", "0")))
    parser.add_argument("--split", default=os.environ.get("AGZ_EVAL_SPLIT", "all"))
    parser.add_argument("--task_filter", default=os.environ.get("AGZ_TASK_FILTER", ""))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("AGZ_EVAL_SEED", "20260708")))
    parser.add_argument("--selector_mode", default=os.environ.get("AGZ_SELECTOR_MODE", "v5_c_frontier_minimax"))
    parser.add_argument("--run_name", default="")
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "tmcd_eval"))
    parser.add_argument("--max_turns", type=int, default=int(os.environ.get("AGZ_AGENT_MAX_TURNS", "5")))
    parser.add_argument("--invalid_penalty", type=float, default=float(os.environ.get("AGZ_INVALID_PENALTY", "0.5")))
    parser.add_argument("--max_input_tokens", type=int, default=int(os.environ.get("AGZ_MAX_INPUT_TOKENS", "2048")))
    parser.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("AGZ_MAX_NEW_TOKENS", "256")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("AGZ_TEMPERATURE", "0.7")))
    parser.add_argument("--top_p", type=float, default=float(os.environ.get("AGZ_TOP_P", "1.0")))
    parser.add_argument("--top_k", type=int, default=int(os.environ.get("AGZ_TOP_K", "0")))
    parser.add_argument("--stop_on_complete_json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory_batch_size", type=int, default=int(os.environ.get("AGZ_TRAJECTORY_BATCH_SIZE", "16")))
    parser.add_argument("--num_shards", type=int, default=int(os.environ.get("AGZ_EVAL_NUM_SHARDS", "1")))
    parser.add_argument("--shard_index", type=int, default=int(os.environ.get("AGZ_EVAL_SHARD_INDEX", "0")))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default=os.environ.get("AGZ_DTYPE", "bf16"))
    parser.add_argument("--device_map", default=os.environ.get("AGZ_EVAL_DEVICE_MAP", ""))
    parser.add_argument("--attn_implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default=os.environ.get("AGZ_ATTN_IMPLEMENTATION", "auto"))
    parser.add_argument("--api_model", default=os.environ.get("AGZ_API_MODEL", os.environ.get("LLM_MODEL", "")))
    parser.add_argument("--api_base_url", default=os.environ.get("AGZ_API_BASE_URL", os.environ.get("LLM_BASE_URL", "")))
    parser.add_argument("--api_key_env", default=os.environ.get("AGZ_API_KEY_ENV", ""))
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("AGZ_API_TIMEOUT", "90")))
    parser.add_argument("--api_retries", type=int, default=int(os.environ.get("AGZ_API_RETRIES", "2")))
    parser.add_argument("--api_response_format_json", action="store_true", default=base.env_flag("AGZ_API_RESPONSE_FORMAT_JSON", False))
    parser.add_argument("--api_disable_thinking", action="store_true", default=base.env_flag("AGZ_API_DISABLE_THINKING", False))
    parser.add_argument("--api_multi_choice", action="store_true", default=base.env_flag("AGZ_API_MULTI_CHOICE", False))
    parser.add_argument("--api_system_prompt", default=os.environ.get("AGZ_API_SYSTEM_PROMPT", "Return compact strict JSON only."))
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard_index must satisfy 0 <= shard_index < num_shards")
    if args.trajectory_batch_size <= 0:
        parser.error("trajectory_batch_size must be positive")
    if not args.run_name:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.system}_{stamp}"
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    resolve_model_path(args)
    if args.system in MODEL_SYSTEMS and args.model_backend == "hf" and not args.model_path:
        raise SystemExit(f"{args.system} requires --model_path or AGZ_MODEL_PATH.")
    rows = load_rows(args.data, split=args.split, limit=args.limit, seed=args.seed, offset=args.offset, task_filter=args.task_filter)
    if not rows:
        raise SystemExit(f"No rows loaded from {args.data}.")

    backend = None
    if args.system in MODEL_SYSTEMS:
        backend = build_backend(args)

    indexed_rows = [(args.offset + index, row) for index, row in enumerate(rows)]
    indexed_rows = indexed_rows[args.shard_index :: args.num_shards]
    output_dir = Path(args.output_dir) / args.run_name
    if args.num_shards > 1:
        output_dir = output_dir / f"shard_{args.shard_index}"
    progress_path = output_dir / "progress.jsonl"
    run_config_path = output_dir / "run_config.json"
    run_config = {
        "run_name": args.run_name,
        "data": str(Path(args.data).resolve()),
        "system": args.system,
        "model_backend": args.model_backend,
        "api_model": args.api_model,
        "model_path": str(Path(args.model_path).resolve()) if args.model_path else "",
        "adapter_path": str(Path(args.adapter_path).resolve()) if args.adapter_path else "",
        "candidate_count": args.candidate_count,
        "selector_mode": args.selector_mode,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "offset": args.offset,
    }
    if run_config_path.exists() and args.resume:
        if json.loads(run_config_path.read_text(encoding="utf-8")) != run_config:
            raise RuntimeError(f"evaluation resume config mismatch: {run_config_path}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path.unlink(missing_ok=True)
        run_config_path.write_text(json_dumps(run_config, indent=2) + "\n", encoding="utf-8")
    existing = load_progress(progress_path) if args.resume else {}
    pending = [
        (index, row)
        for index, row in indexed_rows
        if str(base.scenario_extra_from_row(row).get("scenario_id", row.get("scenario_id", index))) not in existing
    ]
    completed = 0
    for start in range(0, len(pending), args.trajectory_batch_size):
        batch = pending[start : start + args.trajectory_batch_size]
        if args.system in STATIC_POLICIES:
            batch_results = [run_static_one(row, index, args) for index, row in batch]
        else:
            effective_size = max(1, args.trajectory_batch_size // default_candidate_count(args.system, args.candidate_count))
            batch_results = []
            for sub_start in range(0, len(batch), effective_size):
                batch_results.extend(run_model_many(batch[sub_start : sub_start + effective_size], backend, args))
        append_progress(progress_path, batch_results)
        for result in batch_results:
            existing[str(result["scenario_id"])] = result
        completed += len(batch_results)
        print(
            json_dumps(
                {
                    "progress": f"{completed}/{len(pending)}",
                    "cached": len(existing) - completed,
                    "system": args.system,
                    "shard": f"{args.shard_index}/{args.num_shards}",
                }
            ),
            flush=True,
        )

    wanted_ids = {
        str(base.scenario_extra_from_row(row).get("scenario_id", row.get("scenario_id", index)))
        for index, row in indexed_rows
    }
    results = sorted(
        (result for scenario_id, result in existing.items() if scenario_id in wanted_ids),
        key=lambda item: int(item.get("row_index", 0)),
    )

    summary = summarize(results, args)
    if backend is not None and hasattr(backend, "usage"):
        summary.update(getattr(backend, "usage"))
    write_outputs(results, summary, output_dir)
    print(json_dumps({"summary": summary, "output_dir": str(output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
