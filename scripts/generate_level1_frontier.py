#!/usr/bin/env python3
"""Generate Level-1 AgentGuard-Zero scenarios and CFC frontier parquet.

This is a safe symbolic generator. It does not create payloads, exploit steps,
malware behavior, real IPs, or real organizations. Its purpose is to build a
reproducible data pool for VDA warmup/RL experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Callable

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agentguard_zero.env.checker import full_check
from agentguard_zero.env.oracle import score_trajectory
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.schemas.action_schema import DEFAULT_ACTION_PACKET, OBJECTIVE_KEYS
from agentguard_zero.training.vda_dataset import scenario_to_training_row


SPLITS = [
    "Evidence-Level Profile Poisoning",
    "Long-Horizon Trust-Building Betrayal",
    "Objective / Strategy Switching",
    "Business-Constrained Overreaction Induction",
]

SOURCE_POOL = [
    "host_log",
    "identity_log",
    "network_flow",
    "partner_ticket",
    "automation_note",
    "endpoint_summary",
    "decoy_sensor",
    "collaboration_chat",
]

PHASES_BY_OBJECTIVE = {
    "exfiltration": ["initial_access", "discovery", "lateral_movement", "collection", "exfiltration"],
    "sabotage": ["initial_access", "discovery", "lateral_movement", "impact"],
    "persistence": ["initial_access", "discovery", "lateral_movement", "persistence"],
    "credential_theft": ["initial_access", "discovery", "collection", "exfiltration"],
}

FAKE_CLAIMS = {
    "exfiltration": [
        "bulk metadata staging marker",
        "archive-like collection cadence",
        "external transfer preparation hint",
        "unusual dataset access storyline",
    ],
    "sabotage": [
        "availability stress rehearsal marker",
        "service disruption storyline",
        "destructive configuration drift hint",
        "backup integrity concern",
    ],
    "persistence": [
        "recurring access foothold marker",
        "durable session storyline",
        "scheduled access anomaly",
        "redundant access path hint",
    ],
    "credential_theft": [
        "identity token access hint",
        "credential reuse storyline",
        "privilege lookup anomaly",
        "login correlation concern",
    ],
}

ASSET_TEMPLATES = [
    [
        ("workstation", "office", 0.20),
        ("file_server", "server", 0.70),
        ("database", "server", 0.95),
        ("backup_service", "server", 0.80),
    ],
    [
        ("contractor_laptop", "partner", 0.25),
        ("identity_provider", "identity", 0.90),
        ("admin_console", "control", 0.85),
        ("payment_api", "production", 0.95),
    ],
    [
        ("analyst_host", "office", 0.25),
        ("build_runner", "dev", 0.75),
        ("artifact_store", "dev", 0.80),
        ("analytics_db", "server", 0.90),
    ],
    [
        ("support_portal", "dmz", 0.45),
        ("case_queue", "ops", 0.60),
        ("identity_provider", "identity", 0.90),
        ("customer_db", "server", 0.95),
    ],
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _stable_hash(value: Any) -> str:
    return hashlib.md5(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _objective_belief(objective: str) -> dict[str, float]:
    belief = {key: 0.05 for key in OBJECTIVE_KEYS}
    if objective in belief:
        belief[objective] = 0.85
    else:
        equal = 1.0 / len(OBJECTIVE_KEYS)
        belief = {key: equal for key in OBJECTIVE_KEYS}
    return belief


def _action_packet(
    belief_objective: str,
    tool_call: dict[str, Any],
    memory_update: dict[str, Any],
    response_action: str,
    target: str,
    tier: str = "L1",
) -> dict[str, Any]:
    packet = json.loads(json.dumps(DEFAULT_ACTION_PACKET))
    packet["belief"] = _objective_belief(belief_objective)
    packet["evidence_assessment"] = []
    packet["trust_update"] = {"source_reliability": {}, "rationale": "scripted CFC proxy policy"}
    packet["uncertainty"] = 0.35
    packet["tool_call"] = tool_call
    packet["memory_update"] = memory_update
    packet["safety_check"] = {
        "business_risk": 0.3,
        "overresponse_risk": 0.3,
        "justification": "symbolic CFC proxy policy",
    }
    packet["response"] = {"tier": tier, "action": response_action, "target": target}
    return packet


def _build_assets(rng: random.Random, idx: int) -> tuple[list[dict[str, Any]], list[list[str]]]:
    template = rng.choice(ASSET_TEMPLATES)
    prefix = f"site{idx % 97:02d}"
    assets = [{"id": f"{prefix}_{name}", "zone": zone, "criticality": crit} for name, zone, crit in template]
    edges = [[assets[i]["id"], assets[i + 1]["id"]] for i in range(len(assets) - 1)]
    if len(assets) >= 4 and rng.random() < 0.55:
        edges.append([assets[1]["id"], assets[-1]["id"]])
    return assets, edges


def _target_for_objective(objective: str, assets: list[dict[str, Any]]) -> str:
    by_name = {asset["id"]: asset for asset in assets}
    candidates = list(by_name)
    if objective == "credential_theft":
        for name in candidates:
            if "identity" in name:
                return name
    if objective == "persistence":
        for name in candidates:
            if "runner" in name or "console" in name or "provider" in name:
                return name
    if objective == "sabotage":
        for name in candidates:
            if "payment" in name or "backup" in name or "database" in name or "db" in name:
                return name
    return max(assets, key=lambda item: float(item.get("criticality", 0.5)))["id"]


def _source_reliability(rng: random.Random, split: str) -> dict[str, float]:
    reliabilities = {}
    for source in SOURCE_POOL:
        base = rng.uniform(0.35, 0.85)
        if split == "Long-Horizon Trust-Building Betrayal" and source in {"partner_ticket", "collaboration_chat"}:
            base = rng.uniform(0.70, 0.93)
        if split == "Evidence-Level Profile Poisoning" and source in {"host_log", "endpoint_summary"}:
            base = rng.uniform(0.55, 0.88)
        reliabilities[source] = round(base, 3)
    return reliabilities


def _phase_schedule(rng: random.Random, objective: str, split: str) -> list[str]:
    phases = list(PHASES_BY_OBJECTIVE[objective])
    if split == "Long-Horizon Trust-Building Betrayal":
        if "collection" not in phases:
            phases.insert(-1, "collection")
        if len(phases) < 6:
            phases.insert(2, "discovery")
    if split == "Objective / Strategy Switching" and len(phases) < 6:
        phases.insert(2, "lateral_movement")
    if rng.random() < 0.25 and len(phases) < 7:
        phases.insert(max(1, len(phases) - 1), rng.choice(["discovery", "collection", "persistence"]))
    return phases[:7]


def _fake_event_times(rng: random.Random, split: str, max_time: int, count: int) -> list[int]:
    if split == "Evidence-Level Profile Poisoning":
        pool = list(range(0, min(max_time, 3) + 1))
    elif split == "Long-Horizon Trust-Building Betrayal":
        pool = list(range(1, max_time + 1))
    elif split == "Objective / Strategy Switching":
        pool = list(range(0, max_time + 1))
    else:
        pool = list(range(0, min(max_time, 2) + 1))
    if not pool:
        pool = [0]
    return sorted(rng.choice(pool) for _ in range(count))


def generate_scenario(idx: int, rng: random.Random, seed: int) -> dict[str, Any]:
    split = SPLITS[idx % len(SPLITS)]
    true_objective = rng.choice(OBJECTIVE_KEYS)
    false_choices = [obj for obj in OBJECTIVE_KEYS if obj != true_objective]
    if split == "Business-Constrained Overreaction Induction" and true_objective != "sabotage":
        false_objective = "sabotage"
    else:
        false_objective = rng.choice(false_choices)

    assets, edges = _build_assets(rng, idx)
    entry = assets[0]["id"]
    target = _target_for_objective(true_objective, assets)
    phases = _phase_schedule(rng, true_objective, split)
    reliabilities = _source_reliability(rng, split)

    if split == "Evidence-Level Profile Poisoning":
        fake_count = rng.randint(2, 4)
    elif split == "Long-Horizon Trust-Building Betrayal":
        fake_count = rng.randint(2, 3)
    elif split == "Objective / Strategy Switching":
        fake_count = rng.randint(1, 3)
    else:
        fake_count = rng.randint(1, 3)

    fake_times = _fake_event_times(rng, split, max(len(phases) - 1, 0), fake_count)
    fake_evidence = []
    for ev_idx, time in enumerate(fake_times):
        source = rng.choice(SOURCE_POOL)
        ev_type = rng.choice(["fake_log", "fake_ttp", "suppressed_signal"])
        if split == "Evidence-Level Profile Poisoning":
            spoofability = rng.uniform(0.62, 0.90)
        elif split == "Long-Horizon Trust-Building Betrayal":
            spoofability = rng.uniform(0.50, 0.78)
        elif split == "Objective / Strategy Switching":
            spoofability = rng.uniform(0.56, 0.86)
        else:
            spoofability = rng.uniform(0.58, 0.93)
        claim = rng.choice(FAKE_CLAIMS[false_objective])
        fake_evidence.append(
            {
                "time": int(time),
                "type": ev_type,
                "source": source,
                "claim": claim,
                "intended_misbelief": false_objective,
                "spoofability": round(spoofability, 3),
            }
        )

    decoy_behavior = []
    if split in {"Long-Horizon Trust-Building Betrayal", "Business-Constrained Overreaction Induction"} or rng.random() < 0.35:
        decoy_behavior.append(
            {
                "time": min(max(len(phases) - 2, 1), rng.randint(1, max(len(phases) - 1, 1))),
                "decoy_type": rng.choice(["backup_service_decoy", "identity_prompt_decoy", "admin_console_decoy"]),
                "intended_misbelief": false_objective,
            }
        )

    strategy_switch = None
    if split == "Objective / Strategy Switching" or rng.random() < 0.20:
        switch_time = max(1, min(len(phases) - 1, rng.randint(1, max(len(phases) - 1, 1))))
        strategy_switch = {
            "time": switch_time,
            "visible_profile_before": false_objective,
            "true_behavior_after": true_objective,
        }

    if split == "Business-Constrained Overreaction Induction":
        business_budget = round(rng.uniform(1.8, 3.8), 2)
        max_high = rng.choice([0, 1])
    else:
        business_budget = round(rng.uniform(4.0, 7.5), 2)
        max_high = rng.choice([1, 2])
    verification_budget = rng.randint(2, 6)
    if split == "Evidence-Level Profile Poisoning":
        verification_budget = max(2, min(verification_budget, fake_count + 1))

    trust_building_plan = []
    trust_steps = 1
    if split == "Long-Horizon Trust-Building Betrayal":
        trust_steps = rng.randint(3, 5)
    elif split == "Evidence-Level Profile Poisoning":
        trust_steps = rng.randint(1, 3)
    for step in range(trust_steps):
        trust_building_plan.append(
            {
                "time": step,
                "source": rng.choice(["partner_ticket", "automation_note", "collaboration_chat"]),
                "claim": "benign coordination signal",
                "trust_gain": round(rng.uniform(0.05, 0.20), 3),
            }
        )

    betrayal_time = min(len(phases) - 1, max(1, len(trust_building_plan)))
    scenario = {
        "scenario_id": f"L1-S{seed}-{idx:06d}",
        "actors": [
            {"id": "vda", "role": "verification_defense_agent"},
            {"id": "dca", "role": "deceptive_curriculum_agent"},
            {"id": "partner_feed", "role": "noisy_external_signal"},
        ],
        "collaboration_context": {
            "ticket_type": rng.choice(["incident_triage", "service_recovery", "identity_review", "data_access_review"]),
            "coordination_channel": rng.choice(["case_queue", "partner_ticket", "automation_note"]),
        },
        "trust_building_plan": trust_building_plan,
        "betrayal_trigger": {
            "time": betrayal_time,
            "condition": "defender accepts profile update without cross-source verification",
        },
        "source_reliability": reliabilities,
        "strategy_switch": strategy_switch,
        "network_context": {"assets": assets, "reachable_edges": edges},
        "true_attack": {
            "objective": true_objective,
            "entry": entry,
            "target": target,
            "phase_schedule": phases,
        },
        "poisoning_plan": {
            "false_objective": false_objective,
            "fake_evidence": fake_evidence,
            "decoy_behavior": decoy_behavior,
            "strategy_switch": strategy_switch,
        },
        "defense_constraints": {
            "business_budget": business_budget,
            "verification_budget": verification_budget,
            "max_high_impact_actions": max_high,
        },
        "oracle": {
            "true_objective": true_objective,
            "success_condition": f"prevent_{true_objective}",
            "overresponse_condition": "high_impact_action_without_verified_risk",
        },
        "metadata": {
            "generator": "agentguard_zero_symbolic_v1",
            "generation_seed": seed,
            "candidate_index": idx,
            "cfc_split": split,
        },
    }
    return scenario


def _event_objective(event: dict[str, Any], fallback: str) -> str:
    hint = event.get("objective_hint") or event.get("intended_misbelief") or fallback
    return str(hint) if hint in OBJECTIVE_KEYS else fallback


def _pick_fake_like(obs: dict[str, Any]) -> dict[str, Any] | None:
    fake_events = [event for event in obs.get("observed_events", []) if event.get("is_fake")]
    if not fake_events:
        return None
    return max(fake_events, key=lambda event: float(event.get("spoofability", 0.0)))


def _pick_true_like(obs: dict[str, Any]) -> dict[str, Any] | None:
    for event in obs.get("observed_events", []):
        if not event.get("is_fake"):
            return event
    return None


def active_verification_policy(obs: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    fake_like = _pick_fake_like(obs)
    true_like = _pick_true_like(obs)
    target = scenario["true_attack"]["entry"]
    if fake_like is not None:
        event_id = str(fake_like.get("event_id", "unknown"))
        return _action_packet(
            belief_objective=_event_objective(true_like or {}, scenario["oracle"]["true_objective"]),
            tool_call={"name": "CrossCheck", "args": {"event_id": event_id, "sources": ["network_flow", "identity_log", "host_log"]}},
            memory_update={"profile_quarantine": [event_id], "confirmed_profile": [], "rejected_profile": []},
            response_action="Observe",
            target="none",
            tier="L1",
        )
    if true_like is not None:
        event_id = str(true_like.get("event_id", "unknown"))
        return _action_packet(
            belief_objective=_event_objective(true_like, scenario["oracle"]["true_objective"]),
            tool_call={"name": "None", "args": {}},
            memory_update={"profile_quarantine": [], "confirmed_profile": [event_id], "rejected_profile": []},
            response_action="LimitSession",
            target=target,
            tier="L2",
        )
    return _action_packet(
        belief_objective=scenario["oracle"]["true_objective"],
        tool_call={"name": "None", "args": {}},
        memory_update={"profile_quarantine": [], "confirmed_profile": [], "rejected_profile": []},
        response_action="Observe",
        target="none",
        tier="L0",
    )


def passive_susceptible_policy(obs: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    fake_like = _pick_fake_like(obs)
    true_like = _pick_true_like(obs)
    target = scenario["true_attack"]["target"]
    entry = scenario["true_attack"]["entry"]
    split = scenario.get("metadata", {}).get("cfc_split", "")
    if fake_like is not None:
        event_id = str(fake_like.get("event_id", "unknown"))
        action = "Isolate" if split == "Business-Constrained Overreaction Induction" else "LimitSession"
        action_target = target if action == "Isolate" else entry
        return _action_packet(
            belief_objective=_event_objective(fake_like, scenario["poisoning_plan"]["false_objective"]),
            tool_call={"name": "None", "args": {}},
            memory_update={"profile_quarantine": [], "confirmed_profile": [event_id], "rejected_profile": []},
            response_action=action,
            target=action_target,
            tier="L2" if action != "Isolate" else "L3",
        )
    if true_like is not None:
        event_id = str(true_like.get("event_id", "unknown"))
        return _action_packet(
            belief_objective=_event_objective(true_like, scenario["oracle"]["true_objective"]),
            tool_call={"name": "None", "args": {}},
            memory_update={"profile_quarantine": [], "confirmed_profile": [event_id], "rejected_profile": []},
            response_action="LimitSession",
            target=entry,
            tier="L2",
        )
    return _action_packet(
        belief_objective=scenario["poisoning_plan"]["false_objective"],
        tool_call={"name": "None", "args": {}},
        memory_update={"profile_quarantine": [], "confirmed_profile": [], "rejected_profile": []},
        response_action="Observe",
        target="none",
        tier="L0",
    )


def rollout_with_policy(
    scenario: dict[str, Any],
    policy: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    env = instantiate_scenario(scenario)
    done = False
    steps = 0
    while not done and steps <= env.max_steps:
        obs = env.observe()
        packet = policy(obs, scenario)
        _, _, done = env.step(packet)
        steps += 1
    outcome = score_trajectory(
        scenario=scenario,
        history=env.history,
        memory=env.memory,
        attack_mitigated=env.attack_mitigated,
        attack_success=env.attack_success,
        business_cost=env.business_cost,
        verification_cost=env.verification_cost,
        high_impact_count=env.high_impact_count,
    )
    outcome["steps"] = steps
    return outcome


def _target_criticality(scenario: dict[str, Any]) -> float:
    target = scenario["true_attack"]["target"]
    for asset in scenario["network_context"].get("assets", []):
        if asset.get("id") == target:
            return float(asset.get("criticality", 0.5))
    return 0.5


def compute_cfc_metrics(scenario: dict[str, Any]) -> dict[str, Any]:
    checks = full_check(scenario)
    if not checks.get("all_ok", False):
        return {
            "all_ok": False,
            "oracle_solvable": False,
            "frontier_score": 0.0,
            "difficulty": 1.0,
            "checks": checks,
        }

    active = rollout_with_policy(scenario, active_verification_policy)
    passive = rollout_with_policy(scenario, passive_susceptible_policy)
    fake_evidence = scenario.get("poisoning_plan", {}).get("fake_evidence", [])
    fake_count = len(fake_evidence)
    avg_spoof = statistics.mean([float(ev.get("spoofability", 0.0)) for ev in fake_evidence]) if fake_evidence else 0.0
    max_spoof = max([float(ev.get("spoofability", 0.0)) for ev in fake_evidence], default=0.0)
    early_fake = sum(1 for ev in fake_evidence if int(ev.get("time", 0)) <= 1) / max(fake_count, 1)
    reliabilities = scenario.get("source_reliability", {})
    avg_source_reliability = statistics.mean([float(reliabilities.get(ev.get("source"), 0.5)) for ev in fake_evidence]) if fake_evidence else 0.5
    phase_len = len(scenario.get("true_attack", {}).get("phase_schedule", []))
    verification_budget = max(1.0, float(scenario.get("defense_constraints", {}).get("verification_budget", 4)))
    business_budget = max(1e-6, float(scenario.get("defense_constraints", {}).get("business_budget", 5.0)))
    max_high = float(scenario.get("defense_constraints", {}).get("max_high_impact_actions", 1))
    split = scenario.get("metadata", {}).get("cfc_split", "unknown")
    has_switch = 1.0 if scenario.get("strategy_switch") or scenario.get("poisoning_plan", {}).get("strategy_switch") else 0.0
    trust_len = len(scenario.get("trust_building_plan", []))
    target_criticality = _target_criticality(scenario)

    poisoning_subtlety = _clamp(0.55 * avg_spoof + 0.35 * avg_source_reliability + 0.10 * min(1.0, trust_len / 4.0))
    verification_pressure = _clamp(fake_count / verification_budget)
    long_horizon = _clamp((phase_len - 4.0) / 3.0)
    business_pressure = _clamp(0.55 * target_criticality + 0.35 * (1.0 - min(1.0, business_budget / 6.0)) + 0.10 * (1.0 if max_high <= 0 else 0.0))
    deception_pressure = _clamp(0.45 * early_fake + 0.35 * max_spoof + 0.20 * has_switch)
    difficulty = _clamp(
        0.30 * poisoning_subtlety
        + 0.25 * verification_pressure
        + 0.20 * deception_pressure
        + 0.15 * business_pressure
        + 0.10 * long_horizon
    )

    active_safe = bool(active.get("safe_success", False))
    passive_safe = bool(passive.get("safe_success", False))
    active_fail = 1.0 - float(active_safe)
    passive_fail = 1.0 - float(passive_safe)
    verification_value = _clamp(passive_fail - active_fail)
    poisoning_success = 1.0 if float(passive.get("fake_confirmed", 0.0)) > 0 else 0.0
    overresponse_pressure = 1.0 if passive.get("overresponse", False) else 0.0
    oracle_solvable = checks.get("all_ok", False) and active_safe

    frontier_score = 0.0
    if oracle_solvable:
        frontier_score = _clamp(
            0.42 * difficulty
            + 0.23 * verification_value
            + 0.18 * poisoning_success
            + 0.10 * overresponse_pressure
            + 0.07 * (1.0 if split in SPLITS else 0.0)
        )

    return {
        "all_ok": True,
        "checks": checks,
        "oracle_solvable": bool(oracle_solvable),
        "frontier_score": float(frontier_score),
        "difficulty": float(difficulty),
        "poisoning_subtlety": float(poisoning_subtlety),
        "verification_pressure": float(verification_pressure),
        "deception_pressure": float(deception_pressure),
        "business_pressure": float(business_pressure),
        "long_horizon": float(long_horizon),
        "verification_value": float(verification_value),
        "poisoning_success_proxy": float(poisoning_success),
        "overresponse_pressure": float(overresponse_pressure),
        "active_safe_success": float(active_safe),
        "passive_safe_success": float(passive_safe),
        "active_reward": float(active.get("reward", 0.0)),
        "passive_reward": float(passive.get("reward", 0.0)),
        "passive_fake_confirmed": int(passive.get("fake_confirmed", 0)),
        "split": split,
        "true_objective": scenario.get("oracle", {}).get("true_objective", "unknown"),
        "false_objective": scenario.get("poisoning_plan", {}).get("false_objective", "unknown"),
        "fake_count": fake_count,
        "phase_len": phase_len,
    }


def select_frontier(items: list[dict[str, Any]], frontier_size: int) -> list[dict[str, Any]]:
    valid = [item for item in items if item["cfc"].get("oracle_solvable") and item["cfc"].get("frontier_score", 0.0) > 0.0]
    valid.sort(key=lambda item: item["cfc"]["frontier_score"], reverse=True)
    if frontier_size <= 0 or frontier_size >= len(valid):
        return valid

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_split_target = max(1, math.ceil(frontier_size / len(SPLITS)))
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in valid:
        by_split[item["cfc"].get("split", "unknown")].append(item)

    for split in SPLITS:
        for item in by_split.get(split, [])[:per_split_target]:
            fp = item["fingerprint"]
            if fp not in selected_ids and len(selected) < frontier_size:
                selected.append(item)
                selected_ids.add(fp)

    for item in valid:
        fp = item["fingerprint"]
        if fp not in selected_ids and len(selected) < frontier_size:
            selected.append(item)
            selected_ids.add(fp)
    return selected


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)


def _training_rows(frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, item in enumerate(frontier):
        scenario = item["scenario"]
        row = scenario_to_training_row(scenario, split="train")
        cfc = item["cfc"]
        extra = dict(row.get("extra_info", {}))
        extra.update(
            {
                "cfc_rank": rank,
                "cfc_split": cfc.get("split"),
                "frontier_score": cfc.get("frontier_score"),
                "difficulty": cfc.get("difficulty"),
                "oracle_solvable": cfc.get("oracle_solvable"),
            }
        )
        row["extra_info"] = extra
        row["frontier_score"] = cfc.get("frontier_score")
        row["difficulty"] = cfc.get("difficulty")
        row["cfc_split"] = cfc.get("split")
        row["cfc_metrics"] = json.dumps(cfc, ensure_ascii=False)
        rows.append(row)
    return rows


def build_frontier(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    prefix = args.prefix or f"level1_seed{args.seed}_n{args.num_candidates}"
    scenario_dir = os.path.join(args.output_dir, "scenarios")
    os.makedirs(scenario_dir, exist_ok=True)

    items = []
    seen = set()
    attempts = 0
    while len(items) < args.num_candidates and attempts < args.num_candidates * 5:
        attempts += 1
        scenario = generate_scenario(len(items), rng, args.seed)
        fingerprint = _stable_hash({k: v for k, v in scenario.items() if k != "scenario_id"})
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        cfc = compute_cfc_metrics(scenario)
        items.append({"scenario": scenario, "cfc": cfc, "fingerprint": fingerprint})

    frontier = select_frontier(items, args.frontier_size)
    rows = _training_rows(frontier)

    candidates_path = os.path.join(scenario_dir, f"{prefix}_candidates.json")
    frontier_path = os.path.join(scenario_dir, f"{prefix}_frontier.json")
    frontier_parquet = os.path.join(args.output_dir, f"{prefix}_frontier_vda.parquet")
    report_path = os.path.join(args.output_dir, f"{prefix}_cfc_report.json")

    _write_json(candidates_path, items)
    _write_json(frontier_path, frontier)
    os.makedirs(os.path.dirname(frontier_parquet) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(frontier_parquet, index=False)

    split_counts = Counter(item["cfc"].get("split", "unknown") for item in frontier)
    objective_counts = Counter(item["cfc"].get("true_objective", "unknown") for item in frontier)
    false_counts = Counter(item["cfc"].get("false_objective", "unknown") for item in frontier)
    scores = [float(item["cfc"].get("frontier_score", 0.0)) for item in frontier]
    difficulties = [float(item["cfc"].get("difficulty", 0.0)) for item in frontier]
    manifest = {
        "prefix": prefix,
        "seed": args.seed,
        "num_candidates_requested": args.num_candidates,
        "num_candidates_written": len(items),
        "frontier_size_requested": args.frontier_size,
        "frontier_size": len(frontier),
        "oracle_solvable_candidates": sum(1 for item in items if item["cfc"].get("oracle_solvable")),
        "paths": {
            "candidates_json": candidates_path,
            "frontier_json": frontier_path,
            "frontier_parquet": frontier_parquet,
            "report_json": report_path,
        },
        "frontier_split_counts": dict(split_counts),
        "frontier_true_objective_counts": dict(objective_counts),
        "frontier_false_objective_counts": dict(false_counts),
        "frontier_score": {
            "min": min(scores) if scores else 0.0,
            "mean": statistics.mean(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
        },
        "difficulty": {
            "min": min(difficulties) if difficulties else 0.0,
            "mean": statistics.mean(difficulties) if difficulties else 0.0,
            "max": max(difficulties) if difficulties else 0.0,
        },
    }
    _write_json(report_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--frontier-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "data", "level1"))
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    manifest = build_frontier(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
