"""Model-free policies used to validate environment/reward learnability."""

from __future__ import annotations

import copy
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.public_teacher import (
    ActionCandidate,
    PublicStateRobustTeacher,
    admitted_public_candidates,
    public_state_digest,
)
from agentguard_zero.recovery.utility import (
    recovery_core_utility,
    spearman_rank_correlation,
)
from agentguard_zero.runtime_policy import HIGH_IMPACT_ACTIONS
from agentguard_zero.schemas.action_schema import OBJECTIVE_KEYS
from agentguard_zero.schemas.action_schema_v4 import DEFAULT_ACTION_PACKET_V4
from agentguard_zero.tools.business_impact import estimate_business_impact
from agentguard_zero.world.hidden_world import MITIGATION_STRENGTH, PHASE_COMPATIBILITY


def _observe_packet() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_ACTION_PACKET_V4)


def _stable_index(namespace: str, scenario_id: str, time: int, count: int) -> int:
    raw = f"{namespace}:{scenario_id}:{time}:{count}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % max(1, count)


def _random_legal_packet(env: Any) -> dict[str, Any]:
    candidates = admitted_public_candidates([env])
    observation = env.observe()
    observed_entities = {
        str(event.get("entity_id", "")).strip()
        for event in observation.get("observed_events", []) or []
        if isinstance(event, dict) and str(event.get("entity_id", "")).strip()
    }
    context_relevant_mitigations = [
        candidate
        for candidate in candidates
        if candidate.category == "mitigation"
        and str((candidate.packet.get("response", {}) or {}).get("target", ""))
        in observed_entities
    ]
    if context_relevant_mitigations:
        candidates = [
            candidate for candidate in candidates if candidate.category != "mitigation"
        ] + context_relevant_mitigations
    # Sample action classes uniformly before sampling an action. Sampling the
    # flat enumeration would overweight classes that merely have more public
    # parameter combinations (especially mitigations). Mitigations are limited
    # to publicly observed entities; this is still random and hidden-blind, but
    # it does not treat an unrelated asset as a context-relevant legal target.
    by_category: dict[str, list[ActionCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_category[candidate.category].append(candidate)
    categories = sorted(by_category)
    category_index = _stable_index(
        "random_legal_category",
        str(env.scenario.get("scenario_id", "")),
        int(env.t),
        len(categories),
    )
    category = categories[category_index]
    choices = sorted(by_category[category], key=lambda item: item.candidate_id)
    action_index = _stable_index(
        "random_legal_action",
        str(env.scenario.get("scenario_id", "")),
        int(env.t),
        len(choices),
    )
    return copy.deepcopy(choices[action_index].packet)


def _overreact_packet(env: Any) -> dict[str, Any]:
    observation = env.observe()
    assets = (observation.get("defense_context", {}) or {}).get("public_assets", [])
    target = max(
        (
            item
            for item in assets
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ),
        key=lambda item: (float(item.get("criticality", 0.0)), str(item.get("id", ""))),
        default={"id": "none"},
    )
    packet = _observe_packet()
    packet["belief"] = {
        key: (1.0 if key == "exfiltration" else 0.0) for key in OBJECTIVE_KEYS
    }
    packet["uncertainty"] = 0.0
    packet["safety_check"] = {
        "business_risk": 1.0,
        "overresponse_risk": 1.0,
        "justification": "immediate containment",
    }
    packet["response"] = {
        "tier": "L3",
        "action": "Isolate",
        "target": str(target.get("id", "none")),
    }
    return packet


def _oracle_packet(env: Any) -> dict[str, Any]:
    packet = _observe_packet()
    objective = str(env.hidden_world.true_objective)
    packet["belief"] = {
        key: (1.0 if key == objective else 0.0) for key in OBJECTIVE_KEYS
    }
    packet["uncertainty"] = 0.0
    target = str(env.hidden_world.target)
    pressure = float(env.hidden_world.attack_pressure)
    phase = str(env.hidden_world.current_phase)
    constraints = env.scenario.get("defense_constraints", {}) or {}
    remaining_business = float(constraints.get("business_budget", 5.0)) - float(
        env.business_cost
    )
    remaining_high = int(constraints.get("max_high_impact_actions", 1)) - int(
        env.high_impact_count
    )
    criticality = next(
        (
            float(item.get("criticality", 0.5))
            for item in (env.scenario.get("network_context", {}) or {}).get(
                "assets", []
            )
            if str(item.get("id", "")) == target
        ),
        0.5,
    )
    choices: list[tuple[float, bool, float, float, str]] = []
    for action, strength in MITIGATION_STRENGTH.items():
        if action == "Observe" or float(strength) <= 0.0:
            continue
        cost = float(
            estimate_business_impact(
                {"action": action, "target": target},
                criticality,
            ).get("estimated_cost", 0.0)
        )
        if cost > remaining_business + 1.0e-12:
            continue
        if action in HIGH_IMPACT_ACTIONS and remaining_high <= 0:
            continue
        effective = float(strength) * float(
            PHASE_COMPATIBILITY.get(action, {}).get(phase, 1.0)
        )
        reduction = min(pressure, effective)
        choices.append(
            (
                -(reduction / max(cost, 1.0e-12)),
                effective + 1.0e-12 < pressure,
                cost,
                -effective,
                action,
            )
        )
    if not choices:
        return packet
    _, _, _, _, action = min(choices)
    tier = "L3" if action in {"Isolate", "Remove"} else "L2"
    packet["response"] = {"tier": tier, "action": action, "target": target}
    packet["safety_check"] = {
        "business_risk": 0.0,
        "overresponse_risk": 0.0,
        "justification": "oracle upper bound",
    }
    return packet


def run_fixed_policy_episode(
    scenario: Mapping[str, Any],
    policy: str,
) -> tuple[Any, dict[str, Any]]:
    if policy not in {"no_op", "random_legal", "overreact", "oracle"}:
        raise ValueError(f"unsupported independent fixed policy: {policy}")
    env = instantiate_scenario(
        copy.deepcopy(dict(scenario)),
        oracle_mode=policy == "oracle",
    )
    while not (env.t >= env.max_steps or env.attack_mitigated or env.attack_success):
        if policy == "no_op":
            packet = _observe_packet()
        elif policy == "random_legal":
            packet = _random_legal_packet(env)
        elif policy == "overreact":
            packet = _overreact_packet(env)
        else:
            packet = _oracle_packet(env)
        env.step(packet)
    return env, score_trajectory_v2(env)


def run_public_teacher_group(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    teacher: PublicStateRobustTeacher | None = None,
) -> list[tuple[Any, dict[str, Any]]]:
    episodes, _ = _run_public_teacher_group_with_audit(scenarios, teacher=teacher)
    return episodes


def _run_public_teacher_group_with_audit(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    teacher: PublicStateRobustTeacher | None = None,
) -> tuple[list[tuple[Any, dict[str, Any]]], list[dict[str, Any]]]:
    if len(scenarios) < 2:
        raise ValueError("public teacher group requires multiple hidden worlds")
    policy = teacher or PublicStateRobustTeacher()
    live = [instantiate_scenario(copy.deepcopy(dict(item))) for item in scenarios]
    completed: list[Any] = []
    decision_audit: list[dict[str, Any]] = []
    first_decision = True
    while live:
        groups: dict[str, list[Any]] = defaultdict(list)
        for env in live:
            groups[public_state_digest(env.observe())].append(env)
        next_live: list[Any] = []
        for group in groups.values():
            decision = policy.decide(
                group,
                horizon=3,
                enforce_min_worlds=first_decision,
            )
            decision_audit.append(decision.to_audit_dict())
            for env in group:
                env.step(copy.deepcopy(decision.selected_packet))
                if env.t >= env.max_steps or env.attack_mitigated or env.attack_success:
                    completed.append(env)
                else:
                    next_live.append(env)
        live = next_live
        first_decision = False
    return (
        [(env, score_trajectory_v2(env)) for env in completed],
        decision_audit,
    )


def _fixed_policy_worker(
    payload: tuple[Mapping[str, Any], str],
) -> tuple[Any, dict[str, Any]]:
    scenario, policy = payload
    return run_fixed_policy_episode(scenario, policy)


def _teacher_group_worker(
    payload: tuple[Sequence[Mapping[str, Any]], PublicStateRobustTeacher],
) -> tuple[list[tuple[Any, dict[str, Any]]], list[dict[str, Any]]]:
    scenarios, teacher = payload
    return _run_public_teacher_group_with_audit(scenarios, teacher=teacher)


def _summarize_teacher_decisions(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    categories = Counter(str(item.get("selected_category", "")) for item in decisions)
    correlations: list[float] = []
    for item in decisions:
        q_values = item.get("q_audit", {})
        core_values = item.get("core_q_audit", {})
        if not isinstance(q_values, dict) or not isinstance(core_values, dict):
            continue
        shared = sorted(set(q_values).intersection(core_values))
        correlation = spearman_rank_correlation(
            [float(q_values[key]) for key in shared],
            [float(core_values[key]) for key in shared],
        )
        if correlation is not None:
            correlations.append(correlation)
    return {
        "decision_count": len(decisions),
        "selected_action_category_counts": dict(categories),
        "mean_public_candidate_count": (
            mean(float(item.get("public_candidate_count", 0.0)) for item in decisions)
            if decisions
            else 0.0
        ),
        "mean_admitted_candidate_count": (
            mean(float(item.get("admitted_candidate_count", 0.0)) for item in decisions)
            if decisions
            else 0.0
        ),
        "teacher_core_rank_correlation_mean": (
            mean(correlations) if correlations else None
        ),
        "teacher_core_rank_correlation_state_count": len(correlations),
    }


def recovery_safe_utility(env: Any, score: Mapping[str, Any]) -> float:
    """Backward-compatible name for the shared recovery core utility."""

    return recovery_core_utility(env, score)


def summarize_fixed_policy(
    episodes: Iterable[tuple[Any, Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = list(episodes)
    if not rows:
        raise ValueError("fixed policy has no episodes")
    return {
        "scenario_count": len(rows),
        "safe_utility": mean(recovery_safe_utility(env, score) for env, score in rows),
        "attack_mitigation": mean(
            float(bool(score.get("attack_mitigated", False))) for _, score in rows
        ),
        "safe_success": mean(
            float(bool(score.get("safe_success", False))) for _, score in rows
        ),
        "overresponse": mean(
            float(bool(score.get("overresponse", False))) for _, score in rows
        ),
        "business_cost": mean(
            float(score.get("business_cost", 0.0)) for _, score in rows
        ),
    }


def run_stage0_suite(
    scenario_groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    teacher: PublicStateRobustTeacher | None = None,
    workers: int = 1,
) -> dict[str, dict[str, Any]]:
    scenarios = [dict(item) for group in scenario_groups for item in group]
    results: dict[str, dict[str, Any]] = {}
    policy = teacher or PublicStateRobustTeacher()
    if workers <= 1:
        for name in ("no_op", "random_legal", "overreact", "oracle"):
            results[name] = summarize_fixed_policy(
                run_fixed_policy_episode(scenario, name) for scenario in scenarios
            )
        teacher_episodes: list[tuple[Any, Mapping[str, Any]]] = []
        teacher_decisions: list[dict[str, Any]] = []
        for group in scenario_groups:
            episodes, decisions = _run_public_teacher_group_with_audit(
                group,
                teacher=policy,
            )
            teacher_episodes.extend(episodes)
            teacher_decisions.extend(decisions)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            for name in ("no_op", "random_legal", "overreact", "oracle"):
                results[name] = summarize_fixed_policy(
                    executor.map(
                        _fixed_policy_worker,
                        ((scenario, name) for scenario in scenarios),
                    )
                )
            teacher_episodes = []
            teacher_decisions = []
            for rows, decisions in executor.map(
                _teacher_group_worker,
                ((group, policy) for group in scenario_groups),
            ):
                teacher_episodes.extend(rows)
                teacher_decisions.extend(decisions)
    results["public_state_teacher"] = summarize_fixed_policy(teacher_episodes)
    results["public_state_teacher"].update(
        _summarize_teacher_decisions(teacher_decisions)
    )
    return results
