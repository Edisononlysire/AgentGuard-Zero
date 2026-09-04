#!/usr/bin/env python3
"""Evaluate a candidate-ranker VDA on deterministic T1-T4 canonical suites."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.metrics import action_flags, summarize_candidate_traces
from agentguard_zero.candidate.semantic import semantic_digest
from agentguard_zero.candidate.generator import CandidateGenerator
from agentguard_zero.candidate.policy import (
    ACTIVE_PROBE_TOOLS,
    CandidateDecision,
    CandidateRankerPolicy,
)
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.candidate.supervision import (
    policy_consistent_supervision,
    policy_regret,
)
from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.progressive_ablation import (
    PROGRESSIVE_ARM_ORDER,
    progressive_arm,
    progressive_generator,
)
from agentguard_zero.recovery.canonical_scenarios import canonical_recovery_suite
from agentguard_zero.recovery.public_teacher import (
    PublicStateRobustTeacher,
    public_state_digest,
)
from agentguard_zero.recovery.utility import (
    raw_safe_utility,
    recovery_core_utility,
)
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    scenario_fingerprint,
    sha256_file,
    sha256_tree,
    utc_now,
)
from scripts.candidate_ecrg_lib import (
    build_k6_decision,
    candidate_set_digest,
    candidate_set_seed,
    exact_k6_generator,
    select_ecrg,
)
from scripts.ecrg_calibration_lib import admission_allowed


K6_POLICY_MODES = {
    "train_greedy_k6",
    "random_k6",
    "best_of_k6",
    "soft_selector_k6",
    "ecrg_k6",
    "full_k6",
}
RANKER_POLICY_MODES = {
    "candidate_ranker",
    "train_greedy_k6",
    "best_of_k6",
    "soft_selector_k6",
    "full_k6",
}
ECRG_POLICY_MODES = {"ecrg_k6", "full_k6"}


def _scenario_from_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("scenario", "scenario_json"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    if row.get("protocol_version") == "tmcd-v2":
        return row
    raise ValueError("row does not contain a TMCD scenario")


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict(orient="records")
    elif path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload.get("groups"), list):
            rows = [item for group in payload["groups"] for item in group]
        else:
            rows = payload.get("scenarios", [])
    return [_scenario_from_row(dict(row)) for row in rows]


def _paired_trajectory_groups(
    scenarios: list[dict[str, Any]],
    *,
    expected_group_count: int,
    suite_name: str,
) -> list[list[dict[str, Any]]]:
    fair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        group_id = str(
            (scenario.get("metadata") or {}).get("fair_public_group_id", "")
        )
        if not group_id:
            raise RuntimeError("fair trajectory scenario has no public-group identity")
        fair_groups[group_id].append(scenario)
    if len(fair_groups) != expected_group_count or any(
        len(group) != 2 for group in fair_groups.values()
    ):
        raise RuntimeError(
            f"{suite_name} requires {expected_group_count} public groups of two worlds"
        )
    return [fair_groups[key] for key in sorted(fair_groups)]


def _fair_trajectory_groups(
    scenarios: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    return _paired_trajectory_groups(
        scenarios,
        expected_group_count=100,
        suite_name="fair trajectory suite",
    )


def _memory_counterfactual(
    observation: dict[str, Any], *, mode: str
) -> dict[str, Any]:
    if mode == "none":
        return observation
    value = copy.deepcopy(observation)
    defender = value.get("defender_state") or {}
    memory = defender.get("memory") or {}
    buckets = (
        "retrieved_confirmed",
        "retrieved_quarantined",
        "rejected_warnings",
        "retrieved_profiles",
    )
    if mode == "drop":
        for bucket in buckets:
            memory[bucket] = []
        memory["retrieved_memory_ids"] = []
        defender["business_impact_memory"] = {}
        return value
    if mode != "shuffle":
        raise ValueError(f"unsupported memory counterfactual: {mode}")

    rows = [
        row
        for bucket in buckets
        for row in memory.get(bucket, []) or []
        if isinstance(row, dict)
    ]
    payload_keys = (
        "claim",
        "source_ids",
        "source_id",
        "confidence",
        "transition_history",
        "historical_accuracy",
        "recent_accuracy",
        "independent_confirmations",
        "recent_contradictions",
        "trust_trend",
        "suspected_deception_phase",
    )
    if len(rows) >= 2:
        payloads = [
            {key: copy.deepcopy(row[key]) for key in payload_keys if key in row}
            for row in rows
        ]
        payloads = payloads[1:] + payloads[:1]
        for row, payload in zip(rows, payloads, strict=True):
            for key in payload_keys:
                row.pop(key, None)
            row.update(payload)
    elif rows:
        row = rows[0]
        history = list(row.get("transition_history", []) or [])
        if history:
            row["transition_history"] = list(reversed(history))
        if "historical_accuracy" in row or "recent_accuracy" in row:
            historical = float(row.get("historical_accuracy", 0.5))
            recent = float(row.get("recent_accuracy", historical))
            row["historical_accuracy"] = recent
            row["recent_accuracy"] = historical
            row["trust_trend"] = {
                "declining": "improving",
                "improving": "declining",
            }.get(str(row.get("trust_trend", "stable")), "stable")
        claim = row.get("claim") or {}
        if isinstance(claim, dict) and claim:
            objects = ("exfiltration", "sabotage", "persistence", "credential_theft")
            current = str(claim.get("object", ""))
            claim["object"] = objects[(objects.index(current) + 1) % len(objects)] if current in objects else objects[0]
    business = defender.get("business_impact_memory") or {}
    if business:
        business["cumulative_business_cost"], business["remaining_business_budget"] = (
            business.get("remaining_business_budget", 0.0),
            business.get("cumulative_business_cost", 0.0),
        )
    return value


def _is_probe_feedback(value: Any) -> bool:
    """Return whether a public row exposes the result of an active probe."""

    if not isinstance(value, dict):
        return False
    content = value.get("content") or value
    if not isinstance(content, dict):
        return False
    tool = str(content.get("tool", ""))
    claim = content.get("claim_semantics") or {}
    return bool(
        tool in ACTIVE_PROBE_TOOLS
        or content.get("active_probe", False)
        or content.get("probe_generated", False)
        or str(content.get("probe_id", ""))
        or str(content.get("type", "")) == "decoy_probe_result"
        or (
            isinstance(claim, dict)
            and str(claim.get("scope", "")) == "active_probe"
        )
    )


def _probe_feedback_visible(observation: dict[str, Any]) -> bool:
    defender = observation.get("defender_state") or {}
    probe_state = defender.get("probe_state") or []
    return bool(
        _is_probe_feedback(observation.get("last_tool_result") or {})
        or any(_is_probe_feedback(row) for row in observation.get("observed_events", []) or [])
        or any(_is_probe_feedback(row) for row in observation.get("available_evidence", []) or [])
        or any(
            isinstance(row, dict)
            and (
                str(row.get("result", ""))
                or str(row.get("status", "")) == "resolved"
            )
            for row in probe_state
        )
    )


def _probe_counterfactual(
    observation: dict[str, Any], *, mode: str
) -> dict[str, Any]:
    """Hide active-probe feedback while preserving that a probe was spent.

    This is an information intervention, not a simulator ablation: the hidden
    world, probe cost, remaining budget, and attack evolution are unchanged.
    Probe-result evidence and its direct public-state descendants are removed
    from the model view so downstream outcome changes measure information use.
    """

    if mode == "none":
        return observation
    if mode != "drop":
        raise ValueError(f"unsupported probe counterfactual: {mode}")

    value = copy.deepcopy(observation)
    evidence = [
        row
        for row in value.get("available_evidence", []) or []
        if isinstance(row, dict)
    ]
    removed_ids = {
        str(row.get("evidence_id", ""))
        for row in evidence
        if _is_probe_feedback(row) and str(row.get("evidence_id", ""))
    }
    changed = True
    while changed:
        changed = False
        for row in evidence:
            evidence_id = str(row.get("evidence_id", ""))
            parents = set(map(str, row.get("parent_evidence_ids", []) or []))
            if evidence_id and evidence_id not in removed_ids and parents & removed_ids:
                removed_ids.add(evidence_id)
                changed = True

    removed_events = {
        str(row.get("event_id", ""))
        for row in evidence
        if str(row.get("evidence_id", "")) in removed_ids
        and str(row.get("event_id", ""))
    }
    removed_sources = {
        str(row.get("source_id", ""))
        for row in evidence
        if str(row.get("evidence_id", "")) in removed_ids
        and str(row.get("source_id", ""))
    }
    value["available_evidence"] = [
        row
        for row in evidence
        if str(row.get("evidence_id", "")) not in removed_ids
    ]
    retained_sources = {
        str(row.get("source_id", ""))
        for row in value["available_evidence"]
        if str(row.get("source_id", ""))
    }

    filtered_events = []
    for row in value.get("observed_events", []) or []:
        if not isinstance(row, dict):
            continue
        if _is_probe_feedback(row):
            event_id = str(row.get("event_id", ""))
            source_id = str(row.get("source_id") or row.get("source") or "")
            if event_id:
                removed_events.add(event_id)
            if source_id:
                removed_sources.add(source_id)
            continue
        filtered_events.append(row)
    value["observed_events"] = filtered_events

    if _is_probe_feedback(value.get("last_tool_result") or {}):
        value["last_tool_result"] = {"tool": "None"}

    defender = value.get("defender_state") or {}
    sanitized_probe_state = []
    for row in defender.get("probe_state", []) or []:
        if not isinstance(row, dict):
            continue
        sanitized = {
            key: copy.deepcopy(row[key])
            for key in ("probe_id", "type", "zone", "effective_at")
            if key in row
        }
        sanitized["status"] = "pending"
        sanitized_probe_state.append(sanitized)
    defender["probe_state"] = sanitized_probe_state

    trust = defender.get("trust") or {}
    claims = trust.get("current_claim_trust") or {}
    trust["current_claim_trust"] = {
        event_id: row
        for event_id, row in claims.items()
        if str(event_id) not in removed_events
        and not (
            isinstance(row, dict)
            and removed_ids
            & {
                str(reference)
                for key in (
                    "evidence_refs",
                    "support_evidence_refs",
                    "contradiction_evidence_refs",
                )
                for reference in row.get(key, []) or []
            }
        )
    }
    source_reputation = trust.get("source_reputation") or {}
    trust["source_reputation"] = {
        source_id: row
        for source_id, row in source_reputation.items()
        if str(source_id) not in (removed_sources - retained_sources)
    }

    memory = defender.get("memory") or {}
    retained_memory_ids: list[str] = []
    for bucket in (
        "retrieved_confirmed",
        "retrieved_quarantined",
        "rejected_warnings",
        "retrieved_profiles",
    ):
        rows = [
            row
            for row in memory.get(bucket, []) or []
            if isinstance(row, dict)
            and not (
                removed_ids
                & {str(item) for item in row.get("evidence_refs", []) or []}
            )
        ]
        memory[bucket] = rows
        retained_memory_ids.extend(
            str(row.get("memory_id", ""))
            for row in rows
            if str(row.get("memory_id", ""))
        )
    memory["retrieved_memory_ids"] = list(dict.fromkeys(retained_memory_ids))

    impact = defender.get("business_impact_memory") or {}
    estimates = [
        row
        for row in impact.get("impact_estimates", []) or []
        if not (
            isinstance(row, dict)
            and str(row.get("tool", "")) in ACTIVE_PROBE_TOOLS
        )
    ]
    impact["impact_estimates"] = estimates
    impact["impact_estimate_count"] = len(estimates)
    impact["cumulative_estimated_cost"] = sum(
        float(row.get("estimated_cost", 0.0))
        for row in estimates
        if isinstance(row, dict)
    )
    return value


def _task_id(env: Any) -> str:
    metadata = env.scenario.get("metadata", {}) or {}
    return str(metadata.get("task_id", "unknown"))


def _terminal(env: Any) -> bool:
    return bool(env.t >= env.max_steps or env.attack_mitigated or env.attack_success)


def _manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "candidate_ranker_checkpoint":
        raise RuntimeError("not a candidate ranker checkpoint manifest")
    adapter = Path(str(payload.get("adapter_path", "")))
    head = Path(str(payload.get("heads_path") or payload.get("score_head_path", "")))
    if sha256_tree(adapter) != payload.get("adapter_sha256"):
        raise RuntimeError("candidate adapter hash mismatch")
    expected_head_hash = payload.get("heads_sha256") or payload.get("score_head_sha256")
    if sha256_file(head) != expected_head_hash:
        raise RuntimeError("candidate score head hash mismatch")
    return payload


def static_anchor_decision(
    *,
    policy_mode: str,
    observation: dict[str, Any],
    worlds: list[Any],
) -> CandidateDecision:
    """Select a public Random action or privileged Oracle normalization action."""

    from scripts.eval_tmcd_systems import oracle_action, random_action

    if policy_mode == "random_policy":
        packet = random_action(observation, None, worlds[0].t)
        reason = "public_observation_random_policy_anchor"
        privileged = False
    elif policy_mode == "oracle_defender":
        if len(worlds) != 1:
            raise RuntimeError("Oracle normalization requires one sealed hidden world")
        packet = oracle_action(observation, worlds[0].scenario, worlds[0].t)
        reason = "privileged_hidden_world_oracle_nsu_anchor_only"
        privileged = True
    else:
        raise ValueError(f"not a static anchor policy: {policy_mode}")
    flags = action_flags(packet)
    semantic_id = CandidateOption.packet_digest(packet)
    return CandidateDecision(
        candidate_id=semantic_id,
        semantic_id=semantic_id,
        packet=copy.deepcopy(packet),
        valid=True,
        invalid_noop=False,
        reason=reason,
        action_flags=flags,
        candidate_count=1,
        score=1.0 if privileged else None,
        scores={},
        gated_family="privileged_oracle" if privileged else "random_policy",
    )


def _load_candidate_ecrg_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_false = (
        "hidden_state_access",
        "oracle_access",
        "teacher_q_access",
        "tmcd_test_used",
        "parameter_training",
        "vda_parameter_update",
        "dca_parameter_update",
    )
    if not (
        payload.get("kind") == "candidate_ecrg_config"
        and payload.get("status") == "frozen"
        and payload.get("candidate_count") == 6
        and payload.get("candidate_protocol")
        == "one_public_candidate_per_action_family"
        and all(payload.get(key) is False for key in required_false)
        and isinstance(payload.get("parameters"), dict)
    ):
        raise RuntimeError("invalid or non-frozen candidate ECRG config")
    return payload


def _decision_from_k6_row(
    row: dict[str, Any],
    *,
    reason: str,
    scores: dict[str, float] | None = None,
) -> CandidateDecision:
    packet = copy.deepcopy(row["packet"])
    flags = action_flags(packet)
    return CandidateDecision(
        candidate_id=row.get("candidate_key"),
        semantic_id=str(row.get("semantic_id", CandidateOption.packet_digest(packet))),
        packet=packet,
        valid=True,
        invalid_noop=False,
        reason=reason,
        action_flags=flags,
        candidate_count=6,
        score=(
            (scores or {}).get(str(row.get("candidate_key")))
            if row.get("candidate_key") is not None
            else None
        ),
        scores=dict(scores or {}),
        gated_family=str(row.get("action_family", "")),
    )


def _teacher_regret_metrics(
    *,
    teacher_decision: Any,
    selected_semantic_id: str | None,
    options: list[CandidateOption],
    core_tolerance: float,
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Compute optional regret only when Teacher and evaluator IDs align.

    Teacher regret is a diagnostic.  A candidate-generator mismatch must never
    abort an otherwise valid complete trajectory or be silently recomputed on
    a smaller candidate set.
    """

    if teacher_decision is None:
        return None, None, {"available": False, "reason": "teacher_disabled"}
    selected_id = str(selected_semantic_id or "")
    q_audit = dict(teacher_decision.q_audit or {})
    core_q_audit = dict(teacher_decision.core_q_audit or {})
    option_by_id = {str(option.semantic_id): option for option in options}
    q_ids = set(map(str, q_audit))
    missing_options = sorted(q_ids - set(option_by_id))
    missing_core = sorted(q_ids - set(map(str, core_q_audit)))
    required_missing = []
    if selected_id not in q_ids:
        required_missing.append("model_selected_candidate")
    if str(teacher_decision.selected_candidate_id) not in q_ids:
        required_missing.append("teacher_selected_candidate")
    observe_ids = sorted(
        candidate_id
        for candidate_id in q_ids & set(option_by_id)
        if option_by_id[candidate_id].action_flags.observe_only
    )
    if not observe_ids:
        required_missing.append("observe_candidate")
    if missing_options or missing_core or required_missing:
        return None, None, {
            "available": False,
            "reason": "candidate_id_alignment_failed",
            "teacher_candidate_count": len(q_ids),
            "evaluator_candidate_count": len(option_by_id),
            "missing_option_count": len(missing_options),
            "missing_option_ids": missing_options[:8],
            "missing_core_q_count": len(missing_core),
            "required_missing": required_missing,
        }

    candidate_ids = sorted(q_ids)
    observe_id = max(
        observe_ids,
        key=lambda candidate_id: (float(q_audit[candidate_id]), candidate_id),
    )
    q_values = [float(q_audit[candidate_id]) for candidate_id in candidate_ids]
    core_q_values = [
        float(core_q_audit[candidate_id]) for candidate_id in candidate_ids
    ]
    supervision = policy_consistent_supervision(
        q_values,
        core_q_values,
        target_index=candidate_ids.index(str(teacher_decision.selected_candidate_id)),
        observe_index=candidate_ids.index(observe_id),
        core_tolerance=core_tolerance,
        temperature=0.1,
    )
    selected_index = candidate_ids.index(selected_id)
    selected_q = float(q_audit[selected_id])
    return (
        policy_regret(supervision.policy_scores, selected_index),
        max(q_values) - selected_q,
        {
            "available": True,
            "reason": "aligned",
            "teacher_candidate_count": len(q_ids),
            "evaluator_candidate_count": len(option_by_id),
        },
    )


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    information_intervention = bool(
        args.probe_counterfactual != "none"
        or args.memory_counterfactual != "none"
    )
    if args.intervention_consistent_candidates and args.policy_mode != "candidate_ranker":
        raise ValueError(
            "intervention-consistent candidates require candidate_ranker mode"
        )
    if args.policy_mode in RANKER_POLICY_MODES:
        if args.ranker_manifest is None:
            raise ValueError(f"{args.policy_mode} requires --ranker-manifest")
        manifest = _manifest(args.ranker_manifest)
    else:
        if args.ranker_manifest is not None:
            raise ValueError(f"{args.policy_mode} forbids --ranker-manifest")
        if args.scenario_source is None:
            raise ValueError("normalization anchors require a sealed --scenario-source")
        manifest = None
    progressive = (
        progressive_arm(args.progressive_arm)
        if args.progressive_arm is not None
        else None
    )
    if args.protocol_role == "progressive_ablation":
        if progressive is None:
            raise ValueError(
                "progressive ablation requires --progressive-arm"
            )
        if args.policy_mode not in {
            "candidate_ranker",
            "full_k6",
            "random_policy",
            "oracle_defender",
        }:
            raise ValueError("unsupported progressive ablation policy mode")
        if progressive.ecrg_enabled != (args.policy_mode == "full_k6"):
            raise ValueError("ECRG arm must be the only progressive full_k6 mode")
        if manifest is not None:
            declared_arm = manifest.get("progressive_arm")
            if progressive.name == "zero_shot":
                if not (
                    manifest.get("status") == "initialized_untrained"
                    and manifest.get("optimizer_steps") == 0
                    and manifest.get("zero_shot_progressive_arm") is True
                ):
                    raise RuntimeError("Zero-shot arm is not the common initialization")
            elif progressive.name == "ecrg":
                if declared_arm != "coevolution":
                    raise RuntimeError("ECRG must reuse the Coevolution checkpoint")
            elif declared_arm != progressive.name:
                raise RuntimeError("progressive checkpoint arm mismatch")
    elif progressive is not None:
        raise ValueError(
            "--progressive-arm is restricted to progressive_ablation evaluations"
        )
    if args.policy_mode in ECRG_POLICY_MODES:
        if args.ecrg_config is None:
            raise ValueError(f"{args.policy_mode} requires --ecrg-config")
        ecrg_config = _load_candidate_ecrg_config(args.ecrg_config)
        ecrg_config_sha256 = sha256_file(args.ecrg_config)
    else:
        if args.ecrg_config is not None:
            raise ValueError(f"{args.policy_mode} forbids --ecrg-config")
        ecrg_config = None
        ecrg_config_sha256 = None
    cache_root = Path(
        os.environ.get(
            "AGZ_TRITON_CACHE_ROOT",
            f"/tmp/agentguard_zero_triton_{os.environ.get('USER', 'user')}",
        )
    )
    cache = cache_root / f"candidate_policy_eval_{os.getpid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    scenario_source_sha256 = None
    if args.scenario_source is not None:
        scenarios = _load_scenarios(args.scenario_source)
        if args.task_filter is not None:
            scenarios = [
                scenario
                for scenario in scenarios
                if str((scenario.get("metadata") or {}).get("task_id", ""))
                == args.task_filter
            ]
        if len(scenarios) != args.scenario_count:
            raise RuntimeError(
                f"scenario source has {len(scenarios)} rows, expected {args.scenario_count}"
            )
        if args.protocol_role in {
            "fair_trajectory",
            "long_memory_fair",
            "full_trajectory_causal",
            "mechanism_dev",
            "epoch_selection_dev",
        }:
            groups = _paired_trajectory_groups(
                scenarios,
                expected_group_count=(
                    11
                    if args.protocol_role == "mechanism_dev"
                    else 200
                    if args.protocol_role == "epoch_selection_dev"
                    else 100
                ),
                suite_name=(
                    "mechanism Dev suite"
                    if args.protocol_role == "mechanism_dev"
                    else "T1/T2 Epoch-selection Dev suite"
                    if args.protocol_role == "epoch_selection_dev"
                    else "fair trajectory suite"
                ),
            )
        else:
            groups = [[scenario] for scenario in scenarios]
        scenario_source_sha256 = sha256_file(args.scenario_source)
        fingerprints = [scenario_fingerprint(scenario) for scenario in scenarios]
        if len(fingerprints) != len(set(fingerprints)):
            raise RuntimeError("scenario source contains duplicate semantic scenarios")
        task_counts = {
            task: sum(
                str((scenario.get("metadata") or {}).get("task_id", "")) == task
                for scenario in scenarios
            )
            for task in ("T1", "T2", "T3", "T4")
        }
        if args.protocol_role == "xplay" and (
            args.scenario_count != 800
            or task_counts != {task: 200 for task in ("T1", "T2", "T3", "T4")}
        ):
            raise RuntimeError(f"formal xplay must be balanced 800: {task_counts}")
        if args.protocol_role == "tmcd_test" and (
            args.scenario_count != 2400
            or task_counts != {task: 600 for task in ("T1", "T2", "T3", "T4")}
        ):
            raise RuntimeError(f"formal TMCD-Test must be balanced 2400: {task_counts}")
        if args.protocol_role == "progressive_ablation" and (
            args.scenario_count != 2400
            or task_counts != {task: 600 for task in ("T1", "T2", "T3", "T4")}
        ):
            raise RuntimeError(
                f"progressive ablation must use balanced TMCD 2400: {task_counts}"
            )
        if args.protocol_role == "probe_budget" and (
            args.scenario_count != 600
            or args.task_filter != "T1"
            or task_counts != {"T1": 600, "T2": 0, "T3": 0, "T4": 0}
            or args.active_probe_budget not in {0, 1, 2, 3}
        ):
            raise RuntimeError(
                f"formal Probe Budget must be the sealed T1 600: {task_counts}"
            )
        if args.protocol_role == "ecrg_subset" and (
            args.scenario_count != 800
            or task_counts != {task: 200 for task in ("T1", "T2", "T3", "T4")}
            or args.policy_mode
            not in K6_POLICY_MODES | {"random_policy", "oracle_defender"}
        ):
            raise RuntimeError(
                f"formal ECRG subset must be balanced 800 with K=6: {task_counts}"
            )
        if args.protocol_role == "fair_trajectory" and (
            args.scenario_count != 200
            or task_counts != {task: 50 for task in ("T1", "T2", "T3", "T4")}
            or args.policy_mode != "candidate_ranker"
        ):
            raise RuntimeError(
                "fair trajectory requires 200 paired hidden worlds, balanced "
                f"50 per task, with a learned candidate ranker: {task_counts}"
            )
        if args.protocol_role == "long_memory_fair" and (
            args.scenario_count != 200
            or task_counts != {"T1": 0, "T2": 0, "T3": 100, "T4": 100}
            or args.policy_mode != "candidate_ranker"
        ):
            raise RuntimeError(
                "long-memory fair trajectory requires 200 paired hidden worlds, "
                f"balanced T3/T4, with a learned candidate ranker: {task_counts}"
            )
        if args.protocol_role == "full_trajectory_causal" and (
            args.scenario_count != 200
            or task_counts != {task: 50 for task in ("T1", "T2", "T3", "T4")}
            or args.policy_mode != "candidate_ranker"
            or not args.intervention_consistent_candidates
            or not args.skip_teacher_audit
        ):
            raise RuntimeError(
                "full-trajectory causal evaluation requires 200 paired hidden "
                "worlds, 50 per task, learned candidate_ranker selection, "
                "intervention-consistent candidates, and disabled Teacher audit: "
                f"{task_counts}"
            )
        if args.protocol_role == "mechanism_dev" and (
            args.scenario_count != 22
            or task_counts != {"T1": 0, "T2": 0, "T3": 12, "T4": 10}
            or args.policy_mode != "candidate_ranker"
            or not args.post_mitigation_audit_step
            or not args.skip_teacher_audit
            or any(
                (scenario.get("metadata") or {}).get("formal_test") is not False
                for scenario in scenarios
            )
        ):
            raise RuntimeError(
                "mechanism Dev requires 22 paired non-formal scenarios "
                "(T3=12, T4=10), learned candidate_ranker selection, a "
                "post-mitigation audit step, and disabled Teacher audit: "
                f"{task_counts}"
            )
        if args.protocol_role == "epoch_selection_dev" and (
            args.scenario_count != 400
            or task_counts != {"T1": 200, "T2": 200, "T3": 0, "T4": 0}
            or args.policy_mode != "candidate_ranker"
            or not args.post_mitigation_audit_step
            or not args.skip_teacher_audit
            or any(
                (
                    (scenario.get("metadata") or {}).get("formal_test") is not False
                    or (scenario.get("metadata") or {}).get("formal_frozen_test")
                    is not False
                )
                for scenario in scenarios
            )
        ):
            raise RuntimeError(
                "T1/T2 Epoch-selection Dev requires 400 paired non-formal "
                "scenarios (T1=200, T2=200), learned candidate_ranker "
                "selection, a post-mitigation audit step, and disabled "
                f"Teacher audit: {task_counts}"
            )
    else:
        if args.protocol_role in {
            "xplay",
            "tmcd_test",
            "ecrg_subset",
            "progressive_ablation",
            "cage",
            "mechanism_dev",
            "epoch_selection_dev",
        }:
            raise RuntimeError(f"{args.protocol_role} requires a sealed scenario source")
        groups = canonical_recovery_suite(
            scenario_count=args.scenario_count,
            group_offset=args.group_offset,
        )
    if args.protocol_role != "probe_budget" and (
        args.task_filter is not None or args.active_probe_budget is not None
    ):
        raise RuntimeError("task/budget overrides are restricted to Probe Budget")
    suite_semantic_sha256 = semantic_digest(groups)
    groups = [
        group
        for index, group in enumerate(groups)
        if index % args.shard_count == args.shard_index
    ]
    profile_generator = (
        progressive_generator(progressive.name) if progressive is not None else None
    )
    if progressive is not None and args.policy_mode in K6_POLICY_MODES:
        k6_generator = profile_generator
    elif args.policy_mode in K6_POLICY_MODES:
        k6_generator = exact_k6_generator()
    else:
        k6_generator = None
    policy = (
        CandidateRankerPolicy(
            model_path=args.model_path,
            adapter_path=manifest["adapter_path"],
            heads_path=manifest.get("heads_path"),
            score_head_path=(
                None if manifest.get("heads_path") else manifest["score_head_path"]
            ),
            device=args.device,
            max_length=args.max_length,
            score_batch_size=args.score_batch_size,
            selection_mode=str(manifest.get("selection_mode", "candidate_argmax")),
            score_composition=manifest.get("score_composition"),
            generator=profile_generator or k6_generator,
        )
        if manifest is not None
        else None
    )
    if args.protocol_role == "gate_a":
        if args.scenario_count != 200:
            raise RuntimeError("formal Gate A requires exactly 200 scenarios per seed")
        if policy is None:
            raise RuntimeError("formal Gate A requires a learned candidate ranker")
        if policy.selection_mode == "capability_chain_then_hierarchical":
            raise RuntimeError("formal Gate A forbids the hand-coded capability chain")
    teacher = PublicStateRobustTeacher()
    candidate_generator = CandidateGenerator()
    traces: list[dict[str, Any]] = []
    totals: defaultdict[str, float] = defaultdict(float)
    by_task: dict[str, defaultdict[str, float]] = {}
    by_task_execution: dict[str, defaultdict[str, float]] = {}
    probe_pending: dict[int, bool] = {}
    for scenarios in groups:
        live = []
        for row in scenarios:
            source_fingerprint = scenario_fingerprint(row)
            runtime_row = copy.deepcopy(row)
            if progressive is not None:
                metadata = dict(runtime_row.get("metadata") or {})
                metadata["experiment_variant"] = progressive.runtime_variant
                runtime_row["metadata"] = metadata
            env = instantiate_scenario(
                runtime_row,
                oracle_mode=args.policy_mode == "oracle_defender",
            )
            env.evaluation_source_fingerprint = source_fingerprint
            if args.active_probe_budget is not None:
                constraints = env.scenario.setdefault("defense_constraints", {})
                constraints["active_probe_budget"] = args.active_probe_budget
            live.append(env)
        while live:
            public_groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
            for env in live:
                post_mitigation_audit_pending = bool(
                    args.post_mitigation_audit_step
                    and getattr(env, "_v10_post_mitigation_audit_pending", False)
                )
                if not _terminal(env) or post_mitigation_audit_pending:
                    digest = public_state_digest(env.observe())
                    fair_group_id = (
                        str(
                            (env.scenario.get("metadata") or {}).get(
                                "fair_public_group_id", ""
                            )
                        )
                        if args.protocol_role
                        in {
                            "fair_trajectory",
                            "long_memory_fair",
                            "full_trajectory_causal",
                            "mechanism_dev",
                            "epoch_selection_dev",
                        }
                        else ""
                    )
                    public_groups[(fair_group_id, digest)].append(env)
            if not public_groups:
                break
            next_live: list[Any] = []
            for (fair_group_id, digest), worlds in sorted(public_groups.items()):
                observation = worlds[0].observe()
                probe_feedback_visible = _probe_feedback_visible(observation)
                model_observation = _memory_counterfactual(
                    observation,
                    mode=args.memory_counterfactual,
                )
                model_observation = _probe_counterfactual(
                    model_observation,
                    mode=args.probe_counterfactual,
                )
                candidate_observation = (
                    model_observation
                    if args.intervention_consistent_candidates
                    else observation
                )
                permutation_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{args.evaluation_seed}:{digest}".encode("utf-8")
                    ).digest()[:8],
                    "big",
                )
                k6_decision = None
                native_candidate_options = None
                if args.policy_mode in K6_POLICY_MODES:
                    if k6_generator is None:
                        raise RuntimeError("K=6 generator was not initialized")
                    permutation_seed = candidate_set_seed(
                        candidate_observation,
                        evaluation_seed=args.evaluation_seed,
                    )
                    options = k6_generator.generate(
                        candidate_observation,
                        permutation_seed=permutation_seed,
                    )
                    k6_decision = build_k6_decision(candidate_observation, options)
                    candidate_rows = list(k6_decision["candidates"])
                    if args.policy_mode == "train_greedy_k6":
                        assert policy is not None
                        decision = policy.decide(
                            model_observation,
                            seed=permutation_seed,
                            candidates=options,
                        )
                    elif args.policy_mode == "soft_selector_k6":
                        assert policy is not None
                        decision = policy.decide(
                            model_observation,
                            sample=True,
                            temperature=args.selector_temperature,
                            seed=permutation_seed,
                            candidates=options,
                        )
                    elif args.policy_mode == "random_k6":
                        selected = candidate_rows[
                            random.Random(permutation_seed).randrange(6)
                        ]
                        decision = _decision_from_k6_row(
                            selected,
                            reason="public_uniform_random_k6",
                        )
                    elif args.policy_mode == "best_of_k6":
                        assert policy is not None
                        values, _, _ = policy.score_candidates(
                            model_observation,
                            options,
                        )
                        scores = {
                            row["candidate_key"]: values[index]
                            for index, row in enumerate(candidate_rows)
                        }
                        selected = max(
                            candidate_rows,
                            key=lambda row: (
                                scores[str(row["candidate_key"])],
                                str(row["semantic_id"]),
                            ),
                        )
                        decision = _decision_from_k6_row(
                            selected,
                            reason="public_learned_global_best_of_k6",
                            scores=scores,
                        )
                    elif args.policy_mode == "ecrg_k6":
                        assert ecrg_config is not None
                        selected = select_ecrg(
                            k6_decision,
                            ecrg_config["parameters"],
                        )
                        decision = _decision_from_k6_row(
                            selected,
                            reason=f"ecrg:{selected['selection_reason']}",
                        )
                    elif args.policy_mode == "full_k6":
                        assert policy is not None and ecrg_config is not None
                        learned = policy.decide(
                            model_observation,
                            seed=permutation_seed,
                            candidates=options,
                        )
                        guard_selected = select_ecrg(
                            k6_decision,
                            ecrg_config["parameters"],
                        )
                        if int(guard_selected.get("index", -1)) < 0:
                            decision = _decision_from_k6_row(
                                guard_selected,
                                reason=(
                                    "full_ecrg:"
                                    f"{guard_selected['selection_reason']}"
                                ),
                                scores=learned.scores,
                            )
                        else:
                            thresholds = ecrg_config["parameters"][
                                "hard_admission_thresholds"
                            ]
                            admitted = [
                                row
                                for row in candidate_rows
                                if admission_allowed(row["features"], thresholds)
                            ]
                            learned_row = next(
                                (
                                    row
                                    for row in admitted
                                    if row["candidate_key"] == learned.candidate_id
                                ),
                                None,
                            )
                            if learned.valid and learned_row is not None:
                                decision = CandidateDecision(
                                    **{
                                        **learned.__dict__,
                                        "reason": "full_ecrg:learned_choice_admitted",
                                    }
                                )
                            elif admitted:
                                selected = max(
                                    admitted,
                                    key=lambda row: (
                                        learned.scores.get(
                                            str(row["candidate_key"]),
                                            float("-inf"),
                                        ),
                                        str(row["semantic_id"]),
                                    ),
                                )
                                decision = _decision_from_k6_row(
                                    selected,
                                    reason="full_ecrg:next_admitted_learned_rank",
                                    scores=learned.scores,
                                )
                            else:
                                decision = _decision_from_k6_row(
                                    k6_decision["fallback"],
                                    reason="full_ecrg:empty_admissible_set",
                                    scores=learned.scores,
                                )
                    else:
                        raise AssertionError(args.policy_mode)
                else:
                    if policy is not None:
                        # Legacy evaluations freeze the factual candidate set.
                        # Causal evaluations instead generate from the intervened
                        # view so candidate summaries cannot leak masked content.
                        native_candidate_options = candidate_generator.generate(
                            candidate_observation,
                            permutation_seed=permutation_seed,
                        )
                        decision = policy.decide(
                            model_observation,
                            seed=permutation_seed,
                            candidates=native_candidate_options,
                        )
                    else:
                        decision = static_anchor_decision(
                            policy_mode=args.policy_mode,
                            observation=observation,
                            worlds=worlds,
                        )
                teacher_decision = (
                    teacher.decide(worlds, horizon=3, enforce_min_worlds=True)
                    if args.policy_mode == "candidate_ranker"
                    and len(worlds) >= 2
                    and not args.skip_teacher_audit
                    else None
                )
                (
                    candidate_regret,
                    raw_candidate_regret,
                    teacher_regret_audit,
                ) = _teacher_regret_metrics(
                    teacher_decision=teacher_decision,
                    selected_semantic_id=decision.semantic_id,
                    options=(
                        candidate_generator.generate_all(observation)
                        if teacher_decision is not None
                        else []
                    ),
                    core_tolerance=teacher.core_tolerance,
                )
                actionable = bool(
                    teacher_decision is not None
                    and teacher_decision.advantage_over_observe > 0.05 + 1.0e-12
                )
                trace = {
                    "time": int(observation.get("time", 0)),
                    "public_state_digest": digest,
                    "model_state_digest": public_state_digest(model_observation),
                    "fair_public_group_id": fair_group_id or None,
                    "task_id": _task_id(worlds[0]),
                    "candidate_id": decision.candidate_id,
                    "candidate_semantic_id": decision.semantic_id,
                    "candidate_count": decision.candidate_count,
                    "candidate_score": decision.score,
                    "candidate_regret": candidate_regret,
                    "raw_candidate_regret": raw_candidate_regret,
                    "teacher_regret_audit": teacher_regret_audit,
                    "probe_feedback_visible_in_factual_state": probe_feedback_visible,
                    "probe_counterfactual": args.probe_counterfactual,
                    "memory_counterfactual": args.memory_counterfactual,
                    "information_intervention_applied": information_intervention,
                    "valid": decision.valid,
                    "invalid_noop": decision.invalid_noop,
                    "reason": decision.reason,
                    "gated_family": decision.gated_family,
                    "predicted_branch": decision.predicted_branch,
                    "predicted_phase": decision.predicted_phase,
                    "phase_gate_probabilities": decision.phase_probabilities,
                    "memory_operation_probabilities": (
                        decision.memory_operation_probabilities
                    ),
                    "budget_state_prediction": decision.budget_state,
                    "operation_budget_masked_candidate_ids": list(
                        decision.operation_budget_masked_candidate_ids
                    ),
                    "post_mitigation_audit_state": any(
                        isinstance(row, Mapping)
                        and bool(row.get("authorized", False))
                        and str(row.get("action", ""))
                        in {
                            "DeployDecoy",
                            "LimitSession",
                            "ShadowBlock",
                            "Isolate",
                            "Restore",
                            "Remove",
                        }
                        for row in (
                            (observation.get("defender_state") or {}).get(
                                "response_history", []
                            )
                            or []
                        )
                    ),
                    "action_flags": decision.action_flags.to_dict(),
                    "selected_packet": copy.deepcopy(decision.packet),
                    "candidate_set_sha256": (
                        candidate_set_digest(k6_decision)
                        if k6_decision is not None
                        else semantic_digest(
                            [
                                row.public_record(include_packet=True)
                                for row in native_candidate_options
                            ]
                        )
                        if native_candidate_options is not None
                        else None
                    ),
                    "candidate_set_semantic_ids": (
                        [
                            str(row["semantic_id"])
                            for row in k6_decision["candidates"]
                        ]
                        if k6_decision is not None
                        else [
                            str(row.semantic_id)
                            for row in native_candidate_options
                        ]
                        if native_candidate_options is not None
                        else None
                    ),
                    "follows_active_probe": any(
                        probe_pending.get(id(env), False) for env in worlds
                    ),
                    "teacher_actionable": actionable,
                    "teacher_candidate_id": (
                        teacher_decision.selected_candidate_id
                        if teacher_decision is not None
                        else None
                    ),
                    "teacher_action_family": (
                        teacher_decision.selected_category
                        if teacher_decision is not None
                        else None
                    ),
                }
                traces.append(trace)
                totals["decision_count"] += 1
                totals["valid_decision_count"] += int(decision.valid)
                totals["actionable_count"] += int(actionable)
                totals["teacher_audit_decision_count"] += int(
                    teacher_decision is not None
                )
                totals["teacher_regret_aligned_count"] += int(
                    bool(teacher_regret_audit.get("available", False))
                )
                totals["actionable_observe_count"] += int(
                    actionable and decision.action_flags.observe_only
                )
                if candidate_regret is not None:
                    totals["candidate_regret_sum"] += float(candidate_regret)
                    totals["candidate_regret_count"] += 1
                execution_results = []
                for env in worlds:
                    prior_attack_mitigated = bool(env.attack_mitigated)
                    execution_result = {
                        "scenario_id": str(env.scenario.get("scenario_id", "")),
                        "fair_public_group_id": (
                            str(
                                (env.scenario.get("metadata") or {}).get(
                                    "fair_public_group_id", ""
                                )
                            )
                            or None
                        ),
                        "scenario_fingerprint": str(
                            getattr(
                                env,
                                "evaluation_source_fingerprint",
                                scenario_fingerprint(env.scenario),
                            )
                        ),
                    }
                    probe_pending[id(env)] = decision.action_flags.active_probe
                    env.step(copy.deepcopy(decision.packet))
                    public_step = env.history[-1]
                    execution_result.update(
                        {
                            "response_result": copy.deepcopy(
                                public_step.get("response_result") or {}
                            ),
                            "tool_result": copy.deepcopy(
                                public_step.get("tool_result") or {}
                            ),
                            "trust_events": copy.deepcopy(
                                public_step.get("trust_events") or []
                            ),
                            "memory_events": copy.deepcopy(
                                public_step.get("memory_events") or []
                            ),
                            "accepted_memory_ids": copy.deepcopy(
                                public_step.get("accepted_memory_ids") or []
                            ),
                        }
                    )
                    needs_post_mitigation_audit = bool(
                        args.post_mitigation_audit_step
                        and _task_id(env) == "T4"
                        and bool(env.attack_mitigated)
                        and not prior_attack_mitigated
                    )
                    if needs_post_mitigation_audit:
                        env._v10_post_mitigation_audit_pending = True
                        execution_result["terminal"] = False
                        execution_result["post_mitigation_audit_pending"] = True
                        next_live.append(env)
                    elif _terminal(env):
                        env._v10_post_mitigation_audit_pending = False
                        task = _task_id(env)
                        task_totals = by_task.setdefault(task, defaultdict(float))
                        score = score_trajectory_v2(env)
                        utility = recovery_core_utility(env, score)
                        raw_su = raw_safe_utility(
                            score,
                            steps=len(env.history),
                            max_steps=env.max_steps,
                            business_budget=float(
                                (env.scenario.get("defense_constraints") or {}).get(
                                    "business_budget", 5.0
                                )
                            ),
                        )
                        execution_result["terminal"] = True
                        execution_result["terminal_task_id"] = task
                        execution_result["terminal_score"] = copy.deepcopy(score)
                        execution_result["terminal_safe_utility"] = float(utility)
                        execution_result["terminal_raw_safe_utility"] = float(raw_su)
                        execution_result["terminal_steps"] = len(env.history)
                        execution_result["terminal_max_steps"] = env.max_steps
                        execution_result["terminal_business_budget"] = float(
                            (env.scenario.get("defense_constraints") or {}).get(
                                "business_budget", 5.0
                            )
                        )
                        for counter in (totals, task_totals):
                            counter["scenario_count"] += 1
                            counter["safe_success_sum"] += int(score.get("safe_success", False))
                            counter["attack_mitigation_sum"] += int(
                                score.get("attack_mitigated", False)
                            )
                            counter["probe_yield_sum"] += float(score.get("probe_yield", 0.0))
                            counter["probe_result_availability_sum"] += float(
                                score.get("probe_result_availability_rate", 0.0)
                            )
                            counter["probe_grounded_state_updates_sum"] += float(
                                score.get("probe_grounded_state_updates", 0.0)
                            )
                            counter["probe_grounded_action_revisions_sum"] += float(
                                score.get("probe_grounded_action_revisions", 0.0)
                            )
                            counter["probe_grounded_use_sum"] += int(
                                float(score.get("probe_grounded_state_updates", 0.0))
                                > 0.0
                                or float(
                                    score.get("probe_grounded_action_revisions", 0.0)
                                )
                                > 0.0
                            )
                            counter["invalid_operation_sum"] += float(
                                score.get("invalid_state_operations", 0.0)
                            )
                            counter["safe_utility_sum"] += float(utility)
                            counter["raw_safe_utility_sum"] += float(raw_su)
                            counter["overresponse_sum"] += int(
                                score.get("overresponse", False)
                            )
                            counter["business_cost_sum"] += float(
                                score.get("business_cost", 0.0)
                            )
                            counter["verification_cost_sum"] += float(
                                score.get("verification_cost", 0.0)
                            )
                            counter["betrayal_detection_sum"] += float(
                                score.get("betrayal_detection_rate", 0.0)
                            )
                            counter["poison_success_sum"] += float(
                                score.get(
                                    "poison_promotion_rate",
                                    int(
                                        float(score.get("false_promoted_count", 0.0))
                                        > 0.0
                                    ),
                                )
                            )
                    else:
                        execution_result["terminal"] = False
                        next_live.append(env)
                    execution_results.append(execution_result)
                trace["execution_results"] = execution_results
                committed_trust = any(
                    any(bool(item.get("committed", False)) for item in result["trust_events"])
                    for result in execution_results
                )
                committed_memory = any(
                    any(bool(item.get("committed", False)) for item in result["memory_events"])
                    for result in execution_results
                )
                accepted_memory = any(
                    bool(result["accepted_memory_ids"]) for result in execution_results
                )
                trace["committed_trust_operation"] = committed_trust
                trace["committed_memory_operation"] = committed_memory
                trace["accepted_memory_use"] = accepted_memory
                trace["post_mitigation_stop"] = bool(
                    trace["post_mitigation_audit_state"]
                    and decision.action_flags.observe_only
                )
                execution_counter = by_task_execution.setdefault(
                    str(trace["task_id"]), defaultdict(float)
                )
                for counter in (totals, execution_counter):
                    counter["execution_decision_count"] += 1
                    counter["committed_trust_decision_count"] += int(committed_trust)
                    counter["committed_memory_decision_count"] += int(committed_memory)
                    counter["accepted_memory_use_decision_count"] += int(accepted_memory)
            live = next_live

    def terminal_metrics(counter: dict[str, float]) -> dict[str, Any]:
        scenarios = max(1.0, counter.get("scenario_count", 0.0))
        return {
            "scenario_count": int(counter.get("scenario_count", 0.0)),
            "safe_success": counter.get("safe_success_sum", 0.0) / scenarios,
            "attack_mitigation": counter.get("attack_mitigation_sum", 0.0) / scenarios,
            "probe_yield": counter.get("probe_yield_sum", 0.0) / scenarios,
            "probe_result_availability": counter.get(
                "probe_result_availability_sum", 0.0
            )
            / scenarios,
            "probe_grounded_state_updates": counter.get(
                "probe_grounded_state_updates_sum", 0.0
            )
            / scenarios,
            "probe_grounded_action_revisions": counter.get(
                "probe_grounded_action_revisions_sum", 0.0
            )
            / scenarios,
            "probe_grounded_use_rate": counter.get(
                "probe_grounded_use_sum", 0.0
            )
            / scenarios,
            "invalid_operation_rate": counter.get("invalid_operation_sum", 0.0)
            / scenarios,
            "safe_utility": counter.get("safe_utility_sum", 0.0) / scenarios,
            "raw_safe_utility": counter.get("raw_safe_utility_sum", 0.0)
            / scenarios,
            "overresponse": counter.get("overresponse_sum", 0.0) / scenarios,
            "business_cost": counter.get("business_cost_sum", 0.0) / scenarios,
            "verification_cost": counter.get("verification_cost_sum", 0.0)
            / scenarios,
            "betrayal_detection": counter.get("betrayal_detection_sum", 0.0)
            / scenarios,
            "poison_success": counter.get("poison_success_sum", 0.0)
            / scenarios,
        }

    def execution_metrics(counter: dict[str, float]) -> dict[str, float | int]:
        decisions = max(1.0, counter.get("execution_decision_count", 0.0))
        return {
            "execution_decision_count": int(
                counter.get("execution_decision_count", 0.0)
            ),
            "trust_operation_rate": counter.get(
                "committed_trust_decision_count", 0.0
            )
            / decisions,
            "memory_operation_rate": counter.get(
                "committed_memory_decision_count", 0.0
            )
            / decisions,
            "memory_use_rate": counter.get(
                "accepted_memory_use_decision_count", 0.0
            )
            / decisions,
            "accepted_memory_use_count": int(
                counter.get("accepted_memory_use_decision_count", 0.0)
            ),
        }

    action_metrics = summarize_candidate_traces(traces)
    decisions = max(1.0, totals["decision_count"])
    metrics = {
        **terminal_metrics(totals),
        **action_metrics,
        **execution_metrics(totals),
        "action_validity": totals["valid_decision_count"] / decisions,
        "actionable_observe_rate": totals["actionable_observe_count"]
        / max(1.0, totals["actionable_count"]),
        "actionable_decision_count": int(totals["actionable_count"]),
        "teacher_regret_coverage": totals["teacher_regret_aligned_count"]
        / max(1.0, totals["teacher_audit_decision_count"]),
        "decoding": (
            str(manifest.get("selection_mode", "candidate_argmax"))
            if manifest is not None
            else args.policy_mode
        ),
        "candidate_count": (
            6
            if args.policy_mode in K6_POLICY_MODES
            else max(
                (int(row.get("candidate_count", 0)) for row in traces),
                default=1,
            )
        ),
        "by_task_terminal": {
            task: terminal_metrics(counter) for task, counter in sorted(by_task.items())
        },
        "by_task_execution": {
            task: execution_metrics(counter)
            for task, counter in sorted(by_task_execution.items())
        },
    }
    post_mitigation_states = [
        row for row in traces if bool(row.get("post_mitigation_audit_state", False))
    ]
    metrics["phase_trace_coverage"] = sum(
        bool(row.get("predicted_phase"))
        and bool(row.get("phase_gate_probabilities"))
        for row in traces
    ) / max(1, len(traces))
    metrics["post_mitigation_audit_state_count"] = len(post_mitigation_states)
    metrics["post_mitigation_stop_rate"] = sum(
        bool(row.get("post_mitigation_stop", False))
        for row in post_mitigation_states
    ) / max(1, len(post_mitigation_states))
    metrics["mechanism_hard_gates"] = {
        "accepted_memory_use": int(
            (by_task_execution.get("T3") or {}).get(
                "accepted_memory_use_decision_count", 0.0
            )
        )
        > 0,
        "probe_grounded_action_revision": float(
            totals.get("probe_grounded_action_revisions_sum", 0.0)
        )
        > 0.0,
        "post_mitigation_stop": bool(post_mitigation_states)
        and metrics["post_mitigation_stop_rate"] >= 1.0 - 1.0e-12,
    }
    metrics["mechanism_hard_gates_passed"] = all(
        metrics["mechanism_hard_gates"].values()
    )
    metrics["t1_probe_yield"] = float(
        (metrics["by_task_terminal"].get("T1") or {}).get("probe_yield", 0.0)
    )
    metrics["t2_betrayal_detection"] = float(
        (metrics["by_task_terminal"].get("T2") or {}).get(
            "betrayal_detection", 0.0
        )
    )
    metrics["t3_poison_success"] = float(
        (metrics["by_task_terminal"].get("T3") or {}).get(
            "poison_success", 0.0
        )
    )
    metrics["t4_overresponse"] = float(
        (metrics["by_task_terminal"].get("T4") or {}).get("overresponse", 0.0)
    )
    metrics["t4_business_cost"] = float(
        (metrics["by_task_terminal"].get("T4") or {}).get("business_cost", 0.0)
    )
    payload = {
        "schema_version": 1,
        "kind": "candidate_policy_evaluation_shard",
        "created_at": utc_now(),
        "ranker_manifest_sha256": (
            sha256_file(args.ranker_manifest) if args.ranker_manifest else None
        ),
        "policy_mode": args.policy_mode,
        "privileged_nsu_anchor": args.policy_mode in {
            "oracle_defender",
        },
        "scenario_count_requested": args.scenario_count,
        "evaluation_seed": args.evaluation_seed,
        "protocol_role": args.protocol_role,
        "probe_counterfactual": args.probe_counterfactual,
        "memory_counterfactual": args.memory_counterfactual,
        "intervention_consistent_candidates": (
            args.intervention_consistent_candidates
        ),
        "information_intervention": information_intervention,
        "teacher_audit_enabled": not args.skip_teacher_audit,
        "dca_frozen": True,
        "ecrg_enabled": args.policy_mode in ECRG_POLICY_MODES,
        "ecrg_config_sha256": ecrg_config_sha256,
        "candidate_protocol": (
            "progressive_public_exact_k6_profile"
            if progressive is not None
            else "shared_public_exact_k6_one_per_family"
            if args.policy_mode in K6_POLICY_MODES
            else "native_candidate_set"
        ),
        "candidate_sets_shared_at_matching_public_states": (
            args.policy_mode in K6_POLICY_MODES
            or args.policy_mode == "candidate_ranker"
            or progressive is not None
        ),
        "progressive_arm": progressive.name if progressive is not None else None,
        "experiment_variant": (
            progressive.runtime_variant if progressive is not None else None
        ),
        "candidate_profile_quotas": (
            dict(progressive.quotas) if progressive is not None else None
        ),
        "suite_semantic_sha256": suite_semantic_sha256,
        "scenario_source": (
            str(args.scenario_source.resolve()) if args.scenario_source else None
        ),
        "scenario_source_sha256": scenario_source_sha256,
        "task_filter": args.task_filter,
        "active_probe_budget": args.active_probe_budget,
        "post_mitigation_audit_step": args.post_mitigation_audit_step,
        "group_offset": args.group_offset,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "metrics": metrics,
        "raw_totals": dict(totals),
        "traces": traces,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


def merge(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not shards:
        raise ValueError("no evaluation shards")
    expected = set(range(int(shards[0]["shard_count"])))
    if {int(row["shard_index"]) for row in shards} != expected:
        raise RuntimeError("evaluation shard coverage mismatch")
    invariant = (
        "ranker_manifest_sha256",
        "policy_mode",
        "privileged_nsu_anchor",
        "group_offset",
        "shard_count",
        "scenario_count_requested",
        "evaluation_seed",
        "protocol_role",
        "probe_counterfactual",
        "memory_counterfactual",
        "intervention_consistent_candidates",
        "information_intervention",
        "teacher_audit_enabled",
        "dca_frozen",
        "ecrg_enabled",
        "ecrg_config_sha256",
        "candidate_protocol",
        "candidate_sets_shared_at_matching_public_states",
        "suite_semantic_sha256",
        "scenario_source_sha256",
        "task_filter",
        "active_probe_budget",
        "post_mitigation_audit_step",
        "progressive_arm",
        "experiment_variant",
        "candidate_profile_quotas",
    )
    for key in invariant:
        if len({json.dumps(row.get(key), sort_keys=True) for row in shards}) != 1:
            raise RuntimeError(f"evaluation shard invariant mismatch: {key}")
    traces = [trace for shard in shards for trace in shard["traces"]]
    terminal_sums: defaultdict[str, float] = defaultdict(float)
    by_task: dict[str, defaultdict[str, float]] = {}
    terminal_keys = (
        "safe_success",
        "attack_mitigation",
        "probe_yield",
        "probe_result_availability",
        "probe_grounded_state_updates",
        "probe_grounded_action_revisions",
        "probe_grounded_use_rate",
        "invalid_operation_rate",
        "safe_utility",
        "raw_safe_utility",
        "overresponse",
        "business_cost",
        "verification_cost",
        "betrayal_detection",
        "poison_success",
    )
    for shard in shards:
        metrics = shard["metrics"]
        scenarios = int(metrics["scenario_count"])
        terminal_sums["scenario_count"] += scenarios
        for key in terminal_keys:
            terminal_sums[f"{key}_sum"] += float(metrics.get(key, 0.0)) * scenarios
        for task, values in metrics.get("by_task_terminal", {}).items():
            counter = by_task.setdefault(task, defaultdict(float))
            count = int(values["scenario_count"])
            counter["scenario_count"] += count
            for key in terminal_keys:
                counter[f"{key}_sum"] += float(values.get(key, 0.0)) * count

    def pack(counter: dict[str, float]) -> dict[str, Any]:
        count = max(1.0, counter.get("scenario_count", 0.0))
        return {
            "scenario_count": int(counter.get("scenario_count", 0.0)),
            **{
                key: counter.get(f"{key}_sum", 0.0) / count
                for key in terminal_keys
            },
        }

    action = summarize_candidate_traces(traces)
    actionable = [row for row in traces if bool(row.get("teacher_actionable", False))]
    actionable_observe = sum(
        bool((row.get("action_flags") or {}).get("observe_only", False))
        for row in actionable
    )
    execution_decisions = max(1, len(traces))
    committed_trust = sum(
        bool(row.get("committed_trust_operation", False)) for row in traces
    )
    committed_memory = sum(
        bool(row.get("committed_memory_operation", False)) for row in traces
    )
    accepted_memory = sum(bool(row.get("accepted_memory_use", False)) for row in traces)
    by_task_execution: dict[str, dict[str, Any]] = {}
    for task in sorted({str(row.get("task_id", "unknown")) for row in traces}):
        task_traces = [row for row in traces if str(row.get("task_id")) == task]
        task_count = max(1, len(task_traces))
        by_task_execution[task] = {
            "execution_decision_count": len(task_traces),
            "trust_operation_rate": sum(
                bool(row.get("committed_trust_operation", False))
                for row in task_traces
            )
            / task_count,
            "memory_operation_rate": sum(
                bool(row.get("committed_memory_operation", False))
                for row in task_traces
            )
            / task_count,
            "memory_use_rate": sum(
                bool(row.get("accepted_memory_use", False)) for row in task_traces
            )
            / task_count,
            "accepted_memory_use_count": sum(
                bool(row.get("accepted_memory_use", False))
                for row in task_traces
            ),
        }
    metrics = {
        **pack(terminal_sums),
        **action,
        "execution_decision_count": len(traces),
        "trust_operation_rate": committed_trust / execution_decisions,
        "memory_operation_rate": committed_memory / execution_decisions,
        "memory_use_rate": accepted_memory / execution_decisions,
        "action_validity": 1.0 - action["invalid_noop_rate"],
        "actionable_observe_rate": actionable_observe / max(1, len(actionable)),
        "actionable_decision_count": len(actionable),
        "teacher_regret_coverage": sum(
            bool((row.get("teacher_regret_audit") or {}).get("available", False))
            for row in traces
        )
        / max(
            1,
            sum(
                (row.get("teacher_regret_audit") or {}).get("reason")
                != "teacher_disabled"
                for row in traces
            ),
        ),
        "decoding": str(shards[0]["metrics"].get("decoding", "candidate_argmax")),
        "candidate_count": int(shards[0]["metrics"].get("candidate_count", 0)),
        "by_task_terminal": {task: pack(row) for task, row in sorted(by_task.items())},
        "by_task_execution": by_task_execution,
    }
    post_mitigation_states = [
        row for row in traces if bool(row.get("post_mitigation_audit_state", False))
    ]
    metrics["phase_trace_coverage"] = sum(
        bool(row.get("predicted_phase"))
        and bool(row.get("phase_gate_probabilities"))
        for row in traces
    ) / max(1, len(traces))
    metrics["post_mitigation_audit_state_count"] = len(post_mitigation_states)
    metrics["post_mitigation_stop_rate"] = sum(
        bool(row.get("post_mitigation_stop", False))
        for row in post_mitigation_states
    ) / max(1, len(post_mitigation_states))
    metrics["mechanism_hard_gates"] = {
        "accepted_memory_use": int(
            (by_task_execution.get("T3") or {}).get(
                "accepted_memory_use_count", 0
            )
        )
        > 0,
        "probe_grounded_action_revision": float(
            terminal_sums.get("probe_grounded_action_revisions_sum", 0.0)
        )
        > 0.0,
        "post_mitigation_stop": bool(post_mitigation_states)
        and metrics["post_mitigation_stop_rate"] >= 1.0 - 1.0e-12,
    }
    metrics["mechanism_hard_gates_passed"] = all(
        metrics["mechanism_hard_gates"].values()
    )
    metrics["t1_probe_yield"] = float(
        (metrics["by_task_terminal"].get("T1") or {}).get("probe_yield", 0.0)
    )
    metrics["t2_betrayal_detection"] = float(
        (metrics["by_task_terminal"].get("T2") or {}).get(
            "betrayal_detection", 0.0
        )
    )
    metrics["t3_poison_success"] = float(
        (metrics["by_task_terminal"].get("T3") or {}).get(
            "poison_success", 0.0
        )
    )
    metrics["t4_overresponse"] = float(
        (metrics["by_task_terminal"].get("T4") or {}).get("overresponse", 0.0)
    )
    metrics["t4_business_cost"] = float(
        (metrics["by_task_terminal"].get("T4") or {}).get("business_cost", 0.0)
    )
    payload = {
        "schema_version": 1,
        "kind": "candidate_policy_evaluation",
        "created_at": utc_now(),
        **{
            key: shards[0].get(key)
            for key in (
                "ranker_manifest_sha256",
                "policy_mode",
                "privileged_nsu_anchor",
                "scenario_count_requested",
                "evaluation_seed",
                "protocol_role",
                "probe_counterfactual",
                "memory_counterfactual",
                "intervention_consistent_candidates",
                "information_intervention",
                "teacher_audit_enabled",
                "dca_frozen",
                "ecrg_enabled",
                "ecrg_config_sha256",
                "candidate_protocol",
                "candidate_sets_shared_at_matching_public_states",
                "suite_semantic_sha256",
                "scenario_source",
                "scenario_source_sha256",
                "task_filter",
                "active_probe_budget",
                "post_mitigation_audit_step",
                "progressive_arm",
                "experiment_variant",
                "candidate_profile_quotas",
                "group_offset",
            )
        },
        "metrics": metrics,
        "traces": traces,
        "shard_sha256": {path.name: sha256_file(path) for path in args.inputs},
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--ranker-manifest", type=Path)
    run_parser.add_argument(
        "--policy-mode",
        choices=(
            "candidate_ranker",
            "random_policy",
            "oracle_defender",
            "train_greedy_k6",
            "random_k6",
            "best_of_k6",
            "soft_selector_k6",
            "ecrg_k6",
            "full_k6",
        ),
        default="candidate_ranker",
    )
    run_parser.add_argument("--ecrg-config", type=Path)
    run_parser.add_argument("--progressive-arm", choices=PROGRESSIVE_ARM_ORDER)
    run_parser.add_argument("--selector-temperature", type=float, default=1.0)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--scenario-count", type=int, default=32)
    run_parser.add_argument("--scenario-source", type=Path)
    run_parser.add_argument("--task-filter", choices=("T1", "T2", "T3", "T4"))
    run_parser.add_argument("--active-probe-budget", type=int, choices=(0, 1, 2, 3))
    run_parser.add_argument("--group-offset", type=int, default=20000)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--shard-count", type=int, default=1)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--max-length", type=int, default=2048)
    run_parser.add_argument("--score-batch-size", type=int, default=4)
    run_parser.add_argument("--evaluation-seed", type=int, default=20260720)
    run_parser.add_argument(
        "--memory-counterfactual",
        choices=("none", "drop", "shuffle"),
        default="none",
    )
    run_parser.add_argument(
        "--probe-counterfactual",
        choices=("none", "drop"),
        default="none",
    )
    run_parser.add_argument(
        "--intervention-consistent-candidates",
        action="store_true",
        help=(
            "generate candidates from the information-intervened observation "
            "so candidate text cannot leak masked Probe or Memory content"
        ),
    )
    run_parser.add_argument("--skip-teacher-audit", action="store_true")
    run_parser.add_argument(
        "--post-mitigation-audit-step",
        action="store_true",
        help=(
            "continue T4 for one public decision after a successful mitigation "
            "and require the policy to stop with Observe"
        ),
    )
    run_parser.add_argument(
        "--protocol-role",
        choices=(
            "diagnostic",
            "gate_a",
            "xplay",
            "tmcd_test",
            "probe_budget",
            "ecrg_subset",
            "progressive_ablation",
            "cage",
            "fair_trajectory",
            "long_memory_fair",
            "full_trajectory_causal",
            "mechanism_dev",
            "epoch_selection_dev",
        ),
        default="diagnostic",
    )
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run" and args.selector_temperature <= 0.0:
        parser.error("selector-temperature must be positive")
    return run(args) if args.command == "run" else merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
