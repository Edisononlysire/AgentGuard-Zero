"""Build and audit public-only bootstrap SFT records."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from agentguard_zero.recovery.public_teacher import (
    PublicStateRobustTeacher,
    compact_wire_json,
    public_state_digest,
)
from agentguard_zero.schemas.action_schema_v4 import parse_action_json_v4
from agentguard_zero.training.vda_dataset import build_vda_prompt
from agentguard_zero.world.public_projector import assert_public


ACTION_CATEGORIES = (
    "observe",
    "passive_verification",
    "active_probe",
    "trust",
    "memory",
    "mitigation",
)


@dataclass
class BootstrapBuildResult:
    train_records: list[dict[str, Any]]
    audit_records: list[dict[str, Any]]
    manifest: dict[str, Any]


def _scenario_hash(scenario: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(scenario),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _public_context_from_prompt(prompt: str) -> dict[str, Any]:
    marker = "\nCurrent decision instance:"
    if marker not in prompt:
        raise ValueError("bootstrap prompt is missing public-context marker")
    payload = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(payload, dict):
        raise ValueError("bootstrap public context is not an object")
    assert_public(payload)
    return payload


def build_bootstrap_records(
    scenario_groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    teacher: PublicStateRobustTeacher | None = None,
    max_records: int = 3_000,
) -> BootstrapBuildResult:
    from agentguard_zero.env.scenario_instantiator import instantiate_scenario

    policy = teacher or PublicStateRobustTeacher()
    train_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    scenario_count = sum(len(group) for group in scenario_groups)
    if not scenario_groups or any(len(group) < 2 for group in scenario_groups):
        raise ValueError("every initial public-state group must contain at least two worlds")

    for group_index, scenarios in enumerate(scenario_groups):
        live = [instantiate_scenario(copy.deepcopy(dict(item))) for item in scenarios]
        first_decision = True
        decision_index = 0
        while live and len(train_records) < max_records:
            groups: dict[str, list[Any]] = defaultdict(list)
            for env in live:
                groups[public_state_digest(env.observe())].append(env)
            next_live: list[Any] = []
            for public_group in groups.values():
                observation = public_group[0].observe()
                decision = policy.decide(
                    public_group,
                    horizon=3,
                    enforce_min_worlds=first_decision,
                )
                prompt_scenario = copy.deepcopy(public_group[0].scenario)
                # Branch-specific scenario IDs are opaque but can still become
                # a memorized hidden-world key. Recovery prompts therefore use
                # one identifier derived solely from the shared public state.
                prompt_scenario["scenario_id"] = (
                    f"recovery-public-{decision.public_state_digest[:20]}"
                )
                prompt = build_vda_prompt(
                    prompt_scenario,
                    observation,
                    experiment_variant="full",
                )
                _public_context_from_prompt(prompt)
                target = compact_wire_json(decision.selected_packet)
                _, valid, reason = parse_action_json_v4(target)
                if not valid:
                    raise ValueError(f"teacher emitted an invalid target: {reason}")
                replica_count = min(
                    len(public_group),
                    max_records - len(train_records),
                )
                source_hashes = sorted(
                    _scenario_hash(env.scenario) for env in public_group
                )
                for replica_index in range(replica_count):
                    record_id = hashlib.sha256(
                        (
                            f"{decision.public_state_digest}:"
                            f"{decision.selected_candidate_id}:"
                            f"{group_index}:{decision_index}:{replica_index}"
                        ).encode("utf-8")
                    ).hexdigest()
                    train_records.append(
                        {
                            "record_id": record_id,
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": target},
                            ],
                            "prompt": prompt,
                            "target": target,
                            "public_state_digest": decision.public_state_digest,
                            "action_category": decision.selected_category,
                            "source_policy": "public_state_robust_teacher",
                        }
                    )
                    audit = decision.to_audit_dict()
                    audit.update(
                        {
                            "record_id": record_id,
                            "initial_group_index": group_index,
                            "decision_index": decision_index,
                            "public_state_replica_index": replica_index,
                            "public_state_replica_count": len(public_group),
                            "source_scenario_hashes": source_hashes,
                            "target_sha256": hashlib.sha256(
                                target.encode("utf-8")
                            ).hexdigest(),
                            "model_input_hidden_state": False,
                            "model_target_hidden_state": False,
                            "hidden_state_usage": "offline_robust_utility_only",
                        }
                    )
                    audit_records.append(audit)
                decision_index += 1
                for env in public_group:
                    env.step(copy.deepcopy(decision.selected_packet))
                    if not (
                        env.t >= env.max_steps
                        or env.attack_mitigated
                        or env.attack_success
                    ):
                        next_live.append(env)
                if len(train_records) >= max_records:
                    break
            live = next_live
            first_decision = False

    manifest = audit_bootstrap_records(train_records, audit_records)
    manifest.update(
        {
            "scenario_count": scenario_count,
            "initial_public_group_count": len(scenario_groups),
            "record_cap": max_records,
            "teacher": "public_state_robust_max_min",
            "human_action_labels": 0,
            "lineage": "new_recovery_lineage",
        }
    )
    return BootstrapBuildResult(train_records, audit_records, manifest)


def audit_bootstrap_records(
    train_records: Iterable[Mapping[str, Any]],
    audit_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    train = [dict(item) for item in train_records]
    audits = [dict(item) for item in audit_records]
    failures: list[str] = []
    if len(train) != len(audits):
        failures.append("train_audit_count_mismatch")
    train_ids = [str(item.get("record_id", "")) for item in train]
    audit_ids = [str(item.get("record_id", "")) for item in audits]
    if not all(train_ids) or len(set(train_ids)) != len(train_ids):
        failures.append("invalid_or_duplicate_train_record_id")
    if set(train_ids) != set(audit_ids):
        failures.append("train_audit_record_id_mismatch")

    categories: Counter[str] = Counter()
    prompt_hashes: set[str] = set()
    for row in train:
        allowed = {
            "record_id",
            "messages",
            "prompt",
            "target",
            "public_state_digest",
            "action_category",
            "source_policy",
        }
        if set(row).difference(allowed):
            failures.append("unexpected_training_field")
        try:
            context = _public_context_from_prompt(str(row.get("prompt", "")))
            assert_public(context)
        except Exception:
            failures.append("hidden_or_invalid_prompt_context")
        _, valid, _ = parse_action_json_v4(str(row.get("target", "")))
        if not valid:
            failures.append("invalid_target")
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("content") != row.get("prompt")
            or messages[1].get("content") != row.get("target")
        ):
            failures.append("messages_prompt_target_mismatch")
        category = str(row.get("action_category", ""))
        if category not in ACTION_CATEGORIES:
            failures.append("invalid_action_category")
        categories[category] += 1
        prompt_hashes.add(
            hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest()
        )

    for row in audits:
        if row.get("hidden_state_in_target") is not False:
            failures.append("hidden_state_in_target")
        if row.get("model_input_hidden_state") is not False:
            failures.append("hidden_state_in_model_input")
        if row.get("model_target_hidden_state") is not False:
            failures.append("hidden_state_in_model_target")
        if int(row.get("world_count", 0)) < 1:
            failures.append("missing_robust_world_audit")
        if row.get("hidden_state_usage") != "offline_robust_utility_only":
            failures.append("invalid_hidden_state_usage")

    count = len(train)
    ratios = {
        category: categories[category] / max(1, count)
        for category in ACTION_CATEGORIES
    }
    max_ratio = max(ratios.values(), default=1.0)
    if count and max_ratio > 0.40 + 1.0e-12:
        failures.append("single_action_category_above_40pct")
    required_support = {
        "active_probe",
        "trust",
        "memory",
        "mitigation",
    }
    missing_support = sorted(
        category for category in required_support if categories[category] == 0
    )
    if missing_support:
        failures.append("missing_action_support:" + ",".join(missing_support))

    failures = list(dict.fromkeys(failures))
    return {
        "accepted": not failures,
        "status": "accepted" if not failures else "rejected",
        "failures": failures,
        "record_count": count,
        "audit_record_count": len(audits),
        "unique_prompt_count": len(prompt_hashes),
        "action_category_counts": dict(categories),
        "action_category_ratios": ratios,
        "single_category_max_ratio": max_ratio,
        "forbidden_training_fields": [
            "scenario",
            "hidden_state",
            "oracle",
            "teacher_q_values",
            "source_scenario_hashes",
        ],
        "hidden_state_access": "audit_only",
    }
