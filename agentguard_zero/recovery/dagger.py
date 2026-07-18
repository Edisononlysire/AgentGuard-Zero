"""One-pass DAgger collection on model-visited public states."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Protocol, Sequence

from agentguard_zero.recovery.bootstrap_data import (
    BootstrapBuildResult,
    audit_bootstrap_records,
)
from agentguard_zero.recovery.model_policy import ModelDecision
from agentguard_zero.recovery.public_teacher import (
    PublicStateRobustTeacher,
    compact_wire_json,
    public_state_digest,
)
from agentguard_zero.training.vda_dataset import build_vda_prompt


class PublicModelPolicy(Protocol):
    def decide(self, public_prompt: str) -> ModelDecision: ...


def _scenario_hash(scenario: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(scenario),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def collect_dagger_records(
    scenario_groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    model_policy: PublicModelPolicy,
    teacher: PublicStateRobustTeacher | None = None,
    max_records: int = 3_000,
) -> BootstrapBuildResult:
    """Relabel only states visited by the selected frozen Gate-A policy."""

    from agentguard_zero.env.scenario_instantiator import instantiate_scenario

    if not scenario_groups or any(len(group) < 2 for group in scenario_groups):
        raise ValueError("DAgger requires matched counterfactual public groups")
    robust_teacher = teacher or PublicStateRobustTeacher()
    train_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    emitted: set[tuple[str, str]] = set()
    invalid_model_decisions = 0
    total_model_decisions = 0

    for group_index, scenarios in enumerate(scenario_groups):
        live = [instantiate_scenario(copy.deepcopy(dict(row))) for row in scenarios]
        decision_index = 0
        while live and len(train_records) < max_records:
            grouped: dict[str, list[Any]] = defaultdict(list)
            for env in live:
                grouped[public_state_digest(env.observe())].append(env)
            next_live: list[Any] = []
            for digest, worlds in sorted(grouped.items()):
                teacher_decision = robust_teacher.decide(
                    worlds,
                    horizon=3,
                    enforce_min_worlds=False,
                )
                prompt_scenario = copy.deepcopy(worlds[0].scenario)
                prompt_scenario["scenario_id"] = f"recovery-public-{digest[:20]}"
                prompt = build_vda_prompt(
                    prompt_scenario,
                    worlds[0].observe(),
                    experiment_variant="full",
                )
                model_decision = model_policy.decide(prompt)
                total_model_decisions += 1
                invalid_model_decisions += int(not model_decision.valid)
                target = compact_wire_json(teacher_decision.selected_packet)
                key = (digest, teacher_decision.selected_candidate_id)
                if key not in emitted:
                    record_id = hashlib.sha256(
                        (
                            f"dagger:{digest}:{teacher_decision.selected_candidate_id}:"
                            f"{group_index}:{decision_index}"
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
                            "public_state_digest": digest,
                            "action_category": teacher_decision.selected_category,
                            "source_policy": (
                                "finite_counterfactual_teacher_on_model_visited_state"
                            ),
                        }
                    )
                    audit = teacher_decision.to_audit_dict()
                    audit.update(
                        {
                            "record_id": record_id,
                            "initial_group_index": group_index,
                            "decision_index": decision_index,
                            "world_count": len(worlds),
                            "source_scenario_hashes": sorted(
                                _scenario_hash(env.scenario) for env in worlds
                            ),
                            "target_sha256": hashlib.sha256(
                                target.encode("utf-8")
                            ).hexdigest(),
                            "visited_model_action_category": (
                                model_decision.action_category
                            ),
                            "visited_model_action_valid": model_decision.valid,
                            "visited_model_output_sha256": hashlib.sha256(
                                model_decision.text.encode("utf-8")
                            ).hexdigest(),
                            "model_input_hidden_state": False,
                            "model_target_hidden_state": False,
                            "hidden_state_usage": "offline_robust_utility_only",
                        }
                    )
                    audit_records.append(audit)
                    emitted.add(key)
                decision_index += 1
                for env in worlds:
                    env.step(copy.deepcopy(model_decision.packet))
                    if not (
                        env.t >= env.max_steps
                        or env.attack_mitigated
                        or env.attack_success
                    ):
                        next_live.append(env)
            live = next_live

    manifest = audit_bootstrap_records(train_records, audit_records)
    manifest.update(
        {
            "kind": "single_dagger_correction_dataset",
            "scenario_count": sum(len(group) for group in scenario_groups),
            "initial_public_group_count": len(scenario_groups),
            "record_cap": max_records,
            "passes": 1,
            "human_action_labels": 0,
            "lineage": "new_recovery_lineage",
            "model_visited_decision_count": total_model_decisions,
            "model_visited_invalid_decision_count": invalid_model_decisions,
            "model_visited_action_validity": (
                1.0 - invalid_model_decisions / max(1, total_model_decisions)
            ),
        }
    )
    return BootstrapBuildResult(train_records, audit_records, manifest)
