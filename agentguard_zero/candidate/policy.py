"""Inference policy for direct candidate scoring."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agentguard_zero.candidate.compiler import CandidateCompiler
from agentguard_zero.candidate.generator import (
    ACTION_FAMILIES,
    CandidateGenerator,
    candidate_response_cost,
    candidate_is_state_legal,
    public_mitigation_count,
    public_response_anchors,
)
from agentguard_zero.candidate.branching import (
    BRANCH_NAMES,
    BUDGET_TARGET_NAMES,
    MEMORY_OPERATION_NAMES,
    PHASE_NAMES,
    PHASE_NAMES_V2,
    SHARED_LOCAL_LONG_ARCHITECTURE_V2,
    memory_operation_name,
    public_contract_branch,
)
from agentguard_zero.candidate.model import (
    activate_ranker_adapter,
    compose_candidate_utility,
    encode_candidate_pairs,
    encode_public_state,
    load_ranker_components,
    route_branched_experts,
    score_all_encoded,
)
from agentguard_zero.candidate.types import ActionFlags, CandidateOption
from agentguard_zero.schemas.action_schema_v4 import DEFAULT_ACTION_PACKET_V4


ACTIVE_PROBE_TOOLS = {
    "SourceChallenge",
    "CanaryProbe",
    "DecoyProbe",
    "ShadowActionProbe",
}
IMPACT_PROBE_TOOLS = {"BusinessImpactEstimator", "ShadowActionProbe"}
LOW_COST_PASSIVE_TOOLS = {"CrossCheck", "ProvenanceCheck", "GraphQuery"}
MITIGATING_ACTIONS = {
    "DeployDecoy",
    "LimitSession",
    "ShadowBlock",
    "Isolate",
    "Restore",
    "Remove",
}


def capability_chain_family(observation: Mapping[str, Any]) -> str | None:
    """Return the public prerequisite family, or Mitigation once it is met."""

    context = observation.get("defense_context") or {}
    requirements = context.get("response_requirements") or {}
    if not isinstance(requirements, Mapping) or not any(requirements.values()):
        return None
    defender = observation.get("defender_state") or {}
    evidence = observation.get("available_evidence") or []
    last = observation.get("last_tool_result") or {}
    response_history = defender.get("response_history") or []
    if any(
        isinstance(row, Mapping)
        and bool(row.get("authorized", False))
        and str(row.get("action", "")) in MITIGATING_ACTIONS
        for row in response_history
    ):
        # A completed mitigation is a terminal decision in the learned
        # capability chain even when the simulator keeps the episode open for
        # delayed outcome accounting.  Re-gating to mitigation here caused the
        # T4 policy to repeat responses until the business budget was exhausted.
        return "observe"

    tool_names = {str(last.get("tool", ""))}
    active_probe_evidence_ids: set[str] = set()
    if bool(last.get("active_probe")) and str(last.get("evidence_id", "")):
        active_probe_evidence_ids.add(str(last["evidence_id"]))
    for row in evidence:
        if not isinstance(row, Mapping):
            continue
        content = row.get("content") or {}
        if isinstance(content, Mapping):
            tool_name = str(content.get("tool", ""))
            tool_names.add(tool_name)
            if tool_name in ACTIVE_PROBE_TOOLS and str(row.get("evidence_id", "")):
                active_probe_evidence_ids.add(str(row["evidence_id"]))
    probe_state = defender.get("probe_state") or []
    active_probe_complete = bool(last.get("active_probe")) or bool(probe_state)
    active_probe_complete = active_probe_complete or bool(
        tool_names & ACTIVE_PROBE_TOOLS
    )
    impact_memory = defender.get("business_impact_memory") or {}
    impact_probe_complete = bool(tool_names & IMPACT_PROBE_TOOLS) or int(
        impact_memory.get("impact_estimate_count", 0) or 0
    ) > 0

    trust = defender.get("trust") or {}
    claim_rows = (trust.get("current_claim_trust") or {}).values()
    source_rows = (trust.get("source_reputation") or {}).values()
    trust_complete = any(
        str(row.get("status", "unassessed")) != "unassessed"
        for row in claim_rows
        if isinstance(row, Mapping)
    ) or any(
        str(row.get("status", "uncertain")) != "uncertain"
        for row in source_rows
        if isinstance(row, Mapping)
    )
    trust_evidence_ids = {
        str(reference)
        for row in claim_rows
        if isinstance(row, Mapping)
        for key in (
            "evidence_refs",
            "support_evidence_refs",
            "contradiction_evidence_refs",
        )
        for reference in row.get(key, []) or []
    }
    active_probe_grounded = bool(
        active_probe_evidence_ids & trust_evidence_ids
    )
    memory = defender.get("memory") or {}
    memory_ready = bool(memory.get("retrieved_memory_ids"))

    if requirements.get("passive_check_required_before_active_probe") and not (
        tool_names & LOW_COST_PASSIVE_TOOLS
    ):
        return "passive_verification"

    if requirements.get("impact_probe_required_for_mitigation"):
        if not impact_probe_complete:
            return "passive_verification"
        estimates = list(impact_memory.get("impact_estimates", []) or [])
        if estimates:
            last_estimate_time = max(
                int(row.get("time", -1))
                for row in estimates
                if isinstance(row, Mapping)
            )
            if int(observation.get("time", 0)) <= last_estimate_time + 1:
                return "observe"

    if requirements.get("profile_memory_use_required_for_mitigation"):
        profiles = list(memory.get("retrieved_profiles", []) or [])
        if not profiles:
            # Source profiles are created by a grounded source challenge.  An
            # Observe-only fallback cannot create one and therefore deadlocks
            # the T4 profile -> budget -> mitigation chain.
            return "active_probe"
        if not bool(tool_names & {"SourceChallenge", "CanaryProbe"}):
            return "active_probe"
        if not trust_complete:
            return "trust"
        return "mitigation"

    if requirements.get("active_probe_required_for_mitigation") and not active_probe_complete:
        return "active_probe"
    if (
        requirements.get("active_probe_required_for_mitigation")
        and active_probe_evidence_ids
        and not active_probe_grounded
    ):
        return "trust"
    if requirements.get("trust_update_required_for_mitigation") and not trust_complete:
        return "active_probe"
    if requirements.get("memory_use_required_for_mitigation"):
        if not memory_ready:
            risk_objects = {
                "high",
                "active",
                "compromised",
                "credential_theft",
                "exfiltration",
                "lateral_movement",
                "persistence",
            }
            observed_events = observation.get("observed_events", []) or []
            has_risk_claim = any(
                isinstance(row, Mapping)
                and str((row.get("claim_semantics") or {}).get("object", ""))
                in risk_objects
                for row in observed_events
            )
            return "memory" if has_risk_claim or not observed_events else "observe"
        records = [
            row
            for bucket in (
                "retrieved_confirmed",
                "retrieved_quarantined",
                "rejected_warnings",
            )
            for row in memory.get(bucket, []) or []
            if isinstance(row, Mapping)
        ]
        if not records:
            # Backward-compatible fallback for minimal public observations
            # that expose only retrieved_memory_ids.
            return "mitigation"
        if not any(str(row.get("status", "")) == "confirmed" for row in records):
            verification_rows = [
                row
                for row in evidence
                if isinstance(row, Mapping)
                and str((row.get("content") or {}).get("tool", ""))
                in {"CrossCheck", "ProvenanceCheck", "SourceChallenge", "CanaryProbe"}
            ]
            if not verification_rows:
                memory_claims = [
                    row.get("claim") or {}
                    for row in records
                    if isinstance(row.get("claim") or {}, Mapping)
                ]
                raw_roots = {
                    str(root)
                    for row in evidence
                    if isinstance(row, Mapping)
                    and ((row.get("content") or {}).get("claim_semantics") or {})
                    in memory_claims
                    for root in (
                        row.get("root_source_ids")
                        or [row.get("source_id", "")]
                    )
                    if str(root)
                }
                return "passive_verification" if len(raw_roots) >= 2 else "observe"
            if not trust_complete:
                return "trust"
            return "memory"
    return "mitigation"


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str | None
    semantic_id: str | None
    packet: dict[str, Any]
    valid: bool
    invalid_noop: bool
    reason: str
    action_flags: ActionFlags
    candidate_count: int
    score: float | None
    scores: dict[str, float]
    gated_family: str | None = None
    family_scores: dict[str, float] | None = None
    predicted_branch: str | None = None
    predicted_phase: str | None = None
    phase_probabilities: dict[str, float] | None = None
    memory_operation_probabilities: dict[str, float] | None = None
    budget_state: dict[str, float] | None = None
    operation_budget_masked_candidate_ids: tuple[str, ...] = ()


class CandidateRankerPolicy:
    def __init__(
        self,
        *,
        model_path: str | Path,
        adapter_path: str | Path,
        score_head_path: str | Path | None = None,
        heads_path: str | Path | None = None,
        device: str = "cuda:0",
        max_length: int = 2048,
        score_batch_size: int = 4,
        selection_mode: str = "candidate_argmax",
        score_composition: Mapping[str, Any] | None = None,
        generator: CandidateGenerator | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.score_batch_size = max(1, int(score_batch_size))
        if selection_mode not in {
            "candidate_argmax",
            "hierarchical_family_then_utility",
            "capability_chain_then_hierarchical",
        }:
            raise ValueError(f"unsupported candidate selection mode: {selection_mode}")
        self.selection_mode = selection_mode
        self.score_composition = dict(score_composition or {})
        self.architecture = str(
            self.score_composition.get("architecture", "legacy")
        )
        self.memory_tier_encoding = bool(
            self.score_composition.get("memory_tier_encoding", False)
        )
        self.generator = generator or CandidateGenerator()
        self.compiler = CandidateCompiler()
        self.tokenizer, self.backbone, self.heads = load_ranker_components(
            model_path=model_path,
            adapter_path=adapter_path,
            heads_path=heads_path,
            score_head_path=score_head_path,
            trainable=False,
            architecture=self.architecture,
        )
        self.backbone.to(self.device).eval()
        self.heads.to(self.device).eval()

    def _score(
        self, observation: Mapping[str, Any], candidates: list[CandidateOption]
    ) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
        values: list[float] = []
        v2_branch_name = (
            public_contract_branch(observation)
            if self.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
            else None
        )
        encode_memory_tiers = bool(
            self.memory_tier_encoding and v2_branch_name != "local"
        )
        with self.torch.inference_mode():
            if v2_branch_name is not None:
                activate_ranker_adapter(self.backbone, v2_branch_name)
            state_encoded = encode_public_state(
                self.tokenizer,
                observation,
                max_length=self.max_length,
                memory_tier_encoding=encode_memory_tiers,
            )
            state_encoded = {
                key: value.to(self.device) for key, value in state_encoded.items()
            }
            state_predictions = score_all_encoded(
                self.backbone,
                self.heads,
                input_ids=state_encoded["input_ids"],
                attention_mask=state_encoded["attention_mask"],
            )
            state_routes = state_predictions.get("branch_probabilities")
            if (
                self.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                and state_routes is not None
            ):
                state_routes = self.torch.zeros_like(state_routes)
                state_routes[:, BRANCH_NAMES.index(v2_branch_name)] = 1.0
                state_predictions.update(
                    route_branched_experts(
                        state_predictions,
                        state_routes,
                        use_legacy_local=True,
                    )
                )
            for offset in range(0, len(candidates), self.score_batch_size):
                chunk = candidates[offset : offset + self.score_batch_size]
                encoded = encode_candidate_pairs(
                    self.tokenizer,
                    observation,
                    chunk,
                    max_length=self.max_length,
                    memory_tier_encoding=encode_memory_tiers,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                predictions = score_all_encoded(
                    self.backbone,
                    self.heads,
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                if "branch_probabilities" in state_predictions:
                    candidate_routes = (
                        state_routes
                        if state_routes is not None
                        else state_predictions["branch_probabilities"]
                    ).expand(
                        len(chunk), -1
                    )
                    predictions.update(
                        route_branched_experts(
                            predictions,
                            candidate_routes,
                            use_legacy_local=(
                                self.architecture
                                == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                            ),
                        )
                    )
                decision_scores = compose_candidate_utility(
                    predictions, self.score_composition
                )
                if v2_branch_name == "t3_memory":
                    operation_log_probs = self.torch.log_softmax(
                        state_predictions["t3_memory_operation"][0], dim=-1
                    )
                    operation_weight = float(
                        self.score_composition.get(
                            "t3_memory_operation_weight", 1.0
                        )
                    )
                    operation_indices = self.torch.tensor(
                        [
                            MEMORY_OPERATION_NAMES.index(memory_operation_name(row))
                            for row in chunk
                        ],
                        dtype=self.torch.long,
                        device=decision_scores.device,
                    )
                    decision_scores = decision_scores + operation_weight * (
                        operation_log_probs[operation_indices]
                    )
                elif v2_branch_name == "t4_budget":
                    budget_logits = state_predictions["t4_budget_state"][0]
                    predicted_remaining_budget = self.torch.sigmoid(
                        budget_logits[
                            BUDGET_TARGET_NAMES.index("remaining_business_budget")
                        ]
                    )
                    ready_log_probability = self.torch.nn.functional.logsigmoid(
                        budget_logits[BUDGET_TARGET_NAMES.index("mitigation_ready")]
                    )
                    not_stopped_log_probability = self.torch.nn.functional.logsigmoid(
                        -budget_logits[BUDGET_TARGET_NAMES.index("stop_required")]
                    )
                    readiness_weight = float(
                        self.score_composition.get("t4_mitigation_ready_weight", 1.0)
                    )
                    readiness = ready_log_probability + not_stopped_log_probability
                    mitigation_mask = self.torch.tensor(
                        [float(row.action_flags.mitigation) for row in chunk],
                        dtype=decision_scores.dtype,
                        device=decision_scores.device,
                    )
                    decision_scores = (
                        decision_scores
                        + readiness_weight * mitigation_mask * readiness
                    )
                    affordability_weight = float(
                        self.score_composition.get(
                            "t4_remaining_budget_weight", 4.0
                        )
                    )
                    response_costs = self.torch.tensor(
                        [
                            candidate_response_cost(
                                observation, row.compiled_packet
                            )
                            for row in chunk
                        ],
                        dtype=decision_scores.dtype,
                        device=decision_scores.device,
                    )
                    decision_scores = decision_scores - (
                        affordability_weight
                        * mitigation_mask
                        * self.torch.relu(
                            response_costs - predicted_remaining_budget
                        )
                    )
                values.extend(
                    float(item)
                    for item in decision_scores.detach().cpu().tolist()
                )
        family_values = list(
            map(float, state_predictions["family"][0].detach().cpu().tolist())
        )
        support_values = list(
            map(float, state_predictions["support"][0].detach().cpu().tolist())
        )
        diagnostics: dict[str, Any] = {"predicted_branch": v2_branch_name}
        if "phase_gate" in state_predictions:
            phase_names = (
                PHASE_NAMES_V2
                if self.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else PHASE_NAMES
            )
            phase_logits = state_predictions["phase_gate"][0]
            allowed_phase_indices = list(range(len(phase_names)))
            if self.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2:
                prefixes = {
                    "local": ("t1:", "t2:"),
                    "t3_memory": ("t3:",),
                    "t4_budget": ("t4:",),
                }[str(v2_branch_name)]
                allowed_phase_indices = [
                    index
                    for index, name in enumerate(phase_names)
                    if name.startswith(prefixes)
                ]
            phase_values = [0.0] * len(phase_names)
            scoped_values = (
                self.torch.softmax(
                    phase_logits[allowed_phase_indices], dim=-1
                )
                .detach()
                .cpu()
                .tolist()
            )
            for index, value in zip(allowed_phase_indices, scoped_values):
                phase_values[index] = float(value)
            diagnostics["phase_probabilities"] = dict(
                zip(phase_names, phase_values)
            )
            diagnostics["predicted_phase"] = phase_names[
                max(allowed_phase_indices, key=phase_values.__getitem__)
            ]
        if "t3_memory_operation" in state_predictions:
            operation_values = list(
                map(
                    float,
                    self.torch.softmax(
                        state_predictions["t3_memory_operation"][0], dim=-1
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                )
            )
            diagnostics["memory_operation_probabilities"] = dict(
                zip(MEMORY_OPERATION_NAMES, operation_values)
            )
        if "t4_budget_state" in state_predictions:
            budget_values = list(
                map(
                    float,
                    self.torch.sigmoid(
                        state_predictions["t4_budget_state"][0]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                )
            )
            diagnostics["budget_state"] = dict(
                zip(BUDGET_TARGET_NAMES, budget_values)
            )
        return values, family_values, support_values, diagnostics

    def score_candidates(
        self,
        observation: Mapping[str, Any],
        candidates: list[CandidateOption],
    ) -> tuple[list[float], list[float], list[float]]:
        """Score a caller-frozen public candidate set without selecting an action."""

        values, families, support, _ = self._score(
            observation, list(candidates)
        )
        return values, families, support

    def decide(
        self,
        observation: Mapping[str, Any],
        *,
        sample: bool = False,
        temperature: float = 1.0,
        seed: int | None = None,
        candidates: list[CandidateOption] | None = None,
    ) -> CandidateDecision:
        try:
            candidates = (
                self.generator.generate(
                    observation,
                    permutation_seed=int(seed or 0),
                )
                if candidates is None
                else list(candidates)
            )
            if not candidates:
                raise RuntimeError("candidate set is empty")
            public_branch = (
                public_contract_branch(observation)
                if self.architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else None
            )
            if public_branch in {"t3_memory", "t4_budget"}:
                candidates = [
                    row
                    for row in candidates
                    if candidate_is_state_legal(observation, row.compiled_packet)
                ]
                if not candidates:
                    raise RuntimeError("public-state mask removed every candidate")
            values, gate_values, _, diagnostics = self._score(
                observation, candidates
            )
            if len(values) != len(candidates) or not all(
                self.torch.isfinite(self.torch.tensor(value)).item() for value in values
            ):
                raise RuntimeError("ranker produced missing or non-finite scores")
            if sample and temperature <= 0.0:
                raise ValueError("sampling temperature must be positive")
            eligible_indices = list(range(len(candidates)))
            if public_branch == "t3_memory":
                operation_probabilities = diagnostics.get(
                    "memory_operation_probabilities", {}
                )
                if operation_probabilities:
                    maximum = max(operation_probabilities.values())
                    relative_floor = float(
                        self.score_composition.get(
                            "t3_memory_operation_mask_relative_floor", 0.10
                        )
                    )
                    if relative_floor > 0.0:
                        operation_mask = [
                            index
                            for index in eligible_indices
                            if operation_probabilities.get(
                                memory_operation_name(candidates[index]), 0.0
                            )
                            + 1.0e-12
                            >= maximum * relative_floor
                        ]
                        eligible_indices = operation_mask or eligible_indices
            elif public_branch == "t4_budget":
                budget_state = diagnostics.get("budget_state", {})
                threshold = float(
                    self.score_composition.get("t4_binary_mask_threshold", 0.5)
                )
                hard_mask_enabled = bool(
                    self.score_composition.get("t4_binary_mask_enabled", True)
                )
                if hard_mask_enabled and budget_state.get("stop_required", 0.0) >= threshold:
                    stop = [
                        index
                        for index in eligible_indices
                        if candidates[index].action_flags.observe_only
                    ]
                    eligible_indices = stop or eligible_indices
                elif hard_mask_enabled and budget_state.get("mitigation_ready", 0.0) < threshold:
                    continue_chain = [
                        index
                        for index in eligible_indices
                        if not candidates[index].action_flags.mitigation
                    ]
                    eligible_indices = continue_chain or eligible_indices
                elif hard_mask_enabled:
                    predicted_remaining = float(
                        budget_state.get("remaining_business_budget", 0.0)
                    )
                    affordable = [
                        index
                        for index in eligible_indices
                        if not candidates[index].action_flags.mitigation
                        or candidate_response_cost(
                            observation, candidates[index].compiled_packet
                        )
                        <= predicted_remaining + 1.0e-9
                    ]
                    eligible_indices = affordable or eligible_indices
            masked_ids = tuple(
                candidate.candidate_id
                for index, candidate in enumerate(candidates)
                if index not in set(eligible_indices)
            )
            available_families = {
                family
                for family in ACTION_FAMILIES
                if any(
                    candidates[index].action_family == family
                    for index in eligible_indices
                )
            }
            gated_family: str | None = None
            candidate_indices = list(eligible_indices)
            if self.selection_mode in {
                "hierarchical_family_then_utility",
                "capability_chain_then_hierarchical",
            }:
                chain_family = (
                    capability_chain_family(observation)
                    if self.selection_mode == "capability_chain_then_hierarchical"
                    and public_branch != "local"
                    else None
                )
                available_indices = [
                    index
                    for index, family in enumerate(ACTION_FAMILIES)
                    if family in available_families
                ]
                if chain_family in available_families:
                    gated_family = chain_family
                elif sample:
                    family_probabilities = self.torch.softmax(
                        self.torch.tensor(
                            [gate_values[index] for index in available_indices],
                            dtype=self.torch.float32,
                        )
                        / temperature,
                        dim=0,
                    )
                    generator = self.torch.Generator(device="cpu")
                    if seed is not None:
                        generator.manual_seed(int(seed))
                    sampled = int(
                        self.torch.multinomial(
                            family_probabilities,
                            1,
                            replacement=True,
                            generator=generator,
                        ).item()
                    )
                    gated_family = ACTION_FAMILIES[available_indices[sampled]]
                else:
                    family_index = max(
                        available_indices,
                        key=lambda index: (gate_values[index], -index),
                    )
                    gated_family = ACTION_FAMILIES[family_index]
                candidate_indices = [
                    index
                    for index in eligible_indices
                    if candidates[index].action_family == gated_family
                ]
                if (
                    self.selection_mode == "capability_chain_then_hierarchical"
                    and public_branch != "local"
                ):
                    requirements = (
                        (observation.get("defense_context") or {}).get(
                            "response_requirements"
                        )
                        or {}
                    )
                    if (
                        gated_family == "active_probe"
                        and requirements.get("active_probe_required_for_mitigation")
                    ):
                        grounding_probes = [
                            index
                            for index in candidate_indices
                            if str(
                                (
                                    candidates[index].compiled_packet.get("tool_call")
                                    or {}
                                ).get("name", "")
                            )
                            in {"SourceChallenge", "CanaryProbe"}
                        ]
                        candidate_indices = grounding_probes or candidate_indices
                    if (
                        gated_family == "active_probe"
                        and requirements.get("trust_update_required_for_mitigation")
                    ):
                        trust_probes = [
                            index
                            for index in candidate_indices
                            if candidates[index].action_flags.trust
                            and any(
                                str(operation.get("op", "")) == "challenge"
                                for operation in candidates[index].compiled_packet.get(
                                    "trust_operations", []
                                )
                                if isinstance(operation, Mapping)
                            )
                        ]
                        candidate_indices = trust_probes or candidate_indices
                    if (
                        gated_family == "active_probe"
                        and requirements.get("impact_probe_required_for_mitigation")
                    ):
                        public_targets, _ = public_response_anchors(observation)
                        impact = [
                            index
                            for index in candidate_indices
                            if str(
                                (
                                    candidates[index].compiled_packet.get("tool_call")
                                    or {}
                                ).get("name", "")
                            )
                            == "ShadowActionProbe"
                            and str(
                                (
                                    (
                                        candidates[index].compiled_packet.get(
                                            "tool_call"
                                        )
                                        or {}
                                    ).get("args", {})
                                    or {}
                                ).get("action", {}).get("action", "")
                            )
                            == "ShadowBlock"
                            and (
                                not public_targets
                                or str(
                                    (
                                        (
                                            candidates[index].compiled_packet.get(
                                                "tool_call"
                                            )
                                            or {}
                                        ).get("args", {})
                                        or {}
                                    ).get("action", {}).get("target", "")
                                )
                                in public_targets
                            )
                        ]
                        candidate_indices = impact or candidate_indices
                    if gated_family == "trust":
                        probe_refs = {
                            str(row.get("evidence_id", ""))
                            for row in observation.get("available_evidence", []) or []
                            if isinstance(row, Mapping)
                            and str((row.get("content") or {}).get("tool", ""))
                            in ACTIVE_PROBE_TOOLS
                        }
                        grounded = [
                            index
                            for index in candidate_indices
                            if probe_refs
                            & set(candidates[index].referenced_ids)
                        ]
                        candidate_indices = grounded or candidate_indices
                    if gated_family == "mitigation":
                        public_targets, public_objectives = public_response_anchors(
                            observation
                        )
                        preferred_response = (
                            "DeployDecoy"
                            if requirements.get("memory_use_required_for_mitigation")
                            and public_mitigation_count(observation) >= 2
                            else "ShadowBlock"
                        )
                        safe_responses = []
                        for index in candidate_indices:
                            packet = candidates[index].compiled_packet
                            response = packet.get("response") or {}
                            belief = packet.get("belief") or {}
                            top_belief = max(belief, key=belief.get) if belief else ""
                            if (
                                str(response.get("action", "")) == preferred_response
                                and (
                                    not public_targets
                                    or str(response.get("target", "")) in public_targets
                                )
                                and (
                                    not public_objectives
                                    or top_belief in public_objectives
                                )
                            ):
                                safe_responses.append(index)
                        candidate_indices = safe_responses or candidate_indices
                    if (
                        gated_family == "mitigation"
                        and requirements.get("memory_use_required_for_mitigation")
                    ):
                        grounded = [
                            index
                            for index in candidate_indices
                            if candidates[index].action_flags.memory_use
                        ]
                        candidate_indices = grounded or candidate_indices
            if sample:
                generator = self.torch.Generator(device="cpu")
                if seed is not None:
                    generator.manual_seed(int(seed) + 1)
                probabilities = self.torch.softmax(
                    self.torch.tensor(
                        [values[index] for index in candidate_indices],
                        dtype=self.torch.float32,
                    )
                    / temperature,
                    dim=0,
                )
                sampled_index = int(
                    self.torch.multinomial(
                        probabilities, 1, replacement=True, generator=generator
                    ).item()
                )
                index = candidate_indices[sampled_index]
            else:
                index = max(
                    candidate_indices, key=lambda item: (values[item], -item)
                )
            selected = candidates[index]
            packet = self.compiler.compile(selected.candidate_id, candidates)
            return CandidateDecision(
                candidate_id=selected.candidate_id,
                semantic_id=selected.semantic_id,
                packet=packet,
                valid=True,
                invalid_noop=False,
                reason="ok",
                action_flags=selected.action_flags,
                candidate_count=len(candidates),
                score=values[index],
                scores={row.candidate_id: values[i] for i, row in enumerate(candidates)},
                gated_family=gated_family,
                family_scores={
                    family: gate_values[i] for i, family in enumerate(ACTION_FAMILIES)
                },
                predicted_branch=diagnostics.get("predicted_branch"),
                predicted_phase=diagnostics.get("predicted_phase"),
                phase_probabilities=diagnostics.get("phase_probabilities"),
                memory_operation_probabilities=diagnostics.get(
                    "memory_operation_probabilities"
                ),
                budget_state=diagnostics.get("budget_state"),
                operation_budget_masked_candidate_ids=masked_ids,
            )
        except Exception as exc:
            return CandidateDecision(
                candidate_id=None,
                semantic_id=None,
                packet=copy.deepcopy(DEFAULT_ACTION_PACKET_V4),
                valid=False,
                invalid_noop=True,
                reason=f"{type(exc).__name__}: {exc}",
                action_flags=ActionFlags(),
                candidate_count=0,
                score=None,
                scores={},
                gated_family=None,
                family_scores=None,
                predicted_branch=None,
                predicted_phase=None,
                phase_probabilities=None,
                memory_operation_probabilities=None,
                budget_state=None,
                operation_budget_masked_candidate_ids=(),
            )
