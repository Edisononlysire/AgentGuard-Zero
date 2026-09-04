"""Public-only adaptive composition of three frozen candidate rankers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from agentguard_zero.candidate.generator import (
    ACTION_FAMILIES,
    CandidateGenerator,
    candidate_is_state_legal,
)
from agentguard_zero.candidate.policy import CandidateDecision, CandidateRankerPolicy
from agentguard_zero.candidate.types import ActionFlags, CandidateOption
from agentguard_zero.governance.authorization import authorize_public_response
from agentguard_zero.schemas.action_schema_v4 import DEFAULT_ACTION_PACKET_V4
from agentguard_zero.training.coevolution import sha256_file, sha256_tree
from agentguard_zero.world.public_projector import assert_public, project_public


EXPERT_NAMES = ("t12_response", "t3_memory", "t4_budget")
FAMILY_NAMES = tuple(ACTION_FAMILIES)
ROUTER_OUTPUT_DIM = len(EXPERT_NAMES) * len(FAMILY_NAMES)
MITIGATING_ACTIONS = {
    "DeployDecoy",
    "LimitSession",
    "ShadowBlock",
    "Isolate",
    "Restore",
    "Remove",
}
ACTIVE_PROBE_TOOLS = {
    "SourceChallenge",
    "CanaryProbe",
    "DecoyProbe",
    "ShadowActionProbe",
}
PASSIVE_TOOLS = {"LogQuery", "CrossCheck", "ProvenanceCheck", "GraphQuery"}


PUBLIC_FEATURE_NAMES = (
    "time",
    "max_steps",
    "time_fraction",
    "remaining_business_budget",
    "remaining_high_impact_actions",
    "remaining_verification_budget",
    "remaining_active_probe_budget",
    "requirement_count",
    "require_passive",
    "require_active_probe",
    "require_trust",
    "require_memory",
    "require_profile",
    "require_impact",
    "observed_event_count",
    "available_evidence_count",
    "last_tool_active_probe",
    "last_tool_passive_verification",
    "trust_supported_count",
    "trust_challenged_count",
    "trust_rejected_count",
    "trust_unassessed_count",
    "memory_confirmed_count",
    "memory_quarantined_count",
    "memory_rejected_count",
    "memory_other_count",
    "probe_pending_count",
    "probe_resolved_count",
    "response_history_count",
    "authorized_mitigation_count",
    "unauthorized_high_impact_count",
    *tuple(f"candidate_family_available:{family}" for family in FAMILY_NAMES),
)
EXPERT_STAT_FEATURE_NAMES = tuple(
    f"{expert}:{stat}"
    for expert in EXPERT_NAMES
    for stat in ("top_score", "margin", "entropy")
)
ROUTER_FEATURE_NAMES = (*PUBLIC_FEATURE_NAMES, *EXPERT_STAT_FEATURE_NAMES)


def _walk_mappings(value: Any) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for item in value.values():
            rows.extend(_walk_mappings(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_walk_mappings(item))
    return rows


def _status_counts(value: Any, statuses: Sequence[str]) -> dict[str, int]:
    counts = {status: 0 for status in statuses}
    for row in _walk_mappings(value):
        status = str(row.get("status", "")).lower()
        if status in counts:
            counts[status] += 1
    return counts


def _remaining(context: Mapping[str, Any], key: str) -> float:
    value = context.get(key)
    if value is None and key.startswith("remaining_"):
        value = context.get(key.removeprefix("remaining_"))
    try:
        return float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0


def public_router_features(
    observation: Mapping[str, Any],
    candidates: Sequence[CandidateOption],
) -> list[float]:
    """Extract the frozen numeric Router contract from public state only."""

    public = project_public(dict(observation))
    assert_public(public)
    context = public.get("defense_context") or {}
    defender = public.get("defender_state") or {}
    requirements = context.get("response_requirements") or {}
    time_value = float(public.get("time", public.get("t", 0)) or 0)
    max_steps = float(public.get("max_steps", context.get("max_steps", 0)) or 0)
    requirement_keys = [
        str(key).lower() for key, enabled in requirements.items() if bool(enabled)
    ]
    last_tool = str((public.get("last_tool_result") or {}).get("tool", ""))
    trust = _status_counts(
        defender.get("trust") or {},
        ("supported", "challenged", "rejected", "unassessed"),
    )
    memory = _status_counts(
        defender.get("memory") or {},
        ("confirmed", "quarantined", "rejected"),
    )
    memory_rows = _walk_mappings(defender.get("memory") or {})
    memory_other = sum(
        1
        for row in memory_rows
        if str(row.get("memory_id", ""))
        and str(row.get("status", "")).lower()
        not in {"confirmed", "quarantined", "rejected"}
    )
    probe_rows = _walk_mappings(defender.get("probe_state") or [])
    probe_pending = sum(
        str(row.get("status", "")).lower() == "pending" for row in probe_rows
    )
    probe_resolved = sum(
        str(row.get("status", "")).lower() == "resolved" for row in probe_rows
    )
    history = [
        row
        for row in defender.get("response_history", []) or []
        if isinstance(row, Mapping)
    ]
    authorized_mitigation = sum(
        bool(row.get("authorized", False))
        and str(row.get("action", "")) in MITIGATING_ACTIONS
        for row in history
    )
    unauthorized_high = sum(
        not bool(row.get("authorized", False))
        and str(row.get("action", "")) in {"Isolate", "Restore", "Remove"}
        for row in history
    )
    available_families = {candidate.action_family for candidate in candidates}
    features = [
        time_value,
        max_steps,
        time_value / max(1.0, max_steps),
        _remaining(context, "remaining_business_budget"),
        _remaining(context, "remaining_high_impact_actions"),
        _remaining(context, "remaining_verification_budget"),
        _remaining(context, "remaining_active_probe_budget"),
        float(len(requirement_keys)),
        float(any("passive" in key for key in requirement_keys)),
        float(any("active_probe" in key or key == "probe" for key in requirement_keys)),
        float(any("trust" in key for key in requirement_keys)),
        float(any("memory" in key for key in requirement_keys)),
        float(any("profile" in key for key in requirement_keys)),
        float(any("impact" in key or "budget" in key for key in requirement_keys)),
        float(len(public.get("observed_events", []) or [])),
        float(len(public.get("available_evidence", []) or [])),
        float(last_tool in ACTIVE_PROBE_TOOLS),
        float(last_tool in PASSIVE_TOOLS),
        float(trust["supported"]),
        float(trust["challenged"]),
        float(trust["rejected"]),
        float(trust["unassessed"]),
        float(memory["confirmed"]),
        float(memory["quarantined"]),
        float(memory["rejected"]),
        float(memory_other),
        float(probe_pending),
        float(probe_resolved),
        float(len(history)),
        float(authorized_mitigation),
        float(unauthorized_high),
        *[
            float(family in available_families)
            for family in FAMILY_NAMES
        ],
    ]
    if len(features) != len(PUBLIC_FEATURE_NAMES):
        raise RuntimeError("public Router feature contract length mismatch")
    return features


def router_candidate_is_state_legal(
    observation: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> bool:
    """Apply the shared public pre-state legality and authorization contract."""

    if not candidate_is_state_legal(observation, packet):
        return False
    public = project_public(dict(observation))
    assert_public(public)
    context = public.get("defense_context") or {}
    defender = public.get("defender_state") or {}
    response = packet.get("response") or {}
    action = str(response.get("action", "Observe"))
    authorization = authorize_public_response(
        response=dict(response),
        evidence_assessment=list(packet.get("evidence_assessment", []) or []),
        trust_snapshot=dict(defender.get("trust") or {}),
        evidence_snapshot=list(public.get("available_evidence", []) or []),
        public_assets=list(context.get("public_assets", []) or []),
        business_state={
            "remaining_business_budget": _remaining(
                context, "remaining_business_budget"
            ),
            "remaining_high_impact_actions": int(
                _remaining(context, "remaining_high_impact_actions")
            ),
        },
        oracle_override=False,
    )
    if not authorization.allowed:
        return False
    requirements = context.get("response_requirements") or {}
    if not isinstance(requirements, Mapping):
        return False
    evidence = [
        row
        for row in public.get("available_evidence", []) or []
        if isinstance(row, Mapping)
    ]
    last = public.get("last_tool_result") or {}
    observed_tools = {str(last.get("tool", ""))}
    observed_tools.update(
        str((row.get("content") or {}).get("tool", ""))
        for row in evidence
        if isinstance(row.get("content") or {}, Mapping)
    )
    probe_state = defender.get("probe_state") or []
    if (
        action in MITIGATING_ACTIONS
        and requirements.get("active_probe_required_for_mitigation")
        and not (
            bool(last.get("active_probe", False))
            or bool(probe_state)
            or bool(observed_tools & ACTIVE_PROBE_TOOLS)
        )
    ):
        return False
    if (
        action in MITIGATING_ACTIONS
        and requirements.get("trust_update_required_for_mitigation")
    ):
        claims = (
            ((defender.get("trust") or {}).get("current_claim_trust") or {}).values()
        )
        if not any(
            str(row.get("status", "unassessed")) != "unassessed"
            for row in claims
            if isinstance(row, Mapping)
        ):
            return False
    response_usage = [
        row
        for row in packet.get("memory_usage", []) or []
        if isinstance(row, Mapping) and str(row.get("used_for", "")) == "response"
    ]
    if (
        action in MITIGATING_ACTIONS
        and requirements.get("memory_use_required_for_mitigation")
        and not response_usage
    ):
        return False
    if (
        action in MITIGATING_ACTIONS
        and requirements.get("profile_memory_use_required_for_mitigation")
        and not any(
            str(row.get("memory_id", "")).startswith("profile:")
            for row in response_usage
        )
    ):
        return False
    if (
        action in MITIGATING_ACTIONS
        and requirements.get("impact_probe_required_for_mitigation")
        and not bool(observed_tools & {"BusinessImpactEstimator", "ShadowActionProbe"})
    ):
        return False
    tool_name = str((packet.get("tool_call") or {}).get("name", "None"))
    if (
        tool_name in ACTIVE_PROBE_TOOLS
        and requirements.get("passive_check_required_before_active_probe")
        and not bool(observed_tools & PASSIVE_TOOLS)
    ):
        return False
    return True


class PublicGatedExpertPolicy:
    """Expose a frozen expert through the same public legality gate as Router."""

    def __init__(self, policy: CandidateRankerPolicy) -> None:
        self.policy = policy
        self.generator = CandidateGenerator()
        self.selection_mode = policy.selection_mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def score_candidates(
        self,
        observation: Mapping[str, Any],
        candidates: Sequence[CandidateOption],
    ) -> tuple[list[float], list[float], list[float]]:
        return self.policy.score_candidates(dict(observation), list(candidates))

    def decide(
        self,
        observation: Mapping[str, Any],
        *,
        sample: bool = False,
        temperature: float = 1.0,
        seed: int | None = None,
        candidates: list[CandidateOption] | None = None,
    ) -> CandidateDecision:
        all_candidates = (
            self.generator.generate(observation, permutation_seed=int(seed or 0))
            if candidates is None
            else list(candidates)
        )
        legal = [
            candidate
            for candidate in all_candidates
            if router_candidate_is_state_legal(
                observation, candidate.compiled_packet
            )
        ]
        if not legal:
            return CandidateDecision(
                candidate_id=None,
                semantic_id=None,
                packet=dict(DEFAULT_ACTION_PACKET_V4),
                valid=False,
                invalid_noop=True,
                reason="shared_public_legality_empty_noop",
                action_flags=ActionFlags(),
                candidate_count=len(all_candidates),
                score=None,
                scores={},
            )
        decision = self.policy.decide(
            dict(observation),
            sample=sample,
            temperature=temperature,
            seed=seed,
            candidates=legal,
        )
        return replace(
            decision,
            candidate_count=len(all_candidates),
            reason=f"shared_public_legality:{decision.reason}",
        )


@dataclass(frozen=True)
class ScoreCalibration:
    means: tuple[tuple[float, ...], ...]
    scales: tuple[tuple[float, ...], ...]
    temperatures: tuple[tuple[float, ...], ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScoreCalibration":
        def matrix(name: str) -> tuple[tuple[float, ...], ...]:
            value = tuple(
                tuple(map(float, row)) for row in payload.get(name, [])
            )
            if len(value) != len(EXPERT_NAMES) or any(
                len(row) != len(FAMILY_NAMES) for row in value
            ):
                raise ValueError(f"invalid calibration matrix: {name}")
            return value

        result = cls(
            means=matrix("means"),
            scales=matrix("scales"),
            temperatures=matrix("temperatures"),
        )
        if any(
            scale <= 0.0 or temperature <= 0.0
            for expert in range(len(EXPERT_NAMES))
            for scale, temperature in zip(
                result.scales[expert],
                result.temperatures[expert],
                strict=True,
            )
        ):
            raise ValueError("calibration scales and temperatures must be positive")
        return result

    def score(self, expert: int, family: int, raw: float) -> float:
        logit = (
            (float(raw) - self.means[expert][family])
            / self.scales[expert][family]
            / self.temperatures[expert][family]
        )
        # Router weights are non-negative mixture weights.  Mapping calibrated
        # logits to (0, 1) prevents a near-zero route weight from making a
        # very negative candidate look artificially better than a strongly
        # weighted negative candidate.
        if logit >= 0.0:
            inverse = math.exp(-min(logit, 60.0))
            return 1.0 / (1.0 + inverse)
        exponent = math.exp(max(logit, -60.0))
        return exponent / (1.0 + exponent)


def calibrated_score_rows(
    candidates: Sequence[CandidateOption],
    raw_scores: Sequence[Sequence[float]],
    calibration: ScoreCalibration,
) -> list[list[float]]:
    if len(raw_scores) != len(EXPERT_NAMES):
        raise ValueError("exactly three expert score rows are required")
    rows: list[list[float]] = []
    for expert_index, scores in enumerate(raw_scores):
        if len(scores) != len(candidates):
            raise ValueError("expert/candidate score length mismatch")
        rows.append(
            [
                calibration.score(
                    expert_index,
                    FAMILY_NAMES.index(candidate.action_family),
                    float(score),
                )
                for candidate, score in zip(candidates, scores, strict=True)
            ]
        )
    return rows


def _score_statistics(scores: Sequence[float]) -> tuple[float, float, float]:
    finite = [float(score) for score in scores if math.isfinite(float(score))]
    if not finite:
        return 0.0, 0.0, 0.0
    ordered = sorted(finite, reverse=True)
    top = ordered[0]
    margin = top - ordered[1] if len(ordered) > 1 else 0.0
    maximum = max(finite)
    exponentials = [math.exp(min(50.0, value - maximum)) for value in finite]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    entropy = -sum(
        probability * math.log(max(probability, 1.0e-12))
        for probability in probabilities
    )
    return top, margin, entropy


def router_feature_vector(
    observation: Mapping[str, Any],
    candidates: Sequence[CandidateOption],
    calibrated_scores: Sequence[Sequence[float]],
) -> list[float]:
    result = public_router_features(observation, candidates)
    for scores in calibrated_scores:
        result.extend(_score_statistics(scores))
    if len(result) != len(ROUTER_FEATURE_NAMES):
        raise RuntimeError("Router feature vector length mismatch")
    return result


class ExpertRouterMLP:
    """Factory wrapper so importing this module does not require torch."""

    @staticmethod
    def build(input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> Any:
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, ROUTER_OUTPUT_DIM),
        )


def validate_expert_checkpoint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "candidate_ranker_checkpoint":
        raise RuntimeError(f"not a candidate ranker checkpoint: {path}")
    adapter = Path(str(payload.get("adapter_path", "")))
    head = Path(str(payload.get("heads_path") or payload.get("score_head_path", "")))
    if sha256_tree(adapter) != payload.get("adapter_sha256"):
        raise RuntimeError(f"expert adapter hash mismatch: {path}")
    if sha256_file(head) != (
        payload.get("heads_sha256") or payload.get("score_head_sha256")
    ):
        raise RuntimeError(f"expert heads hash mismatch: {path}")
    return payload


def load_expert_policy(
    *,
    model_path: str | Path,
    manifest_path: Path,
    device: str,
    max_length: int = 2048,
    score_batch_size: int = 4,
) -> PublicGatedExpertPolicy:
    manifest = validate_expert_checkpoint(manifest_path)
    score_composition = dict(manifest.get("score_composition") or {})
    score_composition.setdefault(
        "architecture", str(manifest.get("architecture") or "legacy")
    )
    score_composition.setdefault(
        "memory_tier_encoding", bool(manifest.get("memory_tier_encoding", False))
    )
    # Expert continuations use the same learned hierarchical interface.  The
    # T4 checkpoint's historical hand-coded capability-chain selector is not
    # part of the unified policy.
    return PublicGatedExpertPolicy(
        CandidateRankerPolicy(
            model_path=model_path,
            adapter_path=manifest["adapter_path"],
            heads_path=manifest.get("heads_path"),
            score_head_path=(
                None
                if manifest.get("heads_path")
                else manifest.get("score_head_path")
            ),
            device=device,
            max_length=max_length,
            score_batch_size=score_batch_size,
            selection_mode="hierarchical_family_then_utility",
            score_composition=score_composition,
        )
    )


class UnifiedExpertRouterPolicy:
    def __init__(
        self,
        *,
        model_path: str | Path,
        expert_manifests: Sequence[Path],
        router_manifest: Path,
        device: str = "cuda:0",
        max_length: int = 2048,
        score_batch_size: int = 4,
        routing_mode: str = "learned",
    ) -> None:
        import torch

        if len(expert_manifests) != len(EXPERT_NAMES):
            raise ValueError("exactly three expert manifests are required")
        if routing_mode not in {"learned", "equal", "public_rule"}:
            raise ValueError(f"unsupported Router comparison mode: {routing_mode}")
        self.torch = torch
        self.device = torch.device(device)
        self.routing_mode = routing_mode
        self.generator = CandidateGenerator()
        self.expert_manifests = tuple(expert_manifests)
        self.experts = tuple(
            load_expert_policy(
                model_path=model_path,
                manifest_path=manifest,
                device=device,
                max_length=max_length,
                score_batch_size=score_batch_size,
            )
            for manifest in expert_manifests
        )
        manifest = json.loads(router_manifest.read_text(encoding="utf-8"))
        if (
            manifest.get("kind") != "v11_capability_router_checkpoint"
            or manifest.get("accepted") is not True
            or manifest.get("formal_frozen_600_inspected") is not False
        ):
            raise RuntimeError("not a V11 Router checkpoint")
        if manifest.get("feature_names") != list(ROUTER_FEATURE_NAMES):
            raise RuntimeError("Router feature contract mismatch")
        if manifest.get("expert_names") != list(EXPERT_NAMES):
            raise RuntimeError("Router expert ordering mismatch")
        if manifest.get("family_names") != list(FAMILY_NAMES):
            raise RuntimeError("Router family ordering mismatch")
        self.feature_means = tuple(map(float, manifest.get("feature_means", [])))
        self.feature_scales = tuple(map(float, manifest.get("feature_scales", [])))
        if (
            len(self.feature_means) != len(ROUTER_FEATURE_NAMES)
            or len(self.feature_scales) != len(ROUTER_FEATURE_NAMES)
            or any(scale <= 0.0 for scale in self.feature_scales)
        ):
            raise RuntimeError("Router feature normalization contract mismatch")
        expected_expert_hashes = [
            sha256_file(path) for path in expert_manifests
        ]
        if manifest.get("expert_manifest_sha256") != expected_expert_hashes:
            raise RuntimeError("Router expert-bundle binding mismatch")
        if (
            manifest.get("public_legality_gate")
            != "router_candidate_is_state_legal_v1"
            or manifest.get("public_legality_gate_source_sha256")
            != sha256_file(Path(__file__).resolve())
        ):
            raise RuntimeError("Router public-legality gate source mismatch")
        calibration_path = Path(manifest["calibration_path"])
        if sha256_file(calibration_path) != manifest["calibration_sha256"]:
            raise RuntimeError("Router calibration hash mismatch")
        self.calibration = ScoreCalibration.from_payload(
            json.loads(calibration_path.read_text(encoding="utf-8"))
        )
        self.router = ExpertRouterMLP.build(
            len(ROUTER_FEATURE_NAMES),
            hidden_dim=int(manifest["hidden_dim"]),
            dropout=float(manifest["dropout"]),
        )
        state_path = Path(manifest["router_state_path"])
        if sha256_file(state_path) != manifest["router_state_sha256"]:
            raise RuntimeError("Router state hash mismatch")
        self.router.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=True)
        )
        self.router.to(self.device).eval()
        self.last_route_weights: dict[str, dict[str, float]] | None = None
        self.last_expert_scores: dict[str, dict[str, float]] | None = None

    def _public_rule_expert(self, observation: Mapping[str, Any]) -> int:
        public = project_public(dict(observation))
        assert_public(public)
        context = public.get("defense_context") or {}
        defender = public.get("defender_state") or {}
        requirements = context.get("response_requirements") or {}
        enabled = {
            str(key).lower()
            for key, value in requirements.items()
            if bool(value)
        }
        memory = defender.get("memory") or {}
        memory_visible = any(
            bool(memory.get(bucket))
            for bucket in (
                "retrieved_confirmed",
                "retrieved_quarantined",
                "rejected_warnings",
                "retrieved_profiles",
            )
        )
        if memory_visible or any("memory" in key for key in enabled):
            return EXPERT_NAMES.index("t3_memory")
        remaining_business = _remaining(context, "remaining_business_budget")
        remaining_high = _remaining(context, "remaining_high_impact_actions")
        if (
            remaining_business <= 2.0
            or remaining_high <= 0.0
            or any(
                "impact" in key or "budget" in key or "profile" in key
                for key in enabled
            )
        ):
            return EXPERT_NAMES.index("t4_budget")
        return EXPERT_NAMES.index("t12_response")

    def _route_weight_matrix(
        self,
        observation: Mapping[str, Any],
        normalized_features: Sequence[float],
    ) -> list[list[float]]:
        if self.routing_mode == "equal":
            value = 1.0 / ROUTER_OUTPUT_DIM
            return [
                [value for _ in FAMILY_NAMES]
                for _ in EXPERT_NAMES
            ]
        if self.routing_mode == "public_rule":
            selected = self._public_rule_expert(observation)
            return [
                [
                    1.0 / len(FAMILY_NAMES) if expert_index == selected else 0.0
                    for _ in FAMILY_NAMES
                ]
                for expert_index in range(len(EXPERT_NAMES))
            ]
        with self.torch.inference_mode():
            tensor = self.torch.tensor(
                [normalized_features],
                dtype=self.torch.float32,
                device=self.device,
            )
            logits = self.router(tensor)[0]
            # The supervision target is one expert×family cell.  A single
            # 18-way softmax therefore preserves both decisions learned by
            # the Router: which capability expert to trust and which action
            # family should receive mass.
            return (
                self.torch.softmax(logits, dim=0)
                .reshape(len(EXPERT_NAMES), len(FAMILY_NAMES))
                .detach()
                .cpu()
                .tolist()
            )

    def _raw_scores(
        self,
        observation: Mapping[str, Any],
        candidates: Sequence[CandidateOption],
    ) -> list[list[float]]:
        return [
            expert.score_candidates(dict(observation), list(candidates))[0]
            for expert in self.experts
        ]

    def decide(
        self,
        observation: Mapping[str, Any],
        *,
        sample: bool = False,
        temperature: float = 1.0,
        seed: int | None = None,
        candidates: list[CandidateOption] | None = None,
    ) -> CandidateDecision:
        del sample, temperature
        candidates = (
            self.generator.generate(observation, permutation_seed=int(seed or 0))
            if candidates is None
            else list(candidates)
        )
        legal = [
            router_candidate_is_state_legal(observation, candidate.compiled_packet)
            for candidate in candidates
        ]
        if not candidates or not any(legal):
            return CandidateDecision(
                candidate_id=None,
                semantic_id=None,
                packet=dict(DEFAULT_ACTION_PACKET_V4),
                valid=False,
                invalid_noop=True,
                reason="router_public_legality_empty_noop",
                action_flags=ActionFlags(),
                candidate_count=len(candidates),
                score=None,
                scores={},
            )
        raw = self._raw_scores(observation, candidates)
        calibrated = calibrated_score_rows(candidates, raw, self.calibration)
        feature_candidates = [
            candidate for candidate, admitted in zip(candidates, legal, strict=True)
            if admitted
        ]
        feature_scores = [
            [
                score
                for score, admitted in zip(scores, legal, strict=True)
                if admitted
            ]
            for scores in calibrated
        ]
        features = router_feature_vector(
            observation, feature_candidates, feature_scores
        )
        features = [
            (value - mean) / scale
            for value, mean, scale in zip(
                features,
                self.feature_means,
                self.feature_scales,
                strict=True,
            )
        ]
        weights = self._route_weight_matrix(observation, features)
        final_scores: list[float] = []
        for candidate_index, candidate in enumerate(candidates):
            family_index = FAMILY_NAMES.index(candidate.action_family)
            score = sum(
                float(weights[expert_index][family_index])
                * float(calibrated[expert_index][candidate_index])
                for expert_index in range(len(EXPERT_NAMES))
            )
            final_scores.append(score if legal[candidate_index] else float("-inf"))
        selected_index = max(
            range(len(candidates)),
            key=lambda index: (final_scores[index], candidates[index].semantic_id),
        )
        selected = candidates[selected_index]
        self.last_route_weights = {
            expert: {
                family: float(weights[expert_index][family_index])
                for family_index, family in enumerate(FAMILY_NAMES)
            }
            for expert_index, expert in enumerate(EXPERT_NAMES)
        }
        self.last_expert_scores = {
            expert: {
                candidate.candidate_id: float(calibrated[expert_index][candidate_index])
                for candidate_index, candidate in enumerate(candidates)
            }
            for expert_index, expert in enumerate(EXPERT_NAMES)
        }
        return CandidateDecision(
            candidate_id=selected.candidate_id,
            semantic_id=selected.semantic_id,
            packet=selected.compiled_packet,
            valid=True,
            invalid_noop=False,
            reason=f"{self.routing_mode}_capability_router_score_fusion",
            action_flags=selected.action_flags,
            candidate_count=len(candidates),
            score=float(final_scores[selected_index]),
            scores={
                candidate.candidate_id: float(score)
                for candidate, score in zip(candidates, final_scores, strict=True)
            },
            gated_family=selected.action_family,
            family_scores={
                family: max(
                    (
                        final_scores[index]
                        for index, candidate in enumerate(candidates)
                        if candidate.action_family == family
                    ),
                    default=float("-inf"),
                )
                for family in FAMILY_NAMES
            },
        )


def expert_outcome_key(outcome: Mapping[str, Any]) -> tuple[float, ...]:
    """Lower tuple is better; exactly follows the frozen terminal lexicography."""

    return (
        -float(outcome.get("safe_success", 0.0)),
        -float(outcome.get("attack_mitigation", 0.0)),
        float(outcome.get("attack_success", 0.0)),
        -float(outcome.get("intent_accuracy", 0.0)),
        float(outcome.get("overresponse", 0.0)),
        float(outcome.get("unauthorized_high_impact", 0.0)),
        float(outcome.get("business_cost", 0.0)),
    )
