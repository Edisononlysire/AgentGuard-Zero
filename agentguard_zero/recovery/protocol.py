"""Frozen recovery constants proposed after the pure on-policy failure.

Only model-free Stage 0 and the repaired Bootstrap data audit are currently
execution-approved.  SFT, DAgger, Gate B, static-skill RL, and DCA-VDA
co-evolution remain review-locked even when review-only entrypoint code exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


RECOVERY_PROTOCOL_VERSION = "action-support-bootstrap-v1"

OLD_LINEAGE_DISPOSITION = {
    "qwen3.5_base": "bootstrap_initialization_candidate",
    "vda_1": "bootstrap_initialization_candidate_contract_stable_action_inert",
    "vda_2": "excluded_contract_collapse",
    "vda_3": "excluded_contract_collapse",
    "dca_1": "diagnostic_only",
    "dca_2": "excluded_invalid_feedback_lineage",
    "dca_3": "excluded_invalid_feedback_lineage",
    "ecrg_v5c": "negative_runtime_diagnostic_only",
    "old_three_round_data": "failure_analysis_only_not_training",
}


@dataclass(frozen=True)
class TeacherConfig:
    advantage_delta: float = 0.05
    min_worlds_per_public_state: int = 2
    common_horizon: int = 3
    beam_width: int = 20
    max_candidates: int = 96
    core_rank_correlation_min_exclusive: float = 0.50
    require_public_state_identity: bool = True
    require_all_world_admission: bool = True


@dataclass(frozen=True)
class Stage0Config:
    scenario_count: int = 200
    policies: tuple[str, ...] = (
        "no_op",
        "random_legal",
        "overreact",
        "public_state_teacher",
        "oracle",
    )
    teacher_noop_safe_utility_gap: float = 0.20
    teacher_attack_mitigation_min_exclusive: float = 0.40


@dataclass(frozen=True)
class BootstrapSFTConfig:
    pilot_scenarios_per_arm: int = 400
    pilot_records_min: int = 2_000
    pilot_records_max: int = 3_000
    epochs: int = 1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    learning_rate: float = 1.0e-5
    effective_batch_size: int = 64
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    initializations: tuple[str, ...] = ("qwen3.5_base", "vda_1")
    same_data_and_hyperparameters_required: bool = True
    unique_prompt_target_ratio_min: float = 0.95


@dataclass(frozen=True)
class GateAConfig:
    scenario_count: int = 200
    original_gate_scenarios: int = 80
    new_canonical_scenarios: int = 120
    candidate_count: int = 1
    decoding: str = "greedy"
    action_validity_min: float = 0.99
    actionable_observe_max: float = 0.70
    active_probe_min: float = 0.10
    attack_mitigation_min: float = 0.15
    safe_success_min: float = 0.10
    probe_yield_min_exclusive: float = 0.0
    trust_memory_operation_min: float = 0.05
    invalid_operation_max: float = 0.02
    hard_no_go_actionable_observe_exclusive: float = 0.80


@dataclass(frozen=True)
class DaggerConfig:
    canonical_scenarios: int = 500
    correction_records_min: int = 1_000
    correction_records_max: int = 3_000
    passes: int = 1


@dataclass(frozen=True)
class GateBConfig:
    scenario_count_min: int = 160
    scenario_count_max: int = 320
    rl_steps: int = 10
    initial_rollouts: int = 2
    adaptive_rollouts: int = 4
    temperature: float = 0.8
    top_p: float = 0.95
    bootstrap_replay_ratio: float = 0.20
    use_kl_loss: bool = True
    kl_coef: float = 0.02
    action_validity_min: float = 0.98
    actionable_observe_max: float = 0.80
    require_safe_utility_improvement: bool = True
    require_non_decreasing_mitigation: bool = True
    require_positive_probe_yield: bool = True


@dataclass(frozen=True)
class StaticSkillConfig:
    scenario_count: int = 800
    rl_steps_min: int = 25
    rl_steps_max: int = 40
    initial_rollouts: int = 2
    adaptive_rollouts: int = 4
    bootstrap_replay_ratio: float = 0.20
    kl_coef: float = 0.02
    nonzero_advantage_group_min: float = 0.50
    action_validity_min: float = 0.99
    safe_success_min: float = 0.15
    attack_mitigation_min: float = 0.20
    probe_yield_min: float = 0.10
    actionable_observe_max: float = 0.60
    trust_memory_operation_min: float = 0.10
    invalid_operation_max: float = 0.02
    nsu_gain_min: float = 0.03


@dataclass(frozen=True)
class CoevolutionConfig:
    enabled_before_gate_a_and_b: bool = False
    scenarios_per_round_min: int = 1_000
    scenarios_per_round_max: int = 1_200
    rollouts: tuple[int, ...] = (2, 4)
    replay_ratios: tuple[float, ...] = (0.20, 0.10, 0.05)
    counterfactual_weights: tuple[float, ...] = (0.10, 0.05, 0.0)
    curriculum_mix: tuple[dict[str, float], ...] = (
        {"easy": 0.50, "frontier": 0.40, "hard_reachable": 0.10},
        {"easy": 0.30, "frontier": 0.50, "hard_reachable": 0.20},
        {"easy": 0.20, "frontier": 0.50, "hard_reachable": 0.30},
    )
    exclude_zero_safe_success_probability: bool = True


@dataclass(frozen=True)
class RecoveryConfig:
    protocol_version: str = RECOVERY_PROTOCOL_VERSION
    zero_definition: str = "zero_human_labeled_optimal_defense_actions"
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    stage0: Stage0Config = field(default_factory=Stage0Config)
    bootstrap_sft: BootstrapSFTConfig = field(default_factory=BootstrapSFTConfig)
    gate_a: GateAConfig = field(default_factory=GateAConfig)
    dagger: DaggerConfig = field(default_factory=DaggerConfig)
    gate_b: GateBConfig = field(default_factory=GateBConfig)
    static_skill: StaticSkillConfig = field(default_factory=StaticSkillConfig)
    coevolution: CoevolutionConfig = field(default_factory=CoevolutionConfig)
    lineage_disposition: dict[str, str] = field(
        default_factory=lambda: dict(OLD_LINEAGE_DISPOSITION)
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
