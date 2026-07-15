from __future__ import annotations

from dataclasses import dataclass


ACTIVE_PROBE_TOOLS = frozenset(
    {"SourceChallenge", "CanaryProbe", "DecoyProbe", "ShadowActionProbe"}
)
PASSIVE_VERIFICATION_TOOLS = frozenset(
    {"LogQuery", "CrossCheck", "ProvenanceCheck", "GraphQuery"}
)


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    train_dca: bool = True
    frontier_filtering: bool = True
    active_probing: bool = True
    passive_verification: bool = True
    memory_lifecycle: bool = True
    trust_recalibration: bool = True
    business_aware_reward: bool = True
    state_layer: bool = True


_VARIANTS = {
    "full": ExperimentVariant("full"),
    "no_dca_training": ExperimentVariant("no_dca_training", train_dca=False),
    "no_frontier_filtering": ExperimentVariant(
        "no_frontier_filtering", frontier_filtering=False
    ),
    "no_active_probing": ExperimentVariant(
        "no_active_probing", active_probing=False
    ),
    "no_passive_verification": ExperimentVariant(
        "no_passive_verification", passive_verification=False
    ),
    "append_only_memory": ExperimentVariant(
        "append_only_memory", memory_lifecycle=False
    ),
    "no_trust_recalibration": ExperimentVariant(
        "no_trust_recalibration", trust_recalibration=False
    ),
    "no_business_aware_reward": ExperimentVariant(
        "no_business_aware_reward", business_aware_reward=False
    ),
    # Same-policy deployment control used by RQ6, not a co-evolution ablation.
    "no_state_layer": ExperimentVariant("no_state_layer", state_layer=False),
}

TRAINING_VARIANTS = tuple(name for name in _VARIANTS if name != "no_state_layer")
ALL_VARIANTS = tuple(_VARIANTS)


def experiment_variant(name: str) -> ExperimentVariant:
    normalized = str(name or "full")
    try:
        return _VARIANTS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported experiment variant: {normalized}") from exc

