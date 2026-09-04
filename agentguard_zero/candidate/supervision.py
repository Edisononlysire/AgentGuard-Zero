"""Policy-consistent supervision for public-state candidate ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PolicySupervision:
    probabilities: tuple[float, ...]
    policy_scores: tuple[float, ...]
    eligible: tuple[bool, ...]
    retained: tuple[bool, ...]
    acceptable: tuple[bool, ...]


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scaled = [float(value) / float(temperature) for value in values]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


def policy_consistent_supervision(
    q_values: Sequence[float],
    core_q_values: Sequence[float],
    *,
    target_index: int,
    observe_index: int,
    core_tolerance: float,
    temperature: float,
    target_weight: float = 0.7,
    top_k: int = 4,
    policy_margin: float = 0.2,
    acceptable_tolerance: float = 0.02,
) -> PolicySupervision:
    """Build labels that preserve the Teacher's core-first final decision.

    Raw robust Q can prefer a candidate that the Teacher rejects on core safety,
    or prefer an active candidate by less than the Observe advantage threshold.
    Those raw values are useful diagnostics, but must not define the policy loss.
    """

    count = len(q_values)
    if count == 0 or len(core_q_values) != count:
        raise ValueError("Q arrays must have the same non-zero length")
    if not 0 <= target_index < count or not 0 <= observe_index < count:
        raise IndexError("target or Observe index is out of range")
    if not 0.0 <= target_weight <= 1.0:
        raise ValueError("target_weight must be in [0, 1]")

    observe_core = float(core_q_values[observe_index])
    eligible = [
        float(value) >= observe_core - float(core_tolerance)
        for value in core_q_values
    ]
    if not eligible[target_index]:
        raise ValueError("Teacher target is not core-eligible")

    eligible_indices = [index for index, value in enumerate(eligible) if value]
    ranked = sorted(
        eligible_indices,
        key=lambda index: (-float(q_values[index]), index),
    )
    retained_indices = set(ranked[: max(1, int(top_k))])
    retained_indices.update((target_index, observe_index))
    retained = [index in retained_indices and eligible[index] for index in range(count)]

    soft_indices = [index for index, value in enumerate(retained) if value]
    soft_values = _softmax([float(q_values[index]) for index in soft_indices], temperature)
    probabilities = [0.0] * count
    for index, value in zip(soft_indices, soft_values, strict=True):
        probabilities[index] = (1.0 - target_weight) * value
    probabilities[target_index] += target_weight

    floor = min(map(float, q_values)) - 1.0
    policy_scores = [floor] * count
    for index in eligible_indices:
        policy_scores[index] = float(q_values[index])
    strongest_other = max(
        (score for index, score in enumerate(policy_scores) if index != target_index),
        default=policy_scores[target_index] - policy_margin,
    )
    policy_scores[target_index] = max(
        policy_scores[target_index], strongest_other + float(policy_margin)
    )

    target_score = policy_scores[target_index]
    acceptable = [
        bool(eligible[index] and target_score - policy_scores[index] <= acceptable_tolerance)
        for index in range(count)
    ]
    acceptable[target_index] = True
    return PolicySupervision(
        probabilities=tuple(probabilities),
        policy_scores=tuple(policy_scores),
        eligible=tuple(eligible),
        retained=tuple(retained),
        acceptable=tuple(acceptable),
    )


def policy_regret(policy_scores: Sequence[float], selected_index: int) -> float:
    if not policy_scores or not 0 <= selected_index < len(policy_scores):
        raise IndexError("selected candidate is out of range")
    return max(map(float, policy_scores)) - float(policy_scores[selected_index])
