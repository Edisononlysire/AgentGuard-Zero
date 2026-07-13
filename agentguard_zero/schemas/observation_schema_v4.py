from __future__ import annotations

import copy
from typing import Any

from agentguard_zero.world.public_projector import assert_public


def make_observation_v4(
    *,
    time: int,
    observed_events: list[dict[str, Any]],
    evidence_snapshot: list[dict[str, Any]],
    trust_snapshot: dict[str, Any],
    memory_retrieval: dict[str, Any],
    remaining_business_budget: float,
    verification_remaining: float,
    last_tool_result: dict[str, Any] | None,
    public_probe_state: list[dict[str, Any]],
) -> dict[str, Any]:
    observation = {
        "protocol_version": "tmcd-v2",
        "schema_version": 4,
        "time": int(time),
        "observed_events": copy.deepcopy(observed_events),
        "available_evidence": copy.deepcopy(evidence_snapshot),
        "defender_state": {
            "trust": copy.deepcopy(trust_snapshot),
            "memory": copy.deepcopy(memory_retrieval),
            "probe_state": copy.deepcopy(public_probe_state),
        },
        "defense_context": {
            "remaining_business_budget": float(remaining_business_budget),
            "remaining_verification_budget": float(verification_remaining),
        },
        "last_tool_result": copy.deepcopy(last_tool_result),
    }
    assert_public(observation)
    return observation

