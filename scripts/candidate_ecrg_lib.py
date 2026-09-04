#!/usr/bin/env python3
"""Shared exact-K=6 public candidate and ECRG selection helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from agentguard_zero.candidate.generator import ACTION_FAMILIES, CandidateGenerator
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.governance.v5c import safe_probe_fallback
from agentguard_zero.recovery.public_teacher import public_state_digest
from scripts.ecrg_calibration_lib import candidate_features, select_trace_decision


K6_QUOTAS = {family: 1 for family in ACTION_FAMILIES}


def exact_k6_generator() -> CandidateGenerator:
    return CandidateGenerator(
        min_candidates=6,
        max_candidates=6,
        quotas=K6_QUOTAS,
    )


def candidate_set_seed(
    observation: Mapping[str, Any],
    *,
    evaluation_seed: int,
) -> int:
    digest = public_state_digest(observation)
    wire = f"candidate-ecrg-k6:{evaluation_seed}:{digest}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(wire).digest()[:8], "big")


def build_k6_decision(
    observation: dict[str, Any],
    options: list[CandidateOption],
) -> dict[str, Any]:
    if len(options) != 6:
        raise RuntimeError(f"ECRG requires exactly six candidates, got {len(options)}")
    public_context = {"observation": observation}
    candidates = []
    scored = []
    for index, option in enumerate(options):
        text = json.dumps(
            option.compiled_packet,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        features, parsed = candidate_features(public_context, text, index=index)
        scored.append(parsed)
        candidates.append(
            {
                "index": index,
                "candidate_key": option.candidate_key,
                "semantic_id": option.semantic_id,
                "action_family": option.action_family,
                "packet": copy.deepcopy(option.compiled_packet),
                "features": features,
            }
        )
    fallback_packet, fallback_diagnostics = safe_probe_fallback(
        public_context,
        scored,
    )
    fallback_text = json.dumps(
        fallback_packet,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    fallback_features, _ = candidate_features(
        public_context,
        fallback_text,
        index=-1,
    )
    return {
        "candidate_count": 6,
        "candidates": candidates,
        "fallback": {
            "index": -1,
            "candidate_key": None,
            "semantic_id": CandidateOption.packet_digest(fallback_packet),
            "action_family": "fallback",
            "packet": fallback_packet,
            "features": fallback_features,
            "fallback_diagnostics": fallback_diagnostics,
        },
    }


def select_ecrg(
    decision: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    selected = select_trace_decision(decision, config)
    if not isinstance(selected.get("packet"), dict):
        raise RuntimeError("ECRG selected a candidate without a compiled packet")
    return selected


def candidate_set_digest(decision: dict[str, Any]) -> str:
    payload = [
        {
            "index": row["index"],
            "semantic_id": row["semantic_id"],
            "action_family": row["action_family"],
            "packet": row["packet"],
        }
        for row in decision["candidates"]
    ]
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()
