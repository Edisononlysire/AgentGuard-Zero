"""Deterministically compile an admitted candidate into Action Schema v4."""

from __future__ import annotations

import copy
from collections.abc import Iterable

from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.schemas.action_schema_v4 import validate_action_packet_v4
from agentguard_zero.world.public_projector import assert_public


class InvalidCandidate(ValueError):
    """Raised when a policy selection cannot be compiled safely."""


class CandidateCompiler:
    def compile(
        self, candidate_id: str, candidates: Iterable[CandidateOption]
    ) -> dict:
        matches = [row for row in candidates if row.candidate_id == str(candidate_id)]
        if len(matches) != 1:
            raise InvalidCandidate(
                f"candidate_id must identify exactly one option: {candidate_id}"
            )
        packet = copy.deepcopy(matches[0].compiled_packet)
        assert_public(packet)
        valid, reason = validate_action_packet_v4(packet)
        if not valid:
            raise InvalidCandidate(f"candidate packet is invalid: {reason}")
        if CandidateOption.packet_digest(packet) != matches[0].semantic_id:
            raise InvalidCandidate("candidate packet digest changed after admission")
        return packet
