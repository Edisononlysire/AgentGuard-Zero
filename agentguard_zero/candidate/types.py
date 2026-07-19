"""Public candidate types shared by training, inference, and evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from dataclasses import replace
from typing import Any, Mapping


@dataclass(frozen=True)
class ActionFlags:
    passive_verification: bool = False
    active_probe: bool = False
    trust: bool = False
    memory_operation: bool = False
    memory_use: bool = False
    mitigation: bool = False
    observe_only: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateOption:
    candidate_key: str
    semantic_id: str
    action_family: str
    public_summary: str
    referenced_ids: tuple[str, ...]
    compiled_packet: dict[str, Any]
    action_flags: ActionFlags
    audit_metadata: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        """Backward-compatible alias for the ephemeral policy key."""

        return self.candidate_key

    @staticmethod
    def packet_digest(packet: Mapping[str, Any]) -> str:
        wire = json.dumps(
            dict(packet), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(wire.encode("utf-8")).hexdigest()[:20]

    @classmethod
    def from_packet(
        cls,
        *,
        action_family: str,
        public_summary: str,
        referenced_ids: tuple[str, ...],
        compiled_packet: Mapping[str, Any],
        action_flags: ActionFlags,
        audit_metadata: Mapping[str, Any] | None = None,
        candidate_key: str | None = None,
    ) -> "CandidateOption":
        packet = copy.deepcopy(dict(compiled_packet))
        semantic_id = cls.packet_digest(packet)
        return cls(
            candidate_key=str(candidate_key or semantic_id),
            semantic_id=semantic_id,
            action_family=str(action_family),
            public_summary=str(public_summary),
            referenced_ids=tuple(sorted(set(map(str, referenced_ids)))),
            compiled_packet=packet,
            action_flags=action_flags,
            audit_metadata=copy.deepcopy(dict(audit_metadata or {})),
        )

    def with_candidate_key(self, key: str) -> "CandidateOption":
        if not re.fullmatch(r"k_[a-f0-9]{12}", str(key)):
            raise ValueError(f"invalid ephemeral candidate key: {key}")
        return replace(self, candidate_key=str(key))

    def public_record(self, *, include_packet: bool = False) -> dict[str, Any]:
        record = {
            "candidate_key": self.candidate_key,
            "semantic_id": self.semantic_id,
            "action_family": self.action_family,
            "public_summary": self.public_summary,
            "referenced_ids": list(self.referenced_ids),
            "action_flags": self.action_flags.to_dict(),
            "audit_metadata": copy.deepcopy(self.audit_metadata),
        }
        if include_packet:
            record["compiled_packet"] = copy.deepcopy(self.compiled_packet)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CandidateOption":
        packet = copy.deepcopy(dict(record.get("compiled_packet") or {}))
        flags = ActionFlags(**dict(record.get("action_flags") or {}))
        option = cls.from_packet(
            action_family=str(record.get("action_family", "")),
            public_summary=str(record.get("public_summary", "")),
            referenced_ids=tuple(map(str, record.get("referenced_ids", []) or [])),
            compiled_packet=packet,
            action_flags=flags,
            audit_metadata=dict(record.get("audit_metadata") or {}),
            candidate_key=str(
                record.get("candidate_key")
                or record.get("candidate_id")
                or cls.packet_digest(packet)
            ),
        )
        supplied = str(record.get("semantic_id", option.semantic_id))
        if supplied != option.semantic_id:
            raise ValueError("candidate record digest mismatch")
        return option
