from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterable

from agentguard_zero.protocol import (
    EVIDENCE_ORIGIN_RAW_EVENT,
    EVIDENCE_ORIGIN_TOOL_GENERATED,
    RAW_EVENT_ALLOWED_FIELDS,
    RAW_EVENT_FORBIDDEN_SIGNAL_FIELDS,
    RAW_EVENT_RESERVED_TYPES,
)
from agentguard_zero.world.public_projector import assert_public


def _stable_id(prefix: str, payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _claim_identity(claim: Any) -> str:
    if not isinstance(claim, dict):
        return ""
    normalized = {
        key: str(claim.get(key, "")).strip().lower()
        for key in ("entity_id", "predicate", "object", "scope")
    }
    if any(not value for value in normalized.values()):
        return ""
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    """Project an internal evidence record into a compact decision view."""
    payload = copy.deepcopy(record.get("public_payload", {}))
    if not isinstance(payload, dict):
        payload = {"value": payload}
    for key in ("event_id", "source", "source_id", "time", "type"):
        payload.pop(key, None)

    projected = {
        "evidence_id": str(record.get("evidence_id", "")),
        "event_id": str(record.get("event_id", "")),
        "source_id": str(record.get("source_id", "unknown")),
        "evidence_type": str(record.get("evidence_type", "event")),
        "evidence_origin": str(record.get("evidence_origin", "unknown")),
        "content": payload,
    }
    parents = copy.deepcopy(record.get("parent_evidence_ids", []))
    if parents:
        projected["parent_evidence_ids"] = parents
    roots = copy.deepcopy(record.get("root_source_ids", []))
    if roots and roots != [projected["source_id"]]:
        projected["root_source_ids"] = roots
    return projected


class EvidenceStore:
    """Public evidence ledger with availability time and minimal provenance lineage."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._event_index: dict[str, str] = {}

    def add_event(self, public_event: dict[str, Any], *, time: int) -> str:
        assert_public(public_event)
        forbidden_signals = sorted(
            RAW_EVENT_FORBIDDEN_SIGNAL_FIELDS & set(public_event)
        )
        if forbidden_signals:
            raise ValueError(
                f"tool_signal_in_raw_event:{forbidden_signals[0]}"
            )
        extra_fields = sorted(set(public_event) - RAW_EVENT_ALLOWED_FIELDS)
        if extra_fields:
            raise ValueError(f"forbidden_raw_event_field:{extra_fields[0]}")
        event_type = str(public_event.get("type", "")).strip().lower()
        if not event_type:
            raise ValueError("missing_raw_event_type")
        if (
            event_type.startswith("tool:")
            or event_type.endswith("_result")
            or event_type in RAW_EVENT_RESERVED_TYPES
        ):
            raise ValueError(f"tool_result_type_in_raw_event:{event_type}")
        event_id = str(public_event.get("event_id", f"event-{time}"))
        source_id = str(public_event.get("source_id") or public_event.get("source") or "unknown")
        evidence_id = _stable_id("ev", {"event_id": event_id, "time": time, "source": source_id})
        record = {
            "evidence_id": evidence_id,
            "event_id": event_id,
            "source_id": source_id,
            "evidence_type": f"event:{event_type}",
            "evidence_origin": EVIDENCE_ORIGIN_RAW_EVENT,
            "public_payload": copy.deepcopy(public_event),
            "parent_evidence_ids": [],
            "root_source_ids": [source_id],
            "claim_keys": [
                claim_key
                for claim_key in [_claim_identity(public_event.get("claim_semantics"))]
                if claim_key
            ],
            "created_at": int(time),
            "available_at": int(time),
            "integrity_status": "valid",
        }
        self._records.setdefault(evidence_id, record)
        self._event_index[event_id] = evidence_id
        return evidence_id

    def add_probe_result(
        self,
        public_event: dict[str, Any],
        *,
        time: int,
        tool: str,
    ) -> str:
        """Store a delayed environment-generated probe observation."""

        assert_public(public_event)
        tool_name = str(tool).strip()
        if not tool_name:
            raise ValueError("missing_probe_tool")
        event_id = str(public_event.get("event_id", f"probe-event-{time}"))
        source_id = str(
            public_event.get("source_id")
            or public_event.get("source")
            or f"tool:{tool_name}"
        )
        evidence_id = _stable_id(
            "tool-event",
            {
                "event_id": event_id,
                "time": int(time),
                "tool": tool_name,
                "payload": public_event,
            },
        )
        claim_key = _claim_identity(public_event.get("claim_semantics"))
        self._records[evidence_id] = {
            "evidence_id": evidence_id,
            "event_id": event_id,
            "source_id": source_id,
            "evidence_type": f"tool:{tool_name.lower()}",
            "evidence_origin": EVIDENCE_ORIGIN_TOOL_GENERATED,
            "public_payload": copy.deepcopy(public_event),
            "parent_evidence_ids": [],
            "root_source_ids": [source_id],
            "claim_keys": [claim_key] if claim_key else [],
            "created_at": int(time),
            "available_at": int(time),
            "integrity_status": "valid",
        }
        return evidence_id

    def add_tool_result(
        self,
        public_result: dict[str, Any],
        *,
        time: int,
        parent_evidence_ids: Iterable[str] = (),
    ) -> str:
        assert_public(public_result)
        parents = sorted({str(item) for item in parent_evidence_ids if str(item) in self._records})
        tool = str(public_result.get("tool", "Tool"))
        source_id = str(public_result.get("verifier_id") or f"tool:{tool}")
        roots = {
            str(item)
            for item in public_result.get("root_source_ids", []) or []
            if str(item)
        }
        claim_keys = {
            _claim_identity(public_result.get("claim_semantics"))
        } - {""}
        for parent in parents:
            roots.update(str(item) for item in self._records[parent].get("root_source_ids", []))
            claim_keys.update(str(item) for item in self._records[parent].get("claim_keys", []))
        if not roots:
            roots.add(source_id)
        evidence_id = _stable_id(
            "tool",
            {"tool": tool, "time": time, "parents": parents, "payload": public_result},
        )
        self._records[evidence_id] = {
            "evidence_id": evidence_id,
            "event_id": str(public_result.get("event_id", "")),
            "source_id": source_id,
            "evidence_type": f"tool:{tool.lower()}",
            "evidence_origin": EVIDENCE_ORIGIN_TOOL_GENERATED,
            "public_payload": copy.deepcopy(public_result),
            "parent_evidence_ids": parents,
            "root_source_ids": sorted(roots),
            "claim_keys": sorted(claim_keys),
            "created_at": int(time),
            "available_at": int(time) + 1,
            "integrity_status": "valid",
        }
        return evidence_id

    def evidence_for_event(self, event_id: str) -> str | None:
        return self._event_index.get(str(event_id))

    def get(self, evidence_id: str, *, time: int | None = None) -> dict[str, Any] | None:
        record = self._records.get(str(evidence_id))
        if record is None:
            return None
        if time is not None and int(record["available_at"]) > int(time):
            return None
        return copy.deepcopy(record)

    def validate_refs(self, evidence_ids: Iterable[str], *, time: int) -> tuple[bool, str]:
        for evidence_id in evidence_ids or []:
            record = self._records.get(str(evidence_id))
            if record is None:
                return False, f"unknown_evidence:{evidence_id}"
            if record.get("integrity_status") != "valid":
                return False, f"invalid_evidence:{evidence_id}"
            if int(record.get("available_at", 0)) > int(time):
                return False, f"evidence_not_yet_available:{evidence_id}"
        return True, "ok"

    def independent_count(self, evidence_ids: Iterable[str], *, time: int) -> int:
        """Return the maximum number of records with pairwise-disjoint roots."""

        records = [self.get(item, time=time) for item in set(evidence_ids or [])]
        root_sets = [
            frozenset(str(root) for root in record.get("root_source_ids", []) if str(root))
            for record in records
            if record is not None
        ]
        root_sets = [roots for roots in root_sets if roots]
        best = 0

        def search(index: int, used: frozenset[str], count: int) -> None:
            nonlocal best
            if count + len(root_sets) - index <= best:
                return
            if index >= len(root_sets):
                best = max(best, count)
                return
            roots = root_sets[index]
            if roots.isdisjoint(used):
                search(index + 1, used | roots, count + 1)
            search(index + 1, used, count)

        search(0, frozenset(), 0)
        return best

    def root_sources(self, evidence_ids: Iterable[str], *, time: int) -> list[str]:
        roots: set[str] = set()
        for evidence_id in set(evidence_ids or []):
            record = self.get(str(evidence_id), time=time)
            if record is not None:
                roots.update(str(item) for item in record.get("root_source_ids", []) if str(item))
        return sorted(roots)

    def refs_support_claim(
        self,
        evidence_ids: Iterable[str],
        claim: dict[str, Any],
        *,
        time: int,
    ) -> tuple[bool, str]:
        expected = _claim_identity(claim)
        if not expected:
            return False, "invalid_claim_semantics"
        refs = list(evidence_ids or [])
        if not refs:
            return False, "missing_claim_evidence"
        for evidence_id in refs:
            record = self.get(str(evidence_id), time=time)
            if record is None:
                return False, f"unavailable_claim_evidence:{evidence_id}"
            if expected not in set(record.get("claim_keys", [])):
                return False, f"irrelevant_claim_evidence:{evidence_id}"
        return True, "ok"

    def available_records(self, *, time: int) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(record)
            for _, record in sorted(self._records.items())
            if int(record.get("available_at", 0)) <= int(time)
            and record.get("integrity_status") == "valid"
        ]

    def public_snapshot(self, *, time: int, limit: int = 32) -> list[dict[str, Any]]:
        records = sorted(
            self.available_records(time=time),
            key=lambda row: (
                int(row.get("available_at", 0)),
                int(row.get("created_at", 0)),
                str(row.get("evidence_id", "")),
            ),
        )
        rows = [_public_record(row) for row in records[-max(1, int(limit)) :]]
        for row in rows:
            assert_public(row)
        return rows
