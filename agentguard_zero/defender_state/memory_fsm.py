from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Iterable

from agentguard_zero.defender_state.evidence_store import EvidenceStore
from agentguard_zero.defender_state.trust_manager import ContextualTrustManager


MEMORY_STATUSES = {"confirmed", "quarantined", "rejected"}


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def canonical_claim_key(claim: dict[str, Any]) -> str:
    identity = {
        key: _normalize(claim.get(key))
        for key in ("entity_id", "predicate", "object", "scope")
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EvidenceStateMemory:
    """Auditable three-state memory with evidence-gated transitions."""

    def __init__(self, *, promotion_threshold: float = 0.70) -> None:
        self.promotion_threshold = float(promotion_threshold)
        self.records: dict[str, dict[str, Any]] = {}
        self.claim_index: dict[str, str] = {}
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _memory_id(claim_key: str) -> str:
        return f"mem-{claim_key[:16]}"

    def _transition(
        self,
        record: dict[str, Any],
        target: str,
        *,
        op: str,
        refs: list[str],
        time: int,
        reason: str,
    ) -> dict[str, Any]:
        previous = str(record["status"])
        record["status"] = target
        record["updated_at"] = int(time)
        record["version"] = int(record.get("version", 1)) + 1
        record["evidence_refs"] = sorted(set(record.get("evidence_refs", [])) | set(refs))
        transition = {
            "from": previous,
            "to": target,
            "op": op,
            "evidence_refs": refs,
            "time": int(time),
            "reason": reason,
        }
        record["transition_history"].append(transition)
        return transition

    def apply(
        self,
        operations: Iterable[dict[str, Any]],
        *,
        evidence_store: EvidenceStore,
        trust_manager: ContextualTrustManager,
        time: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for operation in operations or []:
            op = str(operation.get("op", ""))
            refs = [str(item) for item in operation.get("evidence_refs", [])]
            valid, reason = evidence_store.validate_refs(refs, time=time)
            if not valid:
                results.append({"committed": False, "op": op, "reason": reason})
                continue

            if op == "ingest":
                claim = copy.deepcopy(operation.get("claim", {}))
                relevant, relevance_reason = evidence_store.refs_support_claim(
                    refs,
                    claim,
                    time=time,
                )
                if not relevant:
                    results.append({"committed": False, "op": op, "reason": relevance_reason})
                    continue
                claim_key = canonical_claim_key(claim)
                memory_id = self.claim_index.get(claim_key, self._memory_id(claim_key))
                if memory_id not in self.records:
                    source_ids = sorted({str(item) for item in operation.get("source_ids", []) if str(item)})
                    self.records[memory_id] = {
                        "memory_id": memory_id,
                        "claim_key": claim_key,
                        "claim": claim,
                        "entity_id": str(claim.get("entity_id", "")),
                        "source_ids": source_ids,
                        "status": "quarantined",
                        "confidence": 0.5,
                        "created_at": int(time),
                        "updated_at": int(time),
                        "version": 1,
                        "evidence_refs": sorted(set(refs)),
                        "support_refs": [],
                        "contradiction_refs": [],
                        "retrieval_count": 0,
                        "acceptance_count": 0,
                        "last_retrieved_at": None,
                        "last_used_for_action": None,
                        "transition_history": [
                            {
                                "from": "unseen",
                                "to": "quarantined",
                                "op": "ingest",
                                "evidence_refs": refs,
                                "time": int(time),
                                "reason": "new_claims_enter_quarantine",
                            }
                        ],
                    }
                    self.claim_index[claim_key] = memory_id
                result = {
                    "committed": True,
                    "op": op,
                    "memory_id": memory_id,
                    "status": "quarantined",
                    "fallback": operation.get("target_status") == "confirmed",
                    "reason": "direct_confirm_downgraded" if operation.get("target_status") == "confirmed" else "ok",
                }
                results.append(result)
                self.events.append(copy.deepcopy(result) | {"time": int(time)})
                continue

            memory_id = str(operation.get("memory_id", ""))
            record = self.records.get(memory_id)
            if record is None:
                results.append({"committed": False, "op": op, "reason": "unknown_memory_id"})
                continue
            current = str(record["status"])
            relevant, relevance_reason = evidence_store.refs_support_claim(
                refs,
                record.get("claim", {}),
                time=time,
            )
            if not relevant:
                results.append(
                    {
                        "committed": False,
                        "op": op,
                        "memory_id": memory_id,
                        "reason": relevance_reason,
                    }
                )
                continue
            event_id = str(operation.get("event_id", ""))
            claim_trust = trust_manager.claim_for(event_id) if event_id else None
            claim_score = float((claim_trust or {}).get("score", 0.0))
            claim_status = str((claim_trust or {}).get("status", "unassessed"))
            independent = evidence_store.independent_count(refs, time=time)

            target = ""
            allowed = False
            reason = "illegal_transition"
            if op == "promote" and current == "quarantined":
                allowed = bool(
                    refs
                    and independent >= 2
                    and claim_score >= self.promotion_threshold
                    and claim_status == "supported"
                )
                target = "confirmed"
                reason = "ok" if allowed else "promotion_requires_supported_claim_and_two_independent_evidence_roots"
            elif op == "demote" and current == "confirmed":
                allowed = bool(refs and claim_status in {"challenged", "contradicted"})
                target = "quarantined"
                reason = "ok" if allowed else "demotion_requires_public_conflict"
            elif op == "reject" and current in {"quarantined", "confirmed"}:
                allowed = bool(refs and claim_status == "contradicted")
                target = "rejected"
                reason = "ok" if allowed else "rejection_requires_verified_contradiction"
            elif op == "reopen" and current == "rejected":
                allowed = bool(refs and independent >= 2 and claim_status in {"challenged", "supported"})
                target = "quarantined"
                reason = "ok" if allowed else "reopen_requires_two_independent_supports"

            if allowed:
                transition = self._transition(
                    record,
                    target,
                    op=op,
                    refs=refs,
                    time=time,
                    reason=reason,
                )
                record["confidence"] = claim_score
                if target == "confirmed":
                    record["support_refs"] = sorted(set(record.get("support_refs", [])) | set(refs))
                if target in {"quarantined", "rejected"} and op in {"demote", "reject"}:
                    record["contradiction_refs"] = sorted(
                        set(record.get("contradiction_refs", [])) | set(refs)
                    )
                result = {"committed": True, "op": op, "memory_id": memory_id, "transition": transition}
            else:
                result = {"committed": False, "op": op, "memory_id": memory_id, "reason": reason}
            results.append(result)
            self.events.append(copy.deepcopy(result) | {"time": int(time)})
        return results

    def record_usage(self, usage: Iterable[dict[str, Any]], *, retrieved_ids: set[str], time: int) -> list[str]:
        accepted: list[str] = []
        for item in usage or []:
            memory_id = str(item.get("memory_id", ""))
            if memory_id not in retrieved_ids or memory_id not in self.records:
                continue
            record = self.records[memory_id]
            record["acceptance_count"] = int(record.get("acceptance_count", 0)) + 1
            record["last_used_for_action"] = {
                "time": int(time),
                "usage": str(item.get("usage", "background")),
                "used_for": str(item.get("used_for", "belief")),
            }
            accepted.append(memory_id)
        return accepted

    def public_record(self, memory_id: str) -> dict[str, Any]:
        record = self.records[memory_id]
        keys = (
            "memory_id",
            "claim_key",
            "claim",
            "entity_id",
            "source_ids",
            "status",
            "confidence",
            "updated_at",
            "version",
            "evidence_refs",
        )
        return {key: copy.deepcopy(record.get(key)) for key in keys}

    def partition_view(self) -> dict[str, list[str]]:
        return {
            "confirmed_profile": sorted(key for key, row in self.records.items() if row["status"] == "confirmed"),
            "quarantined_profile": sorted(key for key, row in self.records.items() if row["status"] == "quarantined"),
            "rejected_profile": sorted(key for key, row in self.records.items() if row["status"] == "rejected"),
        }
