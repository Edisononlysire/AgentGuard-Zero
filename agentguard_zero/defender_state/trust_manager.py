from __future__ import annotations

import copy
import math
from typing import Any, Iterable

from agentguard_zero.defender_state.evidence_store import EvidenceStore


SOURCE_STATUSES = {"stable", "uncertain", "degraded", "recovering"}
CLAIM_STATUSES = {"unassessed", "challenged", "supported", "contradicted"}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _evidence_signal(record: dict[str, Any]) -> tuple[float, float]:
    payload = record.get("public_payload", {}) or {}
    verdict = str(payload.get("verdict", "")).lower()
    positive = 0.0
    negative = 0.0
    if verdict in {"supported", "strongly_supported", "plausible", "challenge_passed", "consistent"}:
        positive += 1.0
    if verdict in {"suspicious", "challenge_failed", "contradicted", "inconsistent"}:
        negative += 1.0
    positive += _clamp(payload.get("consistency", 0.0))
    positive += _clamp(payload.get("challenge_consistency", 0.0))
    positive += _clamp(payload.get("source_reliability", 0.0))
    negative += _clamp(payload.get("contradiction_risk", 0.0))
    negative += _clamp(payload.get("leakage_risk", 0.0)) if payload.get("canary_triggered") else 0.0
    return min(2.0, positive), min(2.0, negative)


class ContextualTrustManager:
    """Persistent defender trust beliefs updated only from public evidence."""

    def __init__(self, *, prior_strength: float = 2.0, decay: float = 0.95) -> None:
        self.prior_strength = max(0.1, float(prior_strength))
        self.decay = _clamp(decay, 0.0, 1.0)
        self.source_reputation: dict[str, dict[str, Any]] = {}
        self.claim_trust: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.applied_trust_evidence: set[tuple[str, str]] = set()

    def ensure_source(self, source_id: str, *, public_prior: float = 0.5, time: int = 0) -> dict[str, Any]:
        source_id = str(source_id or "unknown")
        if source_id not in self.source_reputation:
            prior = _clamp(public_prior)
            self.source_reputation[source_id] = {
                "alpha": self.prior_strength * prior,
                "beta": self.prior_strength * (1.0 - prior),
                "prior": prior,
                "mean": prior,
                "uncertainty": 1.0 / (self.prior_strength + 1.0),
                "status": "uncertain",
                "last_updated_at": int(time),
                "last_verified_at": None,
                "last_conflict_at": None,
                "support_streak": 0,
                "conflict_streak": 0,
                "recovery_streak": 0,
            }
        return self.source_reputation[source_id]

    def register_claim(
        self,
        public_event: dict[str, Any],
        *,
        time: int,
        public_prior: float = 0.5,
    ) -> None:
        event_id = str(public_event.get("event_id", ""))
        if not event_id or event_id in self.claim_trust:
            return
        source_id = str(public_event.get("source_id") or public_event.get("source") or "unknown")
        source = self.ensure_source(
            source_id,
            public_prior=public_prior,
            time=time,
        )
        self.claim_trust[event_id] = {
            "event_id": event_id,
            "source_id": source_id,
            "score": float(source["mean"]),
            "status": "unassessed",
            "provenance_score": 0.0,
            "cross_source_support": 0.0,
            "graph_support": 0.0,
            "contradiction_score": 0.0,
            "evidence_refs": [],
            "claim_semantics": copy.deepcopy(public_event.get("claim_semantics", {})),
            "updated_at": int(time),
        }

    def _decay_source(self, source: dict[str, Any], *, time: int) -> None:
        elapsed = max(0, int(time) - int(source.get("last_updated_at", time)))
        if elapsed <= 0:
            return
        factor = self.decay**elapsed
        prior = float(source.get("prior", 0.5))
        source["alpha"] = self.prior_strength * prior + factor * (
            float(source["alpha"]) - self.prior_strength * prior
        )
        source["beta"] = self.prior_strength * (1.0 - prior) + factor * (
            float(source["beta"]) - self.prior_strength * (1.0 - prior)
        )

    @staticmethod
    def _refresh_source(source: dict[str, Any]) -> None:
        total = max(1e-6, float(source["alpha"]) + float(source["beta"]))
        source["mean"] = _clamp(float(source["alpha"]) / total)
        source["uncertainty"] = _clamp(1.0 / (total + 1.0))
        if source["conflict_streak"] >= 2 or source["mean"] < 0.35:
            source["status"] = "degraded"
        elif source["recovery_streak"] > 0 and source["mean"] < 0.65:
            source["status"] = "recovering"
        elif source["mean"] >= 0.65 and source["uncertainty"] <= 0.25:
            source["status"] = "stable"
        else:
            source["status"] = "uncertain"

    def _refresh_claim(
        self,
        claim: dict[str, Any],
        source: dict[str, Any],
        records: list[dict[str, Any]],
        *,
        time: int,
    ) -> None:
        positive = 0.0
        negative = 0.0
        tool_types: set[str] = set()
        for record in records:
            pos, neg = _evidence_signal(record)
            positive += pos
            negative += neg
            tool_types.add(str(record.get("evidence_type", "")))
        claim["provenance_score"] = _clamp(positive / 2.0) if any("provenance" in x for x in tool_types) else 0.0
        claim["cross_source_support"] = _clamp(positive / 2.0) if any("crosscheck" in x for x in tool_types) else 0.0
        claim["graph_support"] = _clamp(positive / 2.0) if any("graphquery" in x for x in tool_types) else 0.0
        claim["contradiction_score"] = _clamp(negative / 2.0)
        support = _clamp(positive / 3.0)
        contradiction = _clamp(negative / 3.0)
        claim["score"] = _clamp(0.55 * float(source["mean"]) + 0.45 * support - 0.60 * contradiction)
        if contradiction >= 0.5 or claim["score"] <= 0.30:
            claim["status"] = "contradicted"
        elif claim["score"] >= 0.70 and support > contradiction:
            claim["status"] = "supported"
        elif records:
            claim["status"] = "challenged"
        else:
            claim["status"] = "unassessed"
        claim["updated_at"] = int(time)

    def apply(
        self,
        operations: Iterable[dict[str, Any]],
        *,
        evidence_store: EvidenceStore,
        time: int,
    ) -> list[dict[str, Any]]:
        committed: list[dict[str, Any]] = []
        for operation in operations or []:
            op = str(operation.get("op", ""))
            source_id = str(operation.get("source_id", "unknown"))
            event_id = str(operation.get("event_id", ""))
            refs = [str(item) for item in operation.get("evidence_refs", [])]
            valid, reason = evidence_store.validate_refs(refs, time=time)
            if not valid:
                committed.append({"committed": False, "op": op, "reason": reason})
                continue
            registered_claim = self.claim_trust.get(event_id) if event_id else None
            if op != "hold" and registered_claim is None:
                committed.append(
                    {
                        "committed": False,
                        "op": op,
                        "source_id": source_id,
                        "event_id": event_id,
                        "reason": "unknown_claim_event",
                        "time": int(time),
                    }
                )
                continue
            if registered_claim is not None and str(registered_claim.get("source_id")) != source_id:
                committed.append(
                    {
                        "committed": False,
                        "op": op,
                        "source_id": source_id,
                        "event_id": event_id,
                        "reason": "claim_source_mismatch",
                        "time": int(time),
                    }
                )
                continue
            if op != "hold":
                relevant, relevance_reason = evidence_store.refs_support_claim(
                    refs,
                    registered_claim.get("claim_semantics", {}),
                    time=time,
                )
                if not relevant:
                    committed.append(
                        {
                            "committed": False,
                            "op": op,
                            "source_id": source_id,
                            "event_id": event_id,
                            "reason": relevance_reason,
                            "time": int(time),
                        }
                    )
                    continue
            credibility_ops = {"support", "contradict", "recover"}
            fresh_refs = [
                ref
                for ref in refs
                if op not in credibility_ops or (source_id, ref) not in self.applied_trust_evidence
            ]
            if op in credibility_ops and not fresh_refs:
                event = {
                    "committed": False,
                    "op": op,
                    "source_id": source_id,
                    "event_id": event_id,
                    "evidence_refs": refs,
                    "reason": "duplicate_trust_evidence",
                    "time": int(time),
                }
                committed.append(event)
                self.events.append(copy.deepcopy(event))
                continue
            source = self.ensure_source(source_id, time=time)
            self._decay_source(source, time=time)
            records = [
                record
                for ref in fresh_refs
                if (record := evidence_store.get(ref, time=time)) is not None
            ]
            positive, negative = 0.0, 0.0
            for record in records:
                pos, neg = _evidence_signal(record)
                positive += pos
                negative += neg

            allowed = True
            reason = "ok"
            if op == "support":
                allowed = bool(records and positive > negative)
                reason = "support_requires_positive_evidence" if not allowed else "ok"
                if allowed:
                    source["alpha"] += min(2.0, positive)
                    source["support_streak"] += 1
                    source["conflict_streak"] = 0
                    source["last_verified_at"] = int(time)
            elif op == "contradict":
                allowed = bool(records and negative > positive)
                reason = "contradict_requires_negative_evidence" if not allowed else "ok"
                if allowed:
                    source["beta"] += min(2.0, negative)
                    source["conflict_streak"] += 1
                    source["support_streak"] = 0
                    source["last_conflict_at"] = int(time)
            elif op == "recover":
                allowed = evidence_store.independent_count(fresh_refs, time=time) >= 2 and positive > negative
                reason = "recover_requires_two_independent_supports" if not allowed else "ok"
                if allowed:
                    source["alpha"] += min(2.0, positive)
                    source["recovery_streak"] += 1
                    source["conflict_streak"] = max(0, int(source["conflict_streak"]) - 1)
            elif op == "challenge":
                if event_id and event_id in self.claim_trust:
                    self.claim_trust[event_id]["status"] = "challenged"
                    self.claim_trust[event_id]["updated_at"] = int(time)
            elif op != "hold":
                allowed = False
                reason = "invalid_trust_operation"

            if allowed:
                if op in credibility_ops:
                    self.applied_trust_evidence.update((source_id, ref) for ref in fresh_refs)
                source["last_updated_at"] = int(time)
                self._refresh_source(source)
                if event_id and event_id in self.claim_trust:
                    claim = self.claim_trust[event_id]
                    claim["evidence_refs"] = sorted(set(claim.get("evidence_refs", [])) | set(refs))
                    cumulative_records = [
                        record
                        for ref in claim["evidence_refs"]
                        if (record := evidence_store.get(ref, time=time)) is not None
                    ]
                    self._refresh_claim(claim, source, cumulative_records, time=time)
            event = {
                "committed": bool(allowed),
                "op": op,
                "source_id": source_id,
                "event_id": event_id,
                "evidence_refs": fresh_refs if op in credibility_ops else refs,
                "reason": reason,
                "time": int(time),
            }
            committed.append(event)
            self.events.append(copy.deepcopy(event))
        return committed

    def claim_for(self, event_id: str) -> dict[str, Any] | None:
        value = self.claim_trust.get(str(event_id))
        return copy.deepcopy(value) if value is not None else None

    def public_snapshot(self) -> dict[str, Any]:
        def public_source(source: dict[str, Any]) -> dict[str, Any]:
            row = {
                "mean": float(source.get("mean", 0.5)),
                "uncertainty": float(source.get("uncertainty", 1.0)),
                "status": str(source.get("status", "uncertain")),
            }
            for key in ("last_verified_at", "last_conflict_at"):
                if source.get(key) is not None:
                    row[key] = int(source[key])
            for key in ("support_streak", "conflict_streak", "recovery_streak"):
                if int(source.get(key, 0)):
                    row[key] = int(source[key])
            return row

        def public_claim(claim: dict[str, Any]) -> dict[str, Any]:
            row = {
                "source_id": str(claim.get("source_id", "unknown")),
                "score": float(claim.get("score", 0.0)),
                "status": str(claim.get("status", "unassessed")),
                "updated_at": int(claim.get("updated_at", 0)),
            }
            for key in (
                "provenance_score",
                "cross_source_support",
                "graph_support",
                "contradiction_score",
            ):
                if float(claim.get(key, 0.0)):
                    row[key] = float(claim[key])
            if claim.get("evidence_refs"):
                row["evidence_refs"] = copy.deepcopy(claim["evidence_refs"])
            return row

        return {
            "source_reputation": {
                source_id: public_source(source)
                for source_id, source in self.source_reputation.items()
            },
            "current_claim_trust": {
                event_id: public_claim(claim)
                for event_id, claim in self.claim_trust.items()
            },
        }
