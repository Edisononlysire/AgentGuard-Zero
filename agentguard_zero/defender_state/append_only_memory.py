from __future__ import annotations

import copy
from typing import Any, Iterable

from agentguard_zero.defender_state.evidence_store import EvidenceStore
from agentguard_zero.defender_state.memory_fsm import EvidenceStateMemory, MEMORY_STATUSES, canonical_claim_key
from agentguard_zero.defender_state.trust_manager import ContextualTrustManager


class AppendOnlyProfileMemory(EvidenceStateMemory):
    """Ablation control: partitioned memory without lifecycle transitions."""

    def apply(
        self,
        operations: Iterable[dict[str, Any]],
        *,
        evidence_store: EvidenceStore,
        trust_manager: ContextualTrustManager,
        time: int,
    ) -> list[dict[str, Any]]:
        del trust_manager
        results: list[dict[str, Any]] = []
        for operation in operations or []:
            op = str(operation.get("op", ""))
            refs = [str(item) for item in operation.get("evidence_refs", [])]
            valid, reason = evidence_store.validate_refs(refs, time=time)
            if not valid:
                results.append({"committed": False, "op": op, "reason": reason})
                continue
            if op != "ingest":
                result = {
                    "committed": False,
                    "op": op,
                    "memory_id": str(operation.get("memory_id", "")),
                    "reason": "append_only_ablation_has_no_state_transitions",
                }
                results.append(result)
                self.events.append(copy.deepcopy(result) | {"time": int(time)})
                continue
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
            derived_source_ids = evidence_store.root_sources(refs, time=time)
            declared_source_ids = sorted(
                {str(item) for item in operation.get("source_ids", []) if str(item)}
            )
            if declared_source_ids and declared_source_ids != derived_source_ids:
                results.append(
                    {
                        "committed": False,
                        "op": op,
                        "reason": "source_lineage_mismatch",
                        "derived_source_ids": derived_source_ids,
                    }
                )
                continue
            target = str(operation.get("target_status", "quarantined"))
            if target not in MEMORY_STATUSES:
                target = "quarantined"
            if memory_id not in self.records:
                self.records[memory_id] = {
                    "memory_id": memory_id,
                    "claim_key": claim_key,
                    "claim": claim,
                    "entity_id": str(claim.get("entity_id", "")),
                    "source_ids": derived_source_ids,
                    "status": target,
                    "confidence": 0.5,
                    "created_at": int(time),
                    "updated_at": int(time),
                    "version": 1,
                    "evidence_refs": sorted(set(refs)),
                    "support_refs": sorted(set(refs)) if target == "confirmed" else [],
                    "contradiction_refs": sorted(set(refs)) if target == "rejected" else [],
                    "retrieval_count": 0,
                    "acceptance_count": 0,
                    "usage_counts": {"support": 0, "contradict": 0, "background": 0},
                    "last_retrieved_at": None,
                    "last_used_for_action": None,
                    "transition_history": [
                        {
                            "from": "unseen",
                            "to": target,
                            "op": "ingest",
                            "evidence_refs": refs,
                            "time": int(time),
                            "reason": "append_only_direct_partition",
                        }
                    ],
                }
                self.claim_index[claim_key] = memory_id
                reason = "ok"
            else:
                reason = "duplicate_claim_ignored"
            result = {
                "committed": True,
                "op": op,
                "memory_id": memory_id,
                "status": str(self.records[memory_id]["status"]),
                "reason": reason,
                "ablation": "append_only_memory",
            }
            results.append(result)
            self.events.append(copy.deepcopy(result) | {"time": int(time)})
        return results
