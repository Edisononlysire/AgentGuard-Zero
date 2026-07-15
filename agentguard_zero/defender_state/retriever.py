from __future__ import annotations

from typing import Any

from agentguard_zero.defender_state.memory_fsm import EvidenceStateMemory


STATE_LIMITS = {"confirmed": 4, "quarantined": 2, "rejected": 2}


def _tokens(value: Any) -> set[str]:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    return {token for token in text.split() if token}


def _context_features(observed_events: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "entities": {str(event.get("entity_id", "")) for event in observed_events if event.get("entity_id")},
        "sources": {
            str(event.get("source_id") or event.get("source"))
            for event in observed_events
            if event.get("source_id") or event.get("source")
        },
        "objectives": {str(event.get("objective_hint", "")) for event in observed_events if event.get("objective_hint")},
        "types": {str(event.get("type", "")) for event in observed_events if event.get("type")},
        "terms": set().union(*(_tokens(event.get("claim", "")) for event in observed_events)) if observed_events else set(),
    }


def _claim_terms(claim: dict[str, Any]) -> set[str]:
    return set().union(
        *(
            _tokens(claim.get(key, ""))
            for key in ("entity_id", "predicate", "object", "scope", "text")
        )
    )


def retrieve_memory(
    memory: EvidenceStateMemory,
    observed_events: list[dict[str, Any]],
    *,
    time: int,
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    limits = dict(STATE_LIMITS if limits is None else limits)
    features = _context_features(observed_events)
    grouped: dict[str, list[tuple[float, str]]] = {status: [] for status in STATE_LIMITS}
    for memory_id, record in memory.records.items():
        status = str(record.get("status", "quarantined"))
        if status not in grouped:
            continue
        claim = record.get("claim", {}) or {}
        term_overlap = _claim_terms(claim) & features["terms"]
        eligible = bool(
            str(claim.get("entity_id", "")) in features["entities"]
            or set(record.get("source_ids", [])) & features["sources"]
            or str(claim.get("scope", "")) in features["objectives"]
            or str(claim.get("predicate", "")) in features["types"]
            or term_overlap
        )
        if not eligible:
            continue
        score = 0.0
        score += 3.0 if str(claim.get("entity_id", "")) in features["entities"] else 0.0
        score += 2.0 * len(set(record.get("source_ids", [])) & features["sources"])
        score += 2.0 if str(claim.get("scope", "")) in features["objectives"] else 0.0
        score += 1.0 if str(claim.get("predicate", "")) in features["types"] else 0.0
        score += min(1.0, len(term_overlap) * 0.25)
        age = max(0, int(time) - int(record.get("updated_at", time)))
        score += 0.5 / (1.0 + age)
        if score >= 1.0:
            grouped[status].append((score, memory_id))

    result: dict[str, Any] = {"retrieval_id": f"retrieval-t{int(time)}"}
    key_map = {
        "confirmed": "retrieved_confirmed",
        "quarantined": "retrieved_quarantined",
        "rejected": "rejected_warnings",
    }
    retrieved_ids: list[str] = []
    for status, rows in grouped.items():
        selected = [memory_id for _, memory_id in sorted(rows, key=lambda item: (-item[0], item[1]))[: limits[status]]]
        for memory_id in selected:
            record = memory.records[memory_id]
            record["retrieval_count"] = int(record.get("retrieval_count", 0)) + 1
            record["last_retrieved_at"] = int(time)
        result[key_map[status]] = [memory.public_record(memory_id) for memory_id in selected]
        retrieved_ids.extend(selected)
    result["retrieved_memory_ids"] = retrieved_ids
    return result
