"""Identifier-invariant fingerprints for candidate data split audits."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


ID_FIELDS = {
    "event_id": "event",
    "evidence_id": "evidence",
    "source_id": "source",
    "source": "source",
    "memory_id": "memory",
    "scenario_id": "scenario",
    "pair_id": "pair",
}


def _collect(value: Any, mappings: dict[str, dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            prefix = ID_FIELDS.get(str(key))
            if prefix and isinstance(item, str) and item:
                bucket = mappings.setdefault(prefix, {})
                bucket.setdefault(item, f"{prefix}_{len(bucket):03d}")
            if key == "id" and isinstance(item, str) and item:
                bucket = mappings.setdefault("asset", {})
                bucket.setdefault(item, f"asset_{len(bucket):03d}")
            _collect(item, mappings)
    elif isinstance(value, list):
        for item in value:
            _collect(item, mappings)


def normalize_public_identifiers(value: Any) -> Any:
    result = copy.deepcopy(value)
    mappings: dict[str, dict[str, str]] = {}
    _collect(result, mappings)
    replacements = {
        raw: normalized for bucket in mappings.values() for raw, normalized in bucket.items()
    }

    def replace(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: replace(value) for key, value in sorted(item.items())}
        if isinstance(item, list):
            return [replace(value) for value in item]
        if isinstance(item, str):
            return replacements.get(item, item)
        return item

    return replace(result)


def semantic_digest(value: Any) -> str:
    wire = json.dumps(
        normalize_public_identifiers(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()
