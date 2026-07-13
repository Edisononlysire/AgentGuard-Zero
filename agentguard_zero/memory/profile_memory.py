from __future__ import annotations

from typing import Any, Dict, Iterable


def init_memory() -> Dict[str, Any]:
    return {"confirmed_profile": [], "quarantined_profile": [], "rejected_profile": []}


def _add_unique(dst, values: Iterable[str]):
    for v in values or []:
        if not isinstance(v, str) or not v.strip():
            continue
        if v not in dst:
            dst.append(v)


def update_memory(memory: Dict[str, Any], memory_update: Dict[str, Any]) -> Dict[str, Any]:
    memory = {k: list(v) for k, v in memory.items()}
    _add_unique(memory["confirmed_profile"], memory_update.get("confirmed_profile", []))
    _add_unique(memory["quarantined_profile"], memory_update.get("profile_quarantine", []))
    _add_unique(memory["rejected_profile"], memory_update.get("rejected_profile", []))
    return memory
