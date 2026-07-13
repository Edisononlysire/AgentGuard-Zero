from __future__ import annotations

from typing import Any, Dict, List


def log_query(events: List[Dict[str, Any]], source: str | None = None, time: int | None = None) -> Dict[str, Any]:
    out = []
    for e in events:
        if source is not None and e.get("source") != source:
            continue
        if time is not None and e.get("time") != time:
            continue
        out.append(e)
    return {"tool": "LogQuery", "events": out}
