from __future__ import annotations

from typing import Any, Dict


def graph_query(scenario: Dict[str, Any], node: str) -> Dict[str, Any]:
    assets = scenario.get("network_context", {}).get("assets", [])
    asset = next((a for a in assets if a.get("id") == node), None)
    return {"tool": "GraphQuery", "node": node, "asset": asset, "edges": scenario.get("network_context", {}).get("reachable_edges", [])}
