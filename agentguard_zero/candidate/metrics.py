"""Multi-label action and candidate-ranking metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from agentguard_zero.candidate.types import ActionFlags
from agentguard_zero.recovery.model_policy import (
    ACTIVE_PROBE_TOOLS,
    PASSIVE_VERIFICATION_TOOLS,
)


def action_flags(packet: Mapping[str, Any]) -> ActionFlags:
    tool = dict(packet.get("tool_call") or {})
    tool_name = str(tool.get("name", "None"))
    response = dict(packet.get("response") or {})
    flags = ActionFlags(
        passive_verification=tool_name in PASSIVE_VERIFICATION_TOOLS,
        active_probe=tool_name in ACTIVE_PROBE_TOOLS,
        trust=bool(packet.get("trust_operations") or packet.get("trust_operation")),
        memory_operation=bool(
            packet.get("memory_operations") or packet.get("memory_operation")
        ),
        memory_use=bool(packet.get("memory_usage") or packet.get("memory_use")),
        mitigation=str(response.get("action", "Observe")) != "Observe",
    )
    active = any(
        (
            flags.passive_verification,
            flags.active_probe,
            flags.trust,
            flags.memory_operation,
            flags.memory_use,
            flags.mitigation,
        )
    )
    return ActionFlags(**(flags.to_dict() | {"observe_only": not active}))


def summarize_candidate_traces(
    traces: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in traces]
    totals: defaultdict[str, float] = defaultdict(float)
    task_totals: dict[str, defaultdict[str, float]] = {}
    for row in rows:
        task = str(row.get("task_id", "unknown"))
        task_counter = task_totals.setdefault(task, defaultdict(float))
        invalid = bool(row.get("invalid_noop", False))
        totals["decision_count"] += 1
        task_counter["decision_count"] += 1
        totals["invalid_noop"] += int(invalid)
        task_counter["invalid_noop"] += int(invalid)
        flags = row.get("action_flags") or {}
        for name in ActionFlags.__dataclass_fields__:
            value = int(bool(flags.get(name, False)) and not invalid)
            totals[name] += value
            task_counter[name] += value
        regret = row.get("candidate_regret")
        if isinstance(regret, (int, float)):
            totals["candidate_regret_sum"] += float(regret)
            totals["candidate_regret_count"] += 1
            task_counter["candidate_regret_sum"] += float(regret)
            task_counter["candidate_regret_count"] += 1
        follows_probe = bool(row.get("follows_active_probe", False))
        totals["probe_followup_count"] += int(follows_probe)
        task_counter["probe_followup_count"] += int(follows_probe)
        followup_active = follows_probe and not bool(flags.get("observe_only", False)) and not invalid
        totals["probe_followup_active_count"] += int(followup_active)
        task_counter["probe_followup_active_count"] += int(followup_active)

    def pack(counter: Mapping[str, float]) -> dict[str, Any]:
        count = max(1.0, float(counter.get("decision_count", 0.0)))
        result = {
            "decision_count": int(counter.get("decision_count", 0.0)),
            "invalid_noop_rate": float(counter.get("invalid_noop", 0.0)) / count,
        }
        for name in ActionFlags.__dataclass_fields__:
            result[f"{name}_rate"] = float(counter.get(name, 0.0)) / count
        regret_count = float(counter.get("candidate_regret_count", 0.0))
        result["mean_candidate_regret"] = (
            float(counter.get("candidate_regret_sum", 0.0)) / regret_count
            if regret_count
            else None
        )
        followups = float(counter.get("probe_followup_count", 0.0))
        result["probe_followup_nonobserve_rate"] = (
            float(counter.get("probe_followup_active_count", 0.0)) / followups
            if followups
            else None
        )
        return result

    return {
        **pack(totals),
        "by_task": {task: pack(counter) for task, counter in sorted(task_totals.items())},
    }
