"""Audit-only counterfactual worlds derived from fixed TMCD source scenarios.

The source scenario remains the only training scenario.  Its paired world is
used solely by the public-state robust teacher to compute a max-min action;
neither hidden world, oracle fields, nor counterfactual identifiers are ever
placed in the model prompt or target.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from agentguard_zero.env.checker import full_check
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.public_teacher import public_state_digest
from agentguard_zero.schemas.scenario_schema import OBJECTIVES
from agentguard_zero.schemas.scenario_schema_v2 import paired_counterpart_v2


def scenario_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one TMCD scenario from a parquet/JSON row."""

    for key in ("scenario", "scenario_json"):
        value = row.get(key)
        if isinstance(value, dict):
            return copy.deepcopy(value)
        if isinstance(value, str) and value.strip():
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    extra = row.get("extra_info")
    if isinstance(extra, str) and extra.strip():
        extra = json.loads(extra)
    if isinstance(extra, dict):
        value = extra.get("scenario")
        if isinstance(value, dict):
            return copy.deepcopy(value)
        if isinstance(value, str) and value.strip():
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    if row.get("protocol_version") == "tmcd-v2":
        return copy.deepcopy(dict(row))
    raise ValueError("row does not contain a TMCD scenario")


def load_source_scenarios(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict(orient="records")
    elif path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        else:
            rows = payload.get("scenarios", [])
    return [scenario_from_row(dict(row)) for row in rows]


def audit_counterfactual(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return a valid hidden counterpart with an identical initial public state.

    T2 already has a protocol-defined betrayal/legitimate-change counterpart.
    For T1/T3/T4, changing only the hidden attack objective and matching oracle
    objective is sufficient to create a distinct solvable world without
    altering any public observation.  Event truth labels stay untouched: this
    preserves every task-specific validity invariant in the fixed source set.
    """

    original = copy.deepcopy(dict(scenario))
    if original.get("scenario_family") == "trust_betrayal":
        counterpart = paired_counterpart_v2(original)
    else:
        counterpart = copy.deepcopy(original)
        current = str(counterpart["true_attack"]["objective"])
        alternative = (
            "sabotage"
            if current != "sabotage"
            else next(item for item in sorted(OBJECTIVES) if item != current)
        )
        counterpart["scenario_id"] = (
            f"{counterpart['scenario_id']}-recovery-counterfactual"
        )
        counterpart["true_attack"]["objective"] = alternative
        counterpart["oracle"]["true_objective"] = alternative
        counterpart["oracle"]["success_condition"] = f"prevent_{alternative}"
        metadata = dict(counterpart.get("metadata", {}) or {})
        metadata["recovery_counterfactual"] = True
        counterpart["metadata"] = metadata

    checks = full_check(counterpart)
    if not checks.get("all_ok", False):
        raise ValueError(
            "derived counterfactual failed TMCD hard checks: "
            f"{counterpart.get('scenario_id')}: {checks}"
        )
    original_digest = public_state_digest(instantiate_scenario(original).observe())
    paired_digest = public_state_digest(
        instantiate_scenario(counterpart).observe()
    )
    if original_digest != paired_digest:
        raise ValueError("derived counterfactual changed the initial public state")
    return counterpart


def counterfactual_groups(
    scenarios: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Create one two-world audit group for each fixed source scenario."""

    groups: list[list[dict[str, Any]]] = []
    source_ids: set[str] = set()
    for scenario in scenarios:
        source = copy.deepcopy(dict(scenario))
        scenario_id = str(source.get("scenario_id", ""))
        if not scenario_id or scenario_id in source_ids:
            raise ValueError("source scenarios require unique non-empty scenario_id")
        source_ids.add(scenario_id)
        source_checks = full_check(source)
        if not source_checks.get("all_ok", False):
            raise ValueError(
                f"source scenario failed TMCD hard checks: {scenario_id}: "
                f"{source_checks}"
            )
        groups.append([source, audit_counterfactual(source)])
    return groups
