from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any


from agentguard_zero.protocol import TASK_FAMILY_MAP, task_id_from_focus
from agentguard_zero.schemas.scenario_schema_v2 import (
    OOD_FAMILIES,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    minimal_example_v2,
)


TASK_FOCI = (
    "T1 Defensive Active Probing",
    "T2 Long-Horizon Trust-Building Betrayal",
    "T3 Profile and Memory Poisoning",
    "T4 Business-Constrained Overresponse Induction",
)

DCA_PROMPT_VERSION = 6


FOCUS_REQUIREMENTS = {
    "T1 Defensive Active Probing": (
        "Make existing evidence insufficient and include a safe opportunity for SourceChallenge, "
        "CanaryProbe, DecoyProbe, or ShadowActionProbe to produce discriminating evidence."
    ),
    "T2 Long-Horizon Trust-Building Betrayal": (
        "Use at least two truthful trust-building steps before a later high-impact false claim."
    ),
    "T3 Profile and Memory Poisoning": (
        "Use at least two poisoned profile claims whose harm appears in a later clean decision phase."
    ),
    "T4 Business-Constrained Overresponse Induction": (
        "Target a critical asset and make premature isolation exceed or threaten the business budget."
    ),
}

FOCUS_FAMILIES = {
    focus: TASK_FAMILY_MAP[task_id_from_focus(focus)] for focus in TASK_FOCI
}


def task_example_v2(task_focus: str) -> dict[str, Any]:
    """Return a compact, hard-gate-valid example for the requested task."""

    task_id = task_id_from_focus(task_focus)
    scenario = minimal_example_v2(trajectory_type="betrayal")
    scenario["scenario_id"] = f"TMCD-V2-{task_id}-SCHEMA-EXAMPLE"
    scenario["scenario_family"] = FOCUS_FAMILIES.get(
        task_focus, "active_probe_ambiguity"
    )
    scenario["metadata"] = {
        "task_id": task_id,
        "manipulation_family": scenario["scenario_family"],
    }
    if task_id == "T2":
        return scenario

    for key in ("pair_id", "prefix_hash", "divergence_time", "trajectory_type"):
        scenario.pop(key, None)

    if task_id == "T1":
        ambiguous = scenario["event_schedule"][1]
        ambiguous["truth_value"] = False
        ambiguous["is_fake"] = True
        ambiguous["spoofability"] = 0.65
        scenario["oracle"]["actionable_from"] = int(ambiguous["time"])
        scenario["oracle"]["first_actionable_event_id"] = ambiguous["event_id"]
    elif task_id == "T3":
        second = copy.deepcopy(scenario["event_schedule"][2])
        second["event_id"] = "event-poison-profile-2"
        second["time"] = 1
        second["claim"] = "database profile risk is low"
        second["claim_semantics"]["predicate"] = "asset_risk_level"
        second["claim_semantics"]["object"] = "low"
        second["truth_value"] = False
        second["is_fake"] = True
        scenario["event_schedule"].insert(2, second)
        scenario["oracle"]["actionable_from"] = int(second["time"])
        scenario["oracle"]["first_actionable_event_id"] = second["event_id"]
    elif task_id == "T4":
        pressure = scenario["event_schedule"][2]
        pressure["claim"] = "urgent isolation requested for critical database"
        pressure["truth_value"] = False
        pressure["is_fake"] = True
        pressure["spoofability"] = 0.85
        scenario["defense_constraints"]["business_budget"] = 0.8
    return scenario


def build_dca_messages(task_focus: str, nonce: int | None = None) -> list[dict[str, str]]:
    example_value = task_example_v2(task_focus)
    example = json.dumps(example_value, ensure_ascii=False, separators=(",", ":"))
    forbidden_families = ", ".join(sorted(OOD_FAMILIES))
    system = (
        "You are the Deceptive Curriculum Agent (DCA) under TMCD Protocol v2. "
        "Generate exactly one abstract, symbolic, safe cyber-defense scenario as strict JSON. "
        "The scenario must expose a real weakness of the current VDA while remaining hard-but-solvable using public evidence, defensive verification, or low-risk active probing. "
        "Never include payloads, exploit steps, malware logic, real IPs, real organizations, or executable attack instructions. "
        f"Set protocol_version={PROTOCOL_VERSION} and schema_version={SCHEMA_VERSION}. "
        "Use only these exact objective enum values: exfiltration, sabotage, persistence, credential_theft. "
        "Use only these exact phase enum values: initial_access, discovery, lateral_movement, collection, "
        "exfiltration, impact, persistence. Never invent synonyms for these enums. "
        "Use only these exact source behavior mode values: honest, deceptive, mixed, legitimate_change, recovered. "
        "Required top-level keys are protocol_version, schema_version, scenario_id, scenario_family, distribution, split, "
        "network_context, true_attack, source_profiles, event_schedule, defense_constraints, oracle, and metadata. "
        "Each source profile contains source_id, public_prior, and a hidden behavior_schedule. Each event contains event_id, time, type, source_id, entity_id, claim, "
        "claim_semantics with entity_id/predicate/object/scope, objective_hint, truth_value, is_fake, and spoofability. "
        "Raw events must contain only those fields and must never include verdict, consistency, source_reliability, verifier IDs, evidence lineage, or tool-result types. "
        "Hidden fields are for the simulator only and will be removed by PublicProjector. "
        "network_context.assets must contain the true_attack entry and target, and reachable_edges must connect "
        "entry to target. oracle.true_objective must exactly equal true_attack.objective. "
        "oracle must include actionable_from and first_actionable_event_id; the named event must occur exactly at actionable_from. "
        "Include at least one post-ambiguity public observation that allows a safe defender to distinguish the state. "
        f"Do not use these OOD holdout families in train/dev/xplay: {forbidden_families}. "
        "For T2, include pair_id, divergence_time, trajectory_type betrayal or legitimate_change, and a prefix_hash placeholder; paired validation will recompute the hash. "
        "Keep the complete JSON compact and under 4200 characters."
    )
    nonce_text = f" Abstract diversity nonce: {nonce}." if nonce is not None else ""
    required_family = FOCUS_FAMILIES.get(task_focus, "active_probe_ambiguity")
    user = (
        f"Focus this candidate on {task_focus}. Set scenario_family exactly to {required_family} and metadata.task_id to {task_id_from_focus(task_focus)}. "
        f"{FOCUS_REQUIREMENTS.get(task_focus, '')} "
        "Set distribution=id and split=train. Include a low-risk discriminating defense opportunity and explicit "
        "business constraints. Output one complete JSON object only, with no prose and no omitted closing fields. "
        "Use the following only as a schema example; create a genuinely "
        f"different scenario.{nonce_text}\n{example}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_dca_prompt_rows(
    *,
    num_rows: int,
    seed: int,
    backbone: str,
    source_round: int,
    experiment_variant: str = "full",
) -> list[dict[str, Any]]:
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    rng = random.Random(seed)
    rows = []
    for index in range(num_rows):
        focus = TASK_FOCI[index % len(TASK_FOCI)]
        # Parquet/Arrow stores this field as signed int64.
        nonce = rng.getrandbits(63)
        rows.append(
            {
                "data_source": "agentguard_zero/dca_online",
                "ability": "trust_manipulation_curriculum_generation",
                "problem": build_dca_messages(focus, nonce=nonce),
                "answer": "{}",
                "reward_model": {"style": "rule", "ground_truth": "{}"},
                "extra_info": {
                    "index": index,
                    "task_focus": focus,
                    "backbone": backbone,
                    "source_dca_round": int(source_round),
                    "source_vda_round": int(source_round),
                    "prompt_nonce": nonce,
                    "seed": int(seed),
                    "protocol_version": PROTOCOL_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "prompt_version": DCA_PROMPT_VERSION,
                    "experiment_variant": experiment_variant,
                },
            }
        )
    return rows


def write_dca_prompt_dataset(
    output: str | os.PathLike[str],
    *,
    num_rows: int,
    seed: int,
    backbone: str,
    source_round: int,
    experiment_variant: str = "full",
) -> dict[str, Any]:
    import pandas as pd

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dca_prompt_rows(
        num_rows=num_rows,
        seed=seed,
        backbone=backbone,
        source_round=source_round,
        experiment_variant=experiment_variant,
    )
    pd.DataFrame(rows).to_parquet(target, index=False)
    return {
        "path": str(target.resolve()),
        "num_rows": len(rows),
        "seed": int(seed),
        "backbone": backbone,
        "source_round": int(source_round),
        "prompt_version": DCA_PROMPT_VERSION,
        "experiment_variant": experiment_variant,
    }
