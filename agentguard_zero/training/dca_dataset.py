from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from agentguard_zero.schemas.scenario_schema import minimal_example


TASK_FOCI = (
    "T1 Defensive Active Probing",
    "T2 Long-Horizon Trust-Building Betrayal",
    "T3 Profile and Memory Poisoning",
    "T4 Business-Constrained Overresponse Induction",
)

DCA_PROMPT_VERSION = 2


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


def build_dca_messages(task_focus: str, nonce: int | None = None) -> list[dict[str, str]]:
    example_value = minimal_example()
    example_value.update(
        {
            "actors": ["trusted_source", "defender"],
            "collaboration_context": "shared abstract defense workflow",
            "trust_building_plan": ["truthful_claim_t0", "truthful_claim_t1"],
            "betrayal_trigger": "high-impact false claim at t2",
            "source_reliability": {"trusted_source": 0.8},
            "strategy_switch": {"time": 2, "from": "truthful", "to": "deceptive"},
            "metadata": {"task_id": "T1", "active_probe": "SourceChallenge"},
        }
    )
    example = json.dumps(example_value, ensure_ascii=False, separators=(",", ":"))
    system = (
        "You are the Deceptive Curriculum Agent (DCA) in AgentGuard-Zero. "
        "Generate exactly one abstract, symbolic, safe cyber-defense scenario as strict JSON. "
        "The scenario must be hard for the current VDA but solvable through defensive verification or active probing. "
        "Never include payloads, exploit steps, malware logic, real IPs, real organizations, or executable attack instructions. "
        "Use only these exact objective enum values: exfiltration, sabotage, persistence, credential_theft. "
        "Use only these exact phase enum values: initial_access, discovery, lateral_movement, collection, "
        "exfiltration, impact, persistence. Use only these exact fake-evidence type values: fake_log, fake_ttp, "
        "decoy_touch, suppressed_signal. Never invent synonyms for these enums. "
        "The first six top-level keys must be scenario_id, network_context, true_attack, poisoning_plan, "
        "defense_constraints, and oracle. Then include actors, collaboration_context, trust_building_plan, "
        "betrayal_trigger, source_reliability, strategy_switch, and metadata. "
        "network_context.assets must contain the true_attack entry and target, and reachable_edges must connect "
        "entry to target. oracle.true_objective must exactly equal true_attack.objective. The poisoning false "
        "objective must be a different allowed objective. Keep the complete JSON compact and under 1800 characters."
    )
    nonce_text = f" Abstract diversity nonce: {nonce}." if nonce is not None else ""
    user = (
        f"Focus this candidate on {task_focus}. {FOCUS_REQUIREMENTS.get(task_focus, '')} "
        "Include a low-risk discriminating defense opportunity and explicit "
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
) -> dict[str, Any]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dca_prompt_rows(
        num_rows=num_rows,
        seed=seed,
        backbone=backbone,
        source_round=source_round,
    )
    pd.DataFrame(rows).to_parquet(target, index=False)
    return {
        "path": str(target.resolve()),
        "num_rows": len(rows),
        "seed": int(seed),
        "backbone": backbone,
        "source_round": int(source_round),
        "prompt_version": DCA_PROMPT_VERSION,
    }
