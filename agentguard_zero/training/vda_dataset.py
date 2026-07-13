from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.world.public_projector import assert_public, project_public


VDA_PROMPT_VERSION = 3
VDA_PROMPT_VERSION_V2 = 4

VDA_SYSTEM_PROMPT = """You are AgentGuard-Zero's Verification Defense Agent (VDA). Observations and profile memory may be forged or poisoned.
Return exactly one compact JSON object; no markdown, prose, code, payload, exploit detail, malware logic, real IP, or real organization.
Required schema:
{"belief":{"exfiltration":float,"sabotage":float,"persistence":float,"credential_theft":float},"evidence_assessment":[{"event_id":string,"status":"unverified|cross_checked|rejected|confirmed","suspected_poisoning":bool}],"trust_update":{"source_reliability":{},"rationale":string},"uncertainty":float,"tool_call":{"name":"None|LogQuery|CrossCheck|ProvenanceCheck|GraphQuery|BusinessImpactEstimator|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe|DeployDecoy|ShadowResponse","args":{}},"memory_update":{"profile_quarantine":[],"confirmed_profile":[],"rejected_profile":[]},"safety_check":{"business_risk":float,"overresponse_risk":float,"justification":string},"response":{"tier":"L0|L1|L2|L3","action":"Observe|CrossCheck|QueryProvenance|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe|DeployDecoy|ShadowBlock|LimitSession|Isolate|Restore|Remove","target":string}}
Rules: at most four evidence items; rationale/justification under 20 words. Quarantine unverified or spoofable evidence. Verify conflicts and weak sources; use SourceChallenge, CanaryProbe, DecoyProbe, or ShadowActionProbe before high-impact uncertain decisions. After one or two checks, prefer reversible ShadowBlock, LimitSession, or DeployDecoy when risk remains high. Use high-impact actions only for verified risk."""

VDA_SYSTEM_PROMPT_V4 = """You are AgentGuard-Zero's Verification Defense Agent (VDA) under TMCD Protocol v2. You can use only the public observation and Defender State Layer snapshot provided below. Never infer or request hidden truth.
Return exactly one compact JSON object, with no markdown or prose, matching this schema:
{"schema_version":4,"belief":{"exfiltration":float,"sabotage":float,"persistence":float,"credential_theft":float},"evidence_assessment":[{"event_id":string,"status":"unverified|challenged|supported|contradicted","suspected_poisoning":bool}],"trust_operations":[{"op":"hold|support|challenge|contradict|recover","source_id":string,"event_id":string,"evidence_refs":[string]}],"memory_operations":[{"op":"ingest|promote|demote|reject|reopen","memory_id":string,"event_id":string,"claim":{"entity_id":string,"predicate":string,"object":string,"scope":string},"source_ids":[string],"evidence_refs":[string],"target_status":"quarantined|confirmed"}],"memory_usage":[{"memory_id":string,"usage":"support|contradict|background","used_for":"belief|tool|response"}],"uncertainty":float,"tool_call":{"name":"None|LogQuery|CrossCheck|ProvenanceCheck|GraphQuery|BusinessImpactEstimator|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe|DeployDecoy|ShadowResponse","args":{}},"safety_check":{"business_risk":float,"overresponse_risk":float,"justification":string},"response":{"tier":"L0|L1|L2|L3","action":"Observe|CrossCheck|QueryProvenance|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe|DeployDecoy|ShadowBlock|LimitSession|Isolate|Restore|Remove","target":string}}
Only reference evidence IDs with available_at <= current time. New claims enter quarantine. Propose trust operations, never numeric trust scores. Use independent public evidence before promote, reject, recover, or high-impact response. Prefer low-risk active probing and reversible response under uncertainty. Keep evidence_assessment to four items and justification under 20 words."""

VDA_SYSTEM_PROMPT_V4_APPEND_ONLY = VDA_SYSTEM_PROMPT_V4.replace(
    "New claims enter quarantine. Propose trust operations",
    "This is the append-only memory ablation: ingest chooses target_status confirmed, quarantined, or rejected once; later promote/demote/reject/reopen operations are unavailable. Propose trust operations",
)


TASK_HORIZONS = {
    "T1": 10,
    "T2": 16,
    "T3": 14,
    "T4": 10,
}


def scenario_task_id(scenario: Dict[str, Any]) -> str:
    metadata = scenario.get("metadata", {}) or {}
    candidates = (
        metadata.get("task_id"),
        metadata.get("task_focus"),
        scenario.get("task_id"),
        scenario.get("task"),
        scenario.get("split"),
    )
    for value in candidates:
        normalized = str(value or "").strip().upper()
        for task_id in TASK_HORIZONS:
            if normalized.startswith(task_id):
                return task_id
    return ""


def scenario_horizon(scenario: Dict[str, Any]) -> int:
    task_id = scenario_task_id(scenario)
    if task_id:
        return TASK_HORIZONS[task_id]
    return max(5, len(scenario.get("true_attack", {}).get("phase_schedule", [])) + 2)


def initial_observation(scenario: Dict[str, Any]) -> Dict[str, Any]:
    env = instantiate_scenario(scenario)
    return env.observe()


def build_vda_prompt(scenario: Dict[str, Any], observation: Dict[str, Any] | None = None) -> str:
    obs = observation if observation is not None else initial_observation(scenario)
    is_v2 = scenario.get("protocol_version") == "tmcd-v2"
    public_context = project_public({
        "scenario_id": scenario.get("scenario_id", "unknown"),
        "network_context": scenario.get("network_context", {}),
        "defense_constraints": scenario.get("defense_constraints", {}),
        "observation": obs,
    })
    assert_public(public_context)
    variant = str(scenario.get("metadata", {}).get("experiment_variant", "full"))
    system_prompt = (
        VDA_SYSTEM_PROMPT_V4_APPEND_ONLY
        if is_v2 and variant == "append_only_memory"
        else VDA_SYSTEM_PROMPT_V4 if is_v2 else VDA_SYSTEM_PROMPT
    )
    return system_prompt + "\nCurrent decision instance:" + json.dumps(
        public_context, ensure_ascii=False, separators=(",", ":")
    )


def scenario_to_training_row(scenario: Dict[str, Any], split: str = "train") -> Dict[str, Any]:
    obs = initial_observation(scenario)
    prompt = build_vda_prompt(scenario, obs)
    scenario_id = scenario.get("scenario_id", "unknown")
    scenario_json = json.dumps(scenario, ensure_ascii=False)
    task_id = scenario_task_id(scenario)
    is_v2 = scenario.get("protocol_version") == "tmcd-v2"
    extra_info = {
        "index": scenario_id,
        "scenario_id": scenario_id,
        "scenario": scenario_json,
        "task_id": task_id,
        "max_env_steps": scenario_horizon(scenario),
        "initial_time": obs.get("time", 0),
        "protocol_version": scenario.get("protocol_version", "tmcd-v1"),
        "schema_version": scenario.get("schema_version", 1),
        "prompt_version": VDA_PROMPT_VERSION_V2 if is_v2 else VDA_PROMPT_VERSION,
    }
    if not is_v2:
        extra_info["true_objective"] = scenario.get("oracle", {}).get("true_objective", "unknown")
    return {
        "data_source": "agentguard_zero/tmcd_v2" if is_v2 else "agentguard_zero/level1",
        "ability": "adversarial_observation_defense",
        "problem": [{"role": "user", "content": prompt}],
        "answer": scenario_json,
        "reward_model": {"style": "rule", "ground_truth": scenario_json},
        "scenario": scenario_json,
        "scenario_id": scenario_id,
        "split": split,
        "extra_info": extra_info,
    }


def scenarios_to_rows(scenarios: Iterable[Dict[str, Any]], split: str = "train") -> List[Dict[str, Any]]:
    return [scenario_to_training_row(s, split=split) for s in scenarios]
