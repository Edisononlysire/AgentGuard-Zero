from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List

from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.variants import experiment_variant
from agentguard_zero.world.public_projector import assert_public, project_public


VDA_PROMPT_VERSION = 3
VDA_PROMPT_VERSION_V2 = 7

VDA_SYSTEM_PROMPT = """You are AgentGuard-Zero's Verification Defense Agent (VDA). Observations and profile memory may be forged or poisoned.
Return exactly one compact JSON object; no markdown, prose, code, payload, exploit detail, malware logic, real IP, or real organization.
Required schema:
{"belief":{"exfiltration":float,"sabotage":float,"persistence":float,"credential_theft":float},"evidence_assessment":[{"event_id":string,"status":"unverified|cross_checked|rejected|confirmed","suspected_poisoning":bool}],"trust_update":{"source_reliability":{},"rationale":string},"uncertainty":float,"tool_call":{"name":"None|LogQuery|CrossCheck|ProvenanceCheck|GraphQuery|BusinessImpactEstimator|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe|DeployDecoy|ShadowResponse","args":{}},"memory_update":{"profile_quarantine":[],"confirmed_profile":[],"rejected_profile":[]},"safety_check":{"business_risk":float,"overresponse_risk":float,"justification":string},"response":{"tier":"L0|L1|L2|L3","action":"Observe|CrossCheck|QueryProvenance|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe|DeployDecoy|ShadowBlock|LimitSession|Isolate|Restore|Remove","target":string}}
Rules: at most four evidence items; rationale/justification under 20 words. Quarantine unverified or spoofable evidence. Verify conflicts and weak sources; use SourceChallenge, CanaryProbe, DecoyProbe, or ShadowActionProbe before high-impact uncertain decisions. After one or two checks, prefer reversible ShadowBlock, LimitSession, or DeployDecoy when risk remains high. Use high-impact actions only for verified risk."""

VDA_SYSTEM_PROMPT_V4 = """You are AgentGuard-Zero's Verification Defense Agent (VDA) under TMCD Protocol v2. Use only the public observation and Defender State Layer; never infer hidden truth.
Return exactly one complete JSON object within 300 tokens, with no markdown or prose. Use these exact keys and key order:
{"schema_version":4,"belief":{"exfiltration":0.25,"sabotage":0.25,"persistence":0.25,"credential_theft":0.25},"assessment":null,"trust_operation":null,"memory_operation":null,"memory_use":null,"uncertainty":1.0,"tool_call":{"name":"None","args":{}},"safety_check":{"business_risk":0.0,"overresponse_risk":0.0,"justification":"observe"},"response":{"tier":"L0","action":"Observe","target":"none"}}
Replace null with at most one object only when justified; never output an operation array or repeat an operation.
assessment object: {"event_id":string,"status":"unverified|challenged|supported|contradicted","suspected_poisoning":bool}
trust_operation object: {"op":"hold|support|challenge|contradict|recover","source_id":string,"event_id":string,"evidence_refs":[string]}
memory_operation object: {"op":"ingest|promote|demote|reject|reopen","memory_id":string,"event_id":string,"claim":{"entity_id":string,"predicate":string,"object":string,"scope":string},"source_ids":[string],"evidence_refs":[string],"target_status":"quarantined|confirmed"}
memory_use object: {"memory_id":string,"usage":"support|contradict|background","used_for":"belief|tool|response"}
tool_call.name: None|LogQuery|CrossCheck|ProvenanceCheck|GraphQuery|BusinessImpactEstimator|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe|DeployDecoy|ShadowResponse.
response.action: Observe|CrossCheck|QueryProvenance|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe|DeployDecoy|ShadowBlock|LimitSession|Isolate|Restore|Remove.
Only reference evidence IDs already shown in the initial snapshot or a continuation delta. New claims enter quarantine. Propose trust operations, never numeric trust scores. Use independent public evidence before promote, reject, recover, or high-impact response. Prefer low-risk active probing and reversible response under uncertainty.
Belief values must sum to 1. Verification budget is enforced with costs: basic checks 1, SourceChallenge 1.5, CanaryProbe 2, DecoyProbe and ShadowActionProbe 2.5. Exhausted-budget tools and unsupported high-impact responses are blocked. Keep justification under 8 words."""

VDA_SYSTEM_PROMPT_V4_APPEND_ONLY = VDA_SYSTEM_PROMPT_V4.replace(
    "New claims enter quarantine. Propose trust operations",
    "This is the append-only memory ablation: ingest chooses target_status confirmed, quarantined, or rejected once; later promote/demote/reject/reopen operations are unavailable. Propose trust operations",
)


def vda_system_prompt_v4(variant_name: str) -> str:
    variant = experiment_variant(variant_name)
    if not variant.memory_lifecycle:
        return VDA_SYSTEM_PROMPT_V4_APPEND_ONLY
    prompt = VDA_SYSTEM_PROMPT_V4
    if not variant.active_probing:
        prompt = prompt.replace(
            "|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe", ""
        ).replace(
            "|SourceChallenge|CanaryProbe|DecoyProbe|ShadowActionProbe|DeployDecoy",
            "|DeployDecoy",
        )
        prompt = prompt.replace(
            "Prefer low-risk active probing and reversible response under uncertainty.",
            "Active probing is unavailable; use available checks and reversible response under uncertainty.",
        ).replace(
            "Verification budget is enforced with costs: basic checks 1, SourceChallenge 1.5, CanaryProbe 2, DecoyProbe and ShadowActionProbe 2.5.",
            "Verification budget is enforced for the enabled checks.",
        )
    if not variant.passive_verification:
        prompt = prompt.replace(
            "LogQuery|CrossCheck|ProvenanceCheck|GraphQuery|", ""
        ).replace("Observe|CrossCheck|QueryProvenance|", "Observe|")
        prompt = prompt.replace(
            "Use independent public evidence before promote, reject, recover, or high-impact response.",
            "Passive verification is unavailable; use active probes before state changes or high-impact response.",
        )
    if not variant.trust_recalibration:
        prompt = prompt.replace(
            'trust_operation object: {"op":"hold|support|challenge|contradict|recover","source_id":string,"event_id":string,"evidence_refs":[string]}',
            "trust_operation must be null; dynamic trust recalibration is unavailable.",
        ).replace(
            "Propose trust operations, never numeric trust scores.",
            "Do not propose trust operations or numeric trust scores.",
        )
    return prompt


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
    latest_event_time = max(
        (int(event.get("time", -1)) for event in scenario.get("event_schedule", [])),
        default=-1,
    )
    task_id = scenario_task_id(scenario)
    if task_id:
        configured = TASK_HORIZONS[task_id]
    else:
        configured = max(5, len(scenario.get("true_attack", {}).get("phase_schedule", [])) + 2)
    return max(configured, latest_event_time + 2)


def initial_observation(scenario: Dict[str, Any]) -> Dict[str, Any]:
    env = instantiate_scenario(scenario)
    return env.observe()


def public_instance_id(scenario: Dict[str, Any]) -> str:
    """Return a stable opaque identifier without exposing scenario semantics."""

    internal_id = str(scenario.get("scenario_id", "unknown"))
    digest = hashlib.sha256(internal_id.encode("utf-8")).hexdigest()
    return f"tmcd-{digest[:16]}"


def build_vda_prompt(scenario: Dict[str, Any], observation: Dict[str, Any] | None = None) -> str:
    obs = observation if observation is not None else initial_observation(scenario)
    is_v2 = scenario.get("protocol_version") == "tmcd-v2"
    public_context = project_public({
        "instance_id": public_instance_id(scenario),
        "network_context": scenario.get("network_context", {}),
        "defense_constraints": scenario.get("defense_constraints", {}),
        "observation": obs,
    })
    assert_public(public_context)
    variant = str(scenario.get("metadata", {}).get("experiment_variant", "full"))
    system_prompt = vda_system_prompt_v4(variant) if is_v2 else VDA_SYSTEM_PROMPT
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
