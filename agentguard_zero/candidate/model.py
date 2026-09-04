"""Qwen-backed multi-head scorer for public-state action candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from agentguard_zero.candidate.branching import (
    BUDGET_TARGET_NAMES,
    LOCAL_PHASE_INDICES_V2,
    LONG_PHASE_INDICES_V2,
    MEMORY_OPERATION_NAMES,
    PHASE_NAMES,
    PHASE_NAMES_V2,
    SHARED_LOCAL_LONG_ARCHITECTURE,
    SHARED_LOCAL_LONG_ARCHITECTURES,
    SHARED_LOCAL_LONG_ARCHITECTURE_V2,
    phase_branch_probabilities,
)
from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.world.public_projector import assert_public, project_public


FORMAT_VERSION = "candidate_pair_v8_candidate_final_balanced"
OUTCOME_HEAD_NAMES = (
    "information_gain",
    "terminal_attack_mitigation",
    "business_cost",
    "overresponse_cost",
    "memory_dependent_utility_gain",
    "final_safe_success",
    "trajectory_safe_utility",
)
V9_OUTCOME_SCORE_WEIGHTS = {
    "trajectory_safe_utility": 1.0,
    "final_safe_success": 0.5,
    "terminal_attack_mitigation": 0.5,
    "information_gain": 0.25,
    "memory_dependent_utility_gain": 0.25,
    "business_cost": -0.5,
    "overresponse_cost": -0.5,
}
RESIDUAL_HEAD_NAMES = ("utility_residual", "family_residual")
HEAD_NAMES = (
    "utility",
    "family",
    "support",
    "belief",
    "uncertainty",
    "probe_value",
    "business_risk",
    "safety_risk",
    *OUTCOME_HEAD_NAMES,
    *RESIDUAL_HEAD_NAMES,
)
BRANCHED_HEAD_NAMES = (
    "phase_gate",
    "phase_context",
    "local_utility",
    "local_family",
    "t3_memory_utility",
    "t3_memory_family",
    "t3_memory_operation",
    "t4_budget_utility",
    "t4_budget_family",
    "t4_budget_state",
)


def public_state_text(
    observation: Mapping[str, Any], *, memory_tier_encoding: bool = False
) -> str:
    """Encode the public state without leaking a candidate into the family gate."""

    public = project_public(dict(observation))
    assert_public(public)
    aliases = _public_aliases(public)
    return (
        "Choose the safest long-term defense action family using public data.\n"
        f"Public state: {_compact_public_text(public, (), aliases, memory_tier_encoding=memory_tier_encoding)}"
    )


def candidate_pair_text(
    observation: Mapping[str, Any],
    candidate: CandidateOption,
    *,
    memory_tier_encoding: bool = False,
) -> str:
    public = project_public(dict(observation))
    assert_public(public)
    aliases = _public_aliases(public)
    summary = _replace_aliases(candidate.public_summary, aliases)
    references = _replace_aliases(list(candidate.referenced_ids), aliases)
    effect = _compact_effect_text(candidate.compiled_packet, aliases)
    context = _compact_public_text(
        public,
        candidate.referenced_ids,
        aliases,
        memory_tier_encoding=memory_tier_encoding,
    )
    memory_role = (
        f"\nCandidate memory role: {_candidate_memory_role(candidate)}"
        if memory_tier_encoding
        else ""
    )
    return (
        "Rank this legal defense candidate for safe long-term utility using public data.\n"
        f"Public state: {context}\n"
        f"Candidate: {summary}; refs={','.join(references) or 'none'}\n"
        f"Candidate effect: {effect}{memory_role}"
    )


def _candidate_memory_role(candidate: CandidateOption) -> str:
    """Expose whether a candidate consumes working or durable evidence state."""

    packet = candidate.compiled_packet
    tool = packet.get("tool_call") or {}
    tool_name = str(tool.get("name", "None")) if isinstance(tool, Mapping) else "None"
    memory_operations = packet.get("memory_operations") or packet.get("memory_operation") or []
    memory_usage = packet.get("memory_usage") or packet.get("memory_use") or []
    if memory_operations:
        return "LTM_WRITE_OR_TRANSITION:evidence_gated"
    if memory_usage:
        return (
            "LTM_RETRIEVAL_USE:respect_status;confirmed_supports;"
            "quarantined_or_rejected_only_contradicts;must_be_action_relevant"
        )
    if tool_name != "None":
        return "STM_UPDATE:probe_or_verification_result_available_next_step"
    response = packet.get("response") or {}
    if isinstance(response, Mapping) and str(response.get("action", "Observe")) != "Observe":
        return "STM_GROUNDED_RESPONSE:check_recent_probe_and_preconditions"
    return "NO_MEMORY_EFFECT"


def _brief_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _claim_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _brief_number(value)
    return "/".join(
        _brief_number(value.get(key, ""))
        for key in ("entity_id", "predicate", "object")
    )


def _compact_mapping_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return "(" + ",".join(
            f"{key}={_compact_mapping_text(item)}"
            for key, item in sorted(value.items())
        ) + ")"
    if isinstance(value, list):
        return "[" + ",".join(_compact_mapping_text(item) for item in value) + "]"
    return _brief_number(value)


def _compact_effect_text(packet: Mapping[str, Any], aliases: Mapping[str, str]) -> str:
    parts: list[str] = []
    belief = packet.get("belief") or {}
    if isinstance(belief, Mapping):
        parts.append(
            "belief="
            + _compact_mapping_text(
                {key: _brief_number(value) for key, value in sorted(belief.items())}
            )
        )
    if "uncertainty" in packet:
        parts.append(f"uncertainty={_brief_number(packet['uncertainty'])}")
    tool = packet.get("tool_call") or {}
    if isinstance(tool, Mapping) and str(tool.get("name", "None")) != "None":
        args = _replace_aliases(tool.get("args") or {}, aliases)
        parts.append(f"tool={tool.get('name')}{_compact_mapping_text(args)}")
    response = packet.get("response") or {}
    if isinstance(response, Mapping) and str(response.get("action", "Observe")) != "Observe":
        parts.append(
            "response="
            + ":".join(
                _brief_number(_replace_aliases(response.get(key, ""), aliases))
                for key in ("action", "target", "tier")
            )
        )
    for key, label in (
        ("trust_operations", "trust"),
        ("trust_operation", "trust"),
        ("memory_operations", "memory_op"),
        ("memory_operation", "memory_op"),
        ("memory_usage", "memory_use"),
        ("memory_use", "memory_use"),
    ):
        values = packet.get(key) or []
        if isinstance(values, Mapping):
            values = [values]
        for value in values:
            parts.append(
                f"{label}={_compact_mapping_text(_replace_aliases(value, aliases))}"
            )
    return "|".join(parts) or "observe"


def _compact_public_text(
    public: Mapping[str, Any],
    referenced_ids: Sequence[str],
    aliases: Mapping[str, str],
    *,
    memory_tier_encoding: bool = False,
) -> str:
    state = _replace_aliases(
        _compact_public_state(public, referenced_ids), aliases
    )
    parts: list[str] = []
    time_value = state.get("time", state.get("t", state.get("step", 0)))
    defense = state.get("defense_context") or {}
    header = [f"t={_brief_number(time_value)}"]
    if isinstance(defense, Mapping):
        for key, label in (
            ("remaining_business_budget", "biz"),
            ("remaining_high_impact_actions", "high"),
            ("remaining_verification_budget", "verify"),
            ("remaining_active_probe_budget", "probe"),
        ):
            if key in defense:
                header.append(f"{label}={_brief_number(defense[key])}")
        requirements = defense.get("response_requirements") or {}
        enabled = [
            str(key).removesuffix("_required_for_mitigation")
            for key, value in requirements.items()
            if value
        ]
        if enabled:
            header.append("required=" + ",".join(sorted(enabled)))
        assets = []
        for asset in defense.get("public_assets", []) or []:
            if isinstance(asset, Mapping):
                assets.append(
                    ":".join(
                        _brief_number(asset.get(key, ""))
                        for key in ("id", "zone", "criticality")
                    )
                )
        if assets:
            header.append("assets=" + ",".join(assets))
    parts.append(";".join(header))

    events = []
    for event in state.get("observed_events", []) or []:
        if isinstance(event, Mapping):
            events.append(
                ":".join(
                    (
                        _brief_number(event.get("event_id", "")),
                        _brief_number(event.get("source_id", event.get("source", ""))),
                        _brief_number(event.get("type", "")),
                        _claim_text(event.get("claim_semantics") or {}),
                        _brief_number(event.get("objective_hint", "")),
                    )
                )
            )
    if events:
        parts.append("events=" + ";".join(events))

    evidence_rows = []
    for evidence in state.get("available_evidence", []) or []:
        if not isinstance(evidence, Mapping):
            continue
        content = evidence.get("content") or {}
        evidence_rows.append(
            ":".join(
                (
                    _brief_number(evidence.get("evidence_id", "")),
                    _brief_number(evidence.get("event_id", "")),
                    _brief_number(evidence.get("source_id", "")),
                    _brief_number(evidence.get("evidence_origin", "")),
                    _claim_text(content.get("claim_semantics") or {}),
                    _brief_number(content.get("objective_hint", "")),
                    _brief_number(content.get("verdict", "")),
                    _brief_number(content.get("consistency_signal", "")),
                    _brief_number(content.get("provenance_signal", "")),
                )
            ).rstrip(":")
        )
    if evidence_rows:
        parts.append("evidence=" + ";".join(evidence_rows))

    defender = state.get("defender_state") or {}
    trust = defender.get("trust") or {}
    sources = [
        ":".join(
            _brief_number(row.get(key, ""))
            for key in ("source_id", "mean", "status", "uncertainty")
        )
        for row in trust.get("sources", []) or []
        if isinstance(row, Mapping)
    ]
    if sources:
        parts.append("trust_sources=" + ";".join(sources))
    claims = [
        ":".join(
            _brief_number(row.get(key, ""))
            for key in ("event_id", "source_id", "score", "status")
        )
        for row in trust.get("claims", []) or []
        if isinstance(row, Mapping)
    ]
    if claims:
        parts.append("trust_claims=" + ";".join(claims))

    memory = defender.get("memory") or {}
    memory_rows = []
    for row in memory.get("records", []) or []:
        if isinstance(row, Mapping):
            memory_rows.append(
                ":".join(
                    (
                        _brief_number(row.get("memory_id", "")),
                        _brief_number(row.get("status", "")),
                        _brief_number(row.get("confidence", "")),
                        _claim_text(row.get("claim") or {}),
                        ",".join(map(str, row.get("source_ids", []) or [])),
                        ",".join(map(str, row.get("evidence_refs", []) or [])),
                        _brief_number(row.get("version", "")),
                        _compact_mapping_text(row.get("transition_history", []) or []),
                        _compact_mapping_text(row.get("last_used_for_action") or {}),
                    )
                )
            )
    if memory_rows:
        label = "long_term_evidence_memory" if memory_tier_encoding else "memory"
        parts.append(label + "=" + ";".join(memory_rows))
    profile_rows = []
    for row in defender.get("profile_memory", []) or []:
        if isinstance(row, Mapping):
            profile_rows.append(_compact_mapping_text(row))
    if profile_rows:
        parts.append("long_term_profile_memory=" + ";".join(profile_rows))
    if defender.get("business_impact_memory"):
        parts.append(
            "long_term_business_impact_memory="
            + _compact_mapping_text(defender["business_impact_memory"])
        )
    if defender.get("probe_state"):
        label = "short_term_probe_results" if memory_tier_encoding else "probes"
        parts.append(label + "=" + _compact_mapping_text(defender["probe_state"]))
    if defender.get("response_history"):
        label = "short_term_response_history" if memory_tier_encoding else "responses"
        parts.append(label + "=" + _compact_mapping_text(defender["response_history"]))
    last = state.get("last_tool_result") or {}
    if last and str(last.get("tool", "None")) != "None":
        label = "short_term_last_tool_result" if memory_tier_encoding else "last"
        parts.append(label + "=" + _compact_mapping_text(last))
    if memory_tier_encoding:
        parts.append(
            "memory_rule=STM_is_episode_local_and_automatic;"
            "LTM_requires_evidence_gated_write_transition_retrieval_use;"
            "T3_requires_versioned_evidence_lifecycle;"
            "T4_requires_profile_and_business_history_use;"
            "do_not_substitute_LTM_for_missing_probe_preconditions"
        )
    return "\n".join(parts)


def _public_aliases(public: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, set[str]] = {
        "E": set(),
        "V": set(),
        "S": set(),
        "M": set(),
        "A": set(),
        "Z": set(),
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                prefix = {
                    "event_id": "E",
                    "evidence_id": "V",
                    "source_id": "S",
                    "source": "S",
                    "memory_id": "M",
                }.get(str(key))
                if prefix and isinstance(item, str) and item:
                    values[prefix].add(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(public)
    for asset in (public.get("defense_context") or {}).get("public_assets", []) or []:
        if isinstance(asset, Mapping):
            if str(asset.get("id", "")):
                values["A"].add(str(asset["id"]))
            if str(asset.get("zone", "")):
                values["Z"].add(str(asset["zone"]))
    return {
        raw: f"{prefix}{index}"
        for prefix, raw_values in values.items()
        for index, raw in enumerate(sorted(raw_values))
    }


def _replace_aliases(value: Any, aliases: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {key: _replace_aliases(item, aliases) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_aliases(item, aliases) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_aliases(item, aliases) for item in value)
    if isinstance(value, str):
        result = value
        for raw in sorted(aliases, key=len, reverse=True):
            result = result.replace(raw, aliases[raw])
        return result
    return value


def _compact_event(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        key: value[key]
        for key in (
            "event_id",
            "source_id",
            "source",
            "type",
            "claim_semantics",
            "objective_hint",
        )
        if key in value
    }


def _compact_evidence(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    content = value.get("content") or {}
    compact_content = {
        key: content[key]
        for key in (
            "claim_semantics",
            "objective_hint",
            "verdict",
            "consistency_signal",
            "provenance_signal",
        )
        if isinstance(content, Mapping) and key in content
    }
    result = {
        key: value[key]
        for key in (
            "evidence_id",
            "event_id",
            "source_id",
            "evidence_origin",
            "available_at",
        )
        if key in value
    }
    if compact_content:
        result["content"] = compact_content
    return result


def _compact_defender(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    trust = value.get("trust") or value.get("trust_state") or {}
    if isinstance(trust, Mapping):
        sources = trust.get("source_reputation") or {}
        claims = trust.get("current_claim_trust") or {}
        result["trust"] = {
            "sources": [
                {"source_id": source_id, **dict(row)}
                for source_id, row in sorted(sources.items())
                if isinstance(row, Mapping)
            ],
            "claims": [
                {"event_id": event_id, **dict(row)}
                for event_id, row in sorted(claims.items())
                if isinstance(row, Mapping)
            ][-8:],
        }
    memory = value.get("memory") or value.get("memory_state") or {}
    if isinstance(memory, Mapping):
        records = []
        profiles = []
        for bucket in (
            "retrieved_confirmed",
            "retrieved_quarantined",
            "rejected_warnings",
            "retrieved_profiles",
        ):
            for row in memory.get(bucket, []) or []:
                if isinstance(row, Mapping):
                    keys = (
                        "memory_id",
                        "memory_kind",
                        "status",
                        "claim",
                        "confidence",
                        "source_ids",
                        "source_id",
                        "evidence_refs",
                        "created_at",
                        "updated_at",
                        "last_updated_at",
                        "version",
                        "transition_history",
                        "retrieval_count",
                        "last_retrieved_at",
                        "usage_counts",
                        "acceptance_count",
                        "last_used_for_action",
                        "historical_accuracy",
                        "recent_accuracy",
                        "observed_claim_count",
                        "independent_confirmations",
                        "recent_contradictions",
                        "trust_trend",
                        "suspected_deception_phase",
                    )
                    compact = {key: row[key] for key in keys if key in row}
                    if row.get("memory_kind") == "source_profile":
                        profiles.append(compact)
                    else:
                        records.append(compact)
        result["memory"] = {
            "retrieved_memory_ids": list(memory.get("retrieved_memory_ids") or []),
            "records": records,
        }
        if profiles:
            result["profile_memory"] = profiles
    if value.get("probe_state"):
        result["probe_state"] = value["probe_state"]
    if value.get("response_history"):
        result["response_history"] = list(value["response_history"])[-8:]
    if value.get("business_impact_memory"):
        result["business_impact_memory"] = dict(
            value["business_impact_memory"]
        )
    return result


def _compact_public_state(
    public: Mapping[str, Any], referenced_ids: Sequence[str]
) -> dict[str, Any]:
    """Keep decision evidence and lifecycle state while dropping protocol noise."""

    references = set(map(str, referenced_ids))
    result: dict[str, Any] = {}
    for key in ("t", "time", "step", "max_steps", "defense_context"):
        if key in public:
            result[key] = public[key]
    for key in ("observed_events", "available_evidence"):
        values = list(public.get(key) or [])
        selected = []
        for index, value in enumerate(values):
            row = dict(value) if isinstance(value, Mapping) else value
            row_id = ""
            if isinstance(row, Mapping):
                row_id = str(
                    row.get("event_id") or row.get("evidence_id") or row.get("id") or ""
                )
            if row_id in references or index >= max(0, len(values) - 8):
                selected.append(
                    _compact_event(row) if key == "observed_events" else _compact_evidence(row)
                )
        if selected:
            result[key] = selected
    defender = public.get("defender_state")
    if isinstance(defender, Mapping):
        result["defender_state"] = _compact_defender(defender)
    if public.get("last_tool_result"):
        result["last_tool_result"] = public["last_tool_result"]
    return result


def load_ranker_components(
    *,
    model_path: str | Path,
    adapter_path: str | Path | None = None,
    heads_path: str | Path | None = None,
    score_head_path: str | Path | None = None,
    trainable: bool = False,
    dtype: str = "bf16",
    architecture: str = "legacy",
    separate_local_long_adapters: bool = False,
) -> tuple[Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - GPU dependency gate
        raise RuntimeError("candidate ranker requires torch, transformers, and peft") from exc

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    backbone.config.use_cache = False
    if adapter_path:
        adapter_path = Path(adapter_path)
        serialized_dual = all(
            (adapter_path / name / "adapter_config.json").is_file()
            for name in ("local", "long")
        )
        if serialized_dual:
            backbone = PeftModel.from_pretrained(
                backbone,
                str(adapter_path / "local"),
                adapter_name="local",
                is_trainable=trainable,
            )
            backbone.load_adapter(
                str(adapter_path / "long"),
                adapter_name="long",
                is_trainable=trainable,
            )
            separate_local_long_adapters = True
        elif separate_local_long_adapters:
            # The fair main run initializes two independent adapters from the
            # same D0 bytes.  Neither branch inherits the other branch's
            # updates, and the frozen base remains shared.
            backbone = PeftModel.from_pretrained(
                backbone,
                str(adapter_path),
                adapter_name="local",
                is_trainable=trainable,
            )
            backbone.load_adapter(
                str(adapter_path),
                adapter_name="long",
                is_trainable=trainable,
            )
        else:
            backbone = PeftModel.from_pretrained(
                backbone, str(adapter_path), is_trainable=trainable
            )
    elif trainable:
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                task_type=TaskType.FEATURE_EXTRACTION,
            ),
        )
    if separate_local_long_adapters:
        if architecture != SHARED_LOCAL_LONG_ARCHITECTURE_V2:
            raise ValueError("separate local/long adapters require shared_local_long_v2")
        if not all(name in backbone.peft_config for name in ("local", "long")):
            raise RuntimeError("local/long adapter initialization is incomplete")
        backbone.agentguard_adapter_strategy = "separate_local_long"
        backbone.agentguard_adapter_names = ("local", "long")
        backbone.agentguard_adapters_trainable = bool(trainable)
        activate_ranker_adapter(backbone, "local")
    hidden_size = int(
        getattr(backbone.config, "hidden_size", 0)
        or getattr(backbone.config, "text_config", {}).hidden_size
    )
    if architecture not in {"legacy", *SHARED_LOCAL_LONG_ARCHITECTURES}:
        raise ValueError(f"unsupported candidate ranker architecture: {architecture}")
    linear_heads = {
        "utility": torch.nn.Linear(hidden_size, 1),
        "family": torch.nn.Linear(hidden_size, 6),
        "support": torch.nn.Linear(hidden_size, 6),
        "belief": torch.nn.Linear(hidden_size, 4),
        "uncertainty": torch.nn.Linear(hidden_size, 1),
        "probe_value": torch.nn.Linear(hidden_size, 1),
        "business_risk": torch.nn.Linear(hidden_size, 1),
        "safety_risk": torch.nn.Linear(hidden_size, 1),
        **{
            name: torch.nn.Linear(hidden_size, 1)
            for name in OUTCOME_HEAD_NAMES
        },
    }
    if architecture in SHARED_LOCAL_LONG_ARCHITECTURES:
        phase_names = (
            PHASE_NAMES_V2
            if architecture == SHARED_LOCAL_LONG_ARCHITECTURE_V2
            else PHASE_NAMES
        )
        linear_heads.update(
            {
                "phase_gate": torch.nn.Linear(hidden_size, len(phase_names)),
                "phase_context": torch.nn.Linear(
                    len(phase_names), hidden_size, bias=False
                ),
                "local_utility": torch.nn.Linear(hidden_size, 1),
                "local_family": torch.nn.Linear(hidden_size, 6),
                "t3_memory_utility": torch.nn.Linear(hidden_size, 1),
                "t3_memory_family": torch.nn.Linear(hidden_size, 6),
                "t3_memory_operation": torch.nn.Linear(
                    hidden_size, len(MEMORY_OPERATION_NAMES)
                ),
                "t4_budget_utility": torch.nn.Linear(hidden_size, 1),
                "t4_budget_family": torch.nn.Linear(hidden_size, 6),
                "t4_budget_state": torch.nn.Linear(
                    hidden_size, len(BUDGET_TARGET_NAMES)
                ),
            }
        )
    for head in linear_heads.values():
        torch.nn.init.normal_(head.weight, mean=0.0, std=0.02)
        if head.bias is not None:
            torch.nn.init.zeros_(head.bias)
    residual_width = min(512, max(128, hidden_size // 8))

    def residual_head(output_size: int) -> Any:
        module = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, residual_width),
            torch.nn.GELU(),
            torch.nn.Linear(residual_width, output_size),
        )
        torch.nn.init.normal_(module[1].weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(module[1].bias)
        # Preserve byte-for-byte legacy scoring at initialization. The final
        # projection begins learning immediately; gradients then reach the
        # hidden projection on the following optimizer step.
        torch.nn.init.zeros_(module[3].weight)
        torch.nn.init.zeros_(module[3].bias)
        return module

    heads = torch.nn.ModuleDict(
        {
            **linear_heads,
            "utility_residual": residual_head(1),
            "family_residual": residual_head(6),
        }
    )
    heads.agentguard_architecture = architecture
    if heads_path:
        state = torch.load(str(heads_path), map_location="cpu", weights_only=True)
        incompatible = heads.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = set(incompatible.missing_keys)
        allowed_missing = {
            "support.weight",
            "support.bias",
            *{
                f"{name}.{parameter}"
                for name in RESIDUAL_HEAD_NAMES
                for parameter in (
                    "0.weight",
                    "0.bias",
                    "1.weight",
                    "1.bias",
                    "3.weight",
                    "3.bias",
                )
            },
            *{
                f"{name}.{parameter}"
                for name in OUTCOME_HEAD_NAMES
                for parameter in ("weight", "bias")
            },
        }
        if architecture in SHARED_LOCAL_LONG_ARCHITECTURES:
            allowed_missing.update(
                key for key in heads.state_dict() if key.split(".", 1)[0] in BRANCHED_HEAD_NAMES
            )
        if unexpected or not missing.issubset(allowed_missing):
            raise RuntimeError(
                f"incompatible candidate heads: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        # D0 predates the branched architecture. Its local calibration is a
        # valid common warm-start, while every long-horizon head remains new.
        if architecture in SHARED_LOCAL_LONG_ARCHITECTURES and {
            "local_utility.weight",
            "local_utility.bias",
            "local_family.weight",
            "local_family.bias",
        }.issubset(missing):
            heads["local_utility"].load_state_dict(heads["utility"].state_dict())
            heads["local_family"].load_state_dict(heads["family"].state_dict())
    elif score_head_path:
        state = torch.load(str(score_head_path), map_location="cpu", weights_only=True)
        heads["utility"].load_state_dict(state)
    return tokenizer, backbone, heads


def score_all_encoded(
    backbone: Any,
    heads: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
) -> Any:
    pooled = pool_encoded(
        backbone,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return score_all_pooled(heads, pooled)


def pool_encoded(
    backbone: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
) -> Any:
    """Return the last non-padding representation used by every ranker head."""

    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    reverse_positions = attention_mask.long().flip(dims=[1]).argmax(dim=1)
    positions = attention_mask.shape[1] - 1 - reverse_positions
    batch = hidden.shape[0]
    pooled = hidden[positions.new_tensor(range(batch)), positions]
    return pooled.float()


def score_all_pooled(heads: Any, pooled: Any) -> Any:
    """Apply ranker heads to an already-computed public-text representation."""

    long_head_names = {
        "phase_context",
        "t3_memory_utility",
        "t3_memory_family",
        "t3_memory_operation",
        "t4_budget_utility",
        "t4_budget_family",
        "t4_budget_state",
    }
    predictions = {
        name: head(pooled)
        for name, head in heads.items()
        if name not in RESIDUAL_HEAD_NAMES and name not in long_head_names
    }
    predictions["utility"] = predictions["utility"] + heads[
        "utility_residual"
    ](pooled)
    predictions["family"] = predictions["family"] + heads[
        "family_residual"
    ](pooled)
    # V2 reuses these exact legacy heads for the T1/T2 local branch.  Keep
    # explicit aliases before the generic routed output replaces utility/family.
    predictions["legacy_local_utility"] = predictions["utility"]
    predictions["legacy_local_family"] = predictions["family"]
    if "phase_gate" in heads:
        import torch

        phase_probabilities = torch.softmax(predictions["phase_gate"], dim=-1)
        branch_probabilities = phase_branch_probabilities(
            predictions["phase_gate"],
            phase_names=(
                PHASE_NAMES_V2
                if getattr(heads, "agentguard_architecture", "legacy")
                == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                else PHASE_NAMES
            ),
        )
        if (
            getattr(heads, "agentguard_architecture", "legacy")
            == SHARED_LOCAL_LONG_ARCHITECTURE_V2
        ):
            # Local phase rows must never receive gradients from T3/T4 action
            # heads.  Only the long-phase simplex enters the long context.
            long_logits = predictions["phase_gate"].new_full(
                predictions["phase_gate"].shape, float("-inf")
            )
            long_logits[:, list(LONG_PHASE_INDICES_V2)] = predictions[
                "phase_gate"
            ][:, list(LONG_PHASE_INDICES_V2)]
            long_phase_probabilities = torch.softmax(long_logits, dim=-1)
        else:
            long_phase_probabilities = phase_probabilities
        long_pooled = pooled + heads["phase_context"](long_phase_probabilities)
        for name in (
            "t3_memory_utility",
            "t3_memory_family",
            "t3_memory_operation",
            "t4_budget_utility",
            "t4_budget_family",
            "t4_budget_state",
        ):
            predictions[name] = heads[name](long_pooled)
        predictions["branch_probabilities"] = branch_probabilities
        predictions.update(
            route_branched_experts(
                predictions,
                branch_probabilities,
                use_legacy_local=(
                    getattr(heads, "agentguard_architecture", "legacy")
                    == SHARED_LOCAL_LONG_ARCHITECTURE_V2
                ),
            )
        )
    return {
        name: value.squeeze(-1) if value.shape[-1] == 1 else value
        for name, value in predictions.items()
    }


def activate_ranker_adapter(backbone: Any, branch: str) -> str | None:
    """Activate the public-route adapter while preserving trainable peers.

    PEFT's ``set_adapter`` freezes inactive adapters.  During joint DDP
    training both adapters must remain optimizer-visible even though exactly
    one is executed for each per-device state batch.
    """

    names = tuple(getattr(backbone, "agentguard_adapter_names", ()))
    if not names:
        return None
    adapter_name = "local" if branch == "local" else "long"
    if adapter_name not in names:
        raise RuntimeError(f"missing ranker adapter: {adapter_name}")
    backbone.set_adapter(adapter_name)
    trainable = bool(getattr(backbone, "agentguard_adapters_trainable", False))
    for parameter_name, parameter in backbone.named_parameters():
        if any(f".{name}." in parameter_name for name in names):
            parameter.requires_grad_(trainable)
    return adapter_name


def route_branched_experts(
    predictions: Mapping[str, Any],
    branch_probabilities: Any,
    *,
    use_legacy_local: bool = False,
) -> dict[str, Any]:
    """Route every candidate for a state through that state's phase gate."""

    import torch

    utility_experts = torch.stack(
        (
            predictions[
                "legacy_local_utility" if use_legacy_local else "local_utility"
            ],
            predictions["t3_memory_utility"],
            predictions["t4_budget_utility"],
        ),
        dim=-1,
    )
    family_experts = torch.stack(
        (
            predictions[
                "legacy_local_family" if use_legacy_local else "local_family"
            ],
            predictions["t3_memory_family"],
            predictions["t4_budget_family"],
        ),
        dim=-1,
    )

    def route(experts: Any) -> Any:
        weights = branch_probabilities
        while weights.ndim < experts.ndim:
            weights = weights.unsqueeze(-2)
        return (experts * weights).sum(dim=-1)

    return {"utility": route(utility_experts), "family": route(family_experts)}


def compose_candidate_utility(
    predictions: Mapping[str, Any],
    score_composition: Mapping[str, Any] | None,
) -> Any:
    """Return the manifest-declared learned score used by final selection."""

    utility = predictions["utility"]
    if not score_composition:
        return utility
    outcome_weights = dict(score_composition.get("outcome_weights") or {})
    if not outcome_weights:
        return utility
    import torch

    composite = utility * 0.0
    sigmoid_names = {
        "information_gain",
        "terminal_attack_mitigation",
        "business_cost",
        "overresponse_cost",
        "final_safe_success",
    }
    for name, raw_weight in outcome_weights.items():
        if name not in predictions:
            raise RuntimeError(f"score composition requires missing head: {name}")
        value = predictions[name]
        if name in sigmoid_names:
            value = torch.sigmoid(value)
        composite = composite + float(raw_weight) * value
    return (
        float(score_composition.get("utility_weight", 1.0)) * utility
        + float(score_composition.get("outcome_weight", 0.5)) * composite
    )


def score_encoded(
    backbone: Any,
    heads: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
) -> Any:
    """Return utility only; final action selection must not use auxiliary heads."""

    return score_all_encoded(
        backbone,
        heads,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )["utility"]


def encode_candidate_pairs(
    tokenizer: Any,
    observation: Mapping[str, Any],
    candidates: Sequence[CandidateOption],
    *,
    max_length: int,
    memory_tier_encoding: bool = False,
) -> dict[str, Any]:
    texts = [
        candidate_pair_text(
            observation, row, memory_tier_encoding=memory_tier_encoding
        )
        for row in candidates
    ]
    return _encode_without_truncation(tokenizer, texts, max_length=max_length)


def encode_public_state(
    tokenizer: Any,
    observation: Mapping[str, Any],
    *,
    max_length: int,
    memory_tier_encoding: bool = False,
) -> dict[str, Any]:
    return _encode_without_truncation(
        tokenizer,
        [public_state_text(observation, memory_tier_encoding=memory_tier_encoding)],
        max_length=max_length,
    )


def _encode_without_truncation(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
) -> dict[str, Any]:
    audit = tokenizer(
        list(texts),
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    lengths = [len(row) for row in audit["input_ids"]]
    if any(length > int(max_length) for length in lengths):
        raise ValueError(
            "candidate public-state input exceeds the frozen token limit: "
            f"max={max(lengths, default=0)} limit={int(max_length)}"
        )
    return tokenizer(
        list(texts),
        padding=True,
        truncation=False,
        max_length=int(max_length),
        return_tensors="pt",
        add_special_tokens=True,
    )
