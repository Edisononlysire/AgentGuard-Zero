"""Qwen-backed multi-head scorer for public-state action candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from agentguard_zero.candidate.types import CandidateOption
from agentguard_zero.world.public_projector import assert_public, project_public


FORMAT_VERSION = "candidate_pair_v1"
HEAD_NAMES = (
    "utility",
    "family",
    "belief",
    "uncertainty",
    "probe_value",
    "business_risk",
    "safety_risk",
)


def candidate_pair_text(
    observation: Mapping[str, Any], candidate: CandidateOption
) -> str:
    public = project_public(dict(observation))
    assert_public(public)
    context = json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    flags = ",".join(
        name for name, enabled in candidate.action_flags.to_dict().items() if enabled
    )
    packet = candidate.compiled_packet
    belief = json.dumps(
        packet.get("belief", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    safety = packet.get("safety_check", {}) or {}
    return (
        "Score one legal AgentGuard-Zero defense candidate using only the public state. "
        "Higher means better safe long-term utility.\n"
        f"Public state: {context}\n"
        f"Action family: {candidate.action_family}\n"
        f"Candidate: {candidate.public_summary}\n"
        f"Public references: {','.join(candidate.referenced_ids) or 'none'}\n"
        f"Action flags: {flags or 'none'}\n"
        f"Proposed belief: {belief}\n"
        f"Proposed uncertainty: {float(packet.get('uncertainty', 1.0)):.6f}\n"
        f"Declared business risk: {float(safety.get('business_risk', 0.0)):.6f}\n"
        f"Declared overresponse risk: {float(safety.get('overresponse_risk', 0.0)):.6f}"
    )


def load_ranker_components(
    *,
    model_path: str | Path,
    adapter_path: str | Path | None = None,
    heads_path: str | Path | None = None,
    score_head_path: str | Path | None = None,
    trainable: bool = False,
    dtype: str = "bf16",
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
    tokenizer.truncation_side = "left"
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
    hidden_size = int(
        getattr(backbone.config, "hidden_size", 0)
        or getattr(backbone.config, "text_config", {}).hidden_size
    )
    heads = torch.nn.ModuleDict(
        {
            "utility": torch.nn.Linear(hidden_size, 1),
            "family": torch.nn.Linear(hidden_size, 6),
            "belief": torch.nn.Linear(hidden_size, 4),
            "uncertainty": torch.nn.Linear(hidden_size, 1),
            "probe_value": torch.nn.Linear(hidden_size, 1),
            "business_risk": torch.nn.Linear(hidden_size, 1),
            "safety_risk": torch.nn.Linear(hidden_size, 1),
        }
    )
    for head in heads.values():
        torch.nn.init.normal_(head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(head.bias)
    if heads_path:
        state = torch.load(str(heads_path), map_location="cpu", weights_only=True)
        heads.load_state_dict(state)
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
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    positions = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
    batch = hidden.shape[0]
    pooled = hidden[positions.new_tensor(range(batch)), positions]
    pooled = pooled.float()
    return {
        name: head(pooled).squeeze(-1) if name != "family" and name != "belief" else head(pooled)
        for name, head in heads.items()
    }


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
) -> dict[str, Any]:
    texts = [candidate_pair_text(observation, row) for row in candidates]
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
        add_special_tokens=True,
    )
