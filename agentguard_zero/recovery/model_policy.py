"""Frozen K=1 model policy used by recovery Gate A and DAgger."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agentguard_zero.recovery.public_teacher import compact_wire_packet
from agentguard_zero.schemas.action_schema_v4 import (
    DEFAULT_ACTION_PACKET_V4,
    parse_action_json_v4,
)


ACTIVE_PROBE_TOOLS = frozenset(
    {"SourceChallenge", "CanaryProbe", "DecoyProbe", "ShadowActionProbe"}
)
PASSIVE_VERIFICATION_TOOLS = frozenset(
    {
        "LogQuery",
        "CrossCheck",
        "ProvenanceCheck",
        "GraphQuery",
        "BusinessImpactEstimator",
    }
)


@dataclass(frozen=True)
class ModelDecision:
    text: str
    packet: dict[str, Any]
    valid: bool
    parse_reason: str
    action_category: str


def action_category(packet: Mapping[str, Any]) -> str:
    tool = packet.get("tool_call", {}) or {}
    tool_name = str(tool.get("name", "None"))
    if tool_name in ACTIVE_PROBE_TOOLS:
        return "active_probe"
    if tool_name in PASSIVE_VERIFICATION_TOOLS:
        return "passive_verification"
    if packet.get("trust_operations") or packet.get("trust_operation"):
        return "trust"
    if (
        packet.get("memory_operations")
        or packet.get("memory_operation")
        or packet.get("memory_usage")
        or packet.get("memory_use")
    ):
        return "memory"
    response = packet.get("response", {}) or {}
    if str(response.get("action", "Observe")) != "Observe":
        return "mitigation"
    return "observe"


class RecoveryModelPolicy:
    """Lazy Transformers/PEFT wrapper with the frozen Gate-A decoding contract."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        adapter_path: str | Path | None = None,
        device: str = "cuda:0",
        max_new_tokens: int = 320,
        trust_remote_code: bool = True,
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - server dependency gate
            raise RuntimeError(
                "Gate-A inference requires torch, transformers, and peft"
            ) from exc

        self._torch = torch
        self.device = str(device)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                is_trainable=False,
            )
        self.model = model.to(self.device)
        self.model.eval()

    def render_prompt(self, public_prompt: str) -> str:
        messages = [{"role": "user", "content": public_prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return public_prompt

    def decide(self, public_prompt: str) -> ModelDecision:
        rendered = self.render_prompt(public_prompt)
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        suffix = generated[0, encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(suffix, skip_special_tokens=True).strip()
        packet, valid, reason = parse_action_json_v4(text)
        if not valid:
            fallback = copy.deepcopy(DEFAULT_ACTION_PACKET_V4)
            return ModelDecision(
                text=text,
                packet=fallback,
                valid=False,
                parse_reason=reason,
                action_category="observe",
            )
        compact = compact_wire_packet(packet)
        return ModelDecision(
            text=text,
            packet=packet,
            valid=True,
            parse_reason="ok",
            action_category=action_category(compact),
        )
