"""Inference policy for direct candidate scoring."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agentguard_zero.candidate.compiler import CandidateCompiler
from agentguard_zero.candidate.generator import CandidateGenerator
from agentguard_zero.candidate.model import (
    encode_candidate_pairs,
    load_ranker_components,
    score_encoded,
)
from agentguard_zero.candidate.types import ActionFlags, CandidateOption
from agentguard_zero.schemas.action_schema_v4 import DEFAULT_ACTION_PACKET_V4


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str | None
    semantic_id: str | None
    packet: dict[str, Any]
    valid: bool
    invalid_noop: bool
    reason: str
    action_flags: ActionFlags
    candidate_count: int
    score: float | None
    scores: dict[str, float]


class CandidateRankerPolicy:
    def __init__(
        self,
        *,
        model_path: str | Path,
        adapter_path: str | Path,
        score_head_path: str | Path | None = None,
        heads_path: str | Path | None = None,
        device: str = "cuda:0",
        max_length: int = 2048,
        score_batch_size: int = 4,
        generator: CandidateGenerator | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.score_batch_size = max(1, int(score_batch_size))
        self.generator = generator or CandidateGenerator()
        self.compiler = CandidateCompiler()
        self.tokenizer, self.backbone, self.heads = load_ranker_components(
            model_path=model_path,
            adapter_path=adapter_path,
            heads_path=heads_path,
            score_head_path=score_head_path,
            trainable=False,
        )
        self.backbone.to(self.device).eval()
        self.heads.to(self.device).eval()

    def _score(
        self, observation: Mapping[str, Any], candidates: list[CandidateOption]
    ) -> list[float]:
        values: list[float] = []
        with self.torch.inference_mode():
            for offset in range(0, len(candidates), self.score_batch_size):
                chunk = candidates[offset : offset + self.score_batch_size]
                encoded = encode_candidate_pairs(
                    self.tokenizer,
                    observation,
                    chunk,
                    max_length=self.max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                scores = score_encoded(
                    self.backbone,
                    self.heads,
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                values.extend(float(item) for item in scores.detach().cpu().tolist())
        return values

    def decide(
        self,
        observation: Mapping[str, Any],
        *,
        sample: bool = False,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> CandidateDecision:
        try:
            candidates = self.generator.generate(
                observation,
                permutation_seed=int(seed or 0),
            )
            values = self._score(observation, candidates)
            if len(values) != len(candidates) or not all(
                self.torch.isfinite(self.torch.tensor(value)).item() for value in values
            ):
                raise RuntimeError("ranker produced missing or non-finite scores")
            if sample:
                if temperature <= 0.0:
                    raise ValueError("sampling temperature must be positive")
                generator = self.torch.Generator(device="cpu")
                if seed is not None:
                    generator.manual_seed(int(seed))
                probabilities = self.torch.softmax(
                    self.torch.tensor(values, dtype=self.torch.float32) / temperature,
                    dim=0,
                )
                index = int(
                    self.torch.multinomial(
                        probabilities, 1, replacement=True, generator=generator
                    ).item()
                )
            else:
                index = max(range(len(values)), key=lambda item: (values[item], -item))
            selected = candidates[index]
            packet = self.compiler.compile(selected.candidate_id, candidates)
            return CandidateDecision(
                candidate_id=selected.candidate_id,
                semantic_id=selected.semantic_id,
                packet=packet,
                valid=True,
                invalid_noop=False,
                reason="ok",
                action_flags=selected.action_flags,
                candidate_count=len(candidates),
                score=values[index],
                scores={row.candidate_id: values[i] for i, row in enumerate(candidates)},
            )
        except Exception as exc:
            return CandidateDecision(
                candidate_id=None,
                semantic_id=None,
                packet=copy.deepcopy(DEFAULT_ACTION_PACKET_V4),
                valid=False,
                invalid_noop=True,
                reason=f"{type(exc).__name__}: {exc}",
                action_flags=ActionFlags(),
                candidate_count=0,
                score=None,
                scores={},
            )
