#!/usr/bin/env python3
"""Smoke test for AgentGuard-Zero trajectory-reward routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "executor_train", ROOT / "executor_train" / "verl", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from verl import DataProto
from verl_tool.trainer.ppo.reward import compute_reward


def _fallback_reward(data: DataProto, return_dict: bool = False):
    responses = data.batch["responses"]
    reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
    reward_tensor[:, -1] = 0.75
    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": {"fallback_marker": [0.75] * len(data)},
        }
    return reward_tensor


def main() -> None:
    responses = torch.ones((3, 5), dtype=torch.long)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    data = DataProto.from_dict(
        tensors={
            "responses": responses,
            "attention_mask": attention_mask,
        },
        non_tensors={
            "turn_rewards": np.array([[None, 4.25], [], [None]], dtype=object),
            "tool_interact_info": np.array(
                [
                    [],
                    [{"score": {"reward": -1.5}}],
                    [],
                ],
                dtype=object,
            ),
        },
    )

    reward_tensor, extra = compute_reward(data, _fallback_reward)
    sequence_scores = reward_tensor.sum(dim=-1).tolist()
    expected_scores = [4.25, -1.5, 0.75]
    assert sequence_scores == expected_scores, (sequence_scores, expected_scores)
    assert reward_tensor[0, 4].item() == 4.25
    assert reward_tensor[1, 2].item() == -1.5
    assert reward_tensor[2, 3].item() == 0.75
    assert extra["level1_trajectory_reward_available"] == [1.0, 1.0, 0.0]
    assert extra["level1_trajectory_reward_source"] == [1.0, 2.0, 0.0]
    assert extra["single_step_reward_fallback"] == [0.0, 0.0, 1.0]

    print(
        json.dumps(
            {
                "ok": True,
                "sequence_scores": sequence_scores,
                "extra": {
                    key: extra[key]
                    for key in [
                        "level1_trajectory_reward_available",
                        "level1_trajectory_reward_source",
                        "single_step_reward_fallback",
                    ]
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
