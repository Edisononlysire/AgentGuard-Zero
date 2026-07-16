# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
import math
import numbers
from functools import partial

import numpy as np
import ray
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl_tool.workers.reward_manager import get_reward_manager_cls # added by verl-tool


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Try to get a custom reward function based on the configuration
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    reward_manager = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )
    # added by verl-tool
    reward_manager.run_id = config.trainer.experiment_name
    return reward_manager


def _as_float(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, numbers.Real):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _last_numeric_reward(value):
    numeric = _as_float(value)
    if numeric is not None:
        return numeric

    if isinstance(value, dict):
        for key in ("reward", "score"):
            if key in value:
                numeric = _last_numeric_reward(value[key])
                if numeric is not None:
                    return numeric
        return None

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _last_numeric_reward(value.item())
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            numeric = _last_numeric_reward(item)
            if numeric is not None:
                return numeric
    return None


def _extract_level1_trajectory_rewards(data: DataProto):
    batch_size = len(data)
    rewards = [None] * batch_size
    sources = [0.0] * batch_size
    non_tensors = getattr(data, "non_tensor_batch", {}) or {}

    for key, source_id in (("turn_rewards", 1.0), ("tool_interact_info", 2.0)):
        values = non_tensors.get(key)
        if values is None:
            continue
        for idx, value in enumerate(values[:batch_size]):
            if rewards[idx] is not None:
                continue
            reward = _last_numeric_reward(value)
            if reward is not None:
                rewards[idx] = reward
                sources[idx] = source_id

    return rewards, sources


def _token_level_reward_from_sequence_rewards(data: DataProto, sequence_rewards):
    responses = data.batch["responses"]
    reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
    if "response_mask" in data.batch:
        response_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        response_mask = data.batch["attention_mask"][:, -responses.shape[-1]:]
    else:
        response_mask = torch.ones_like(responses, dtype=torch.long)

    last_token_positions = response_mask.sum(dim=-1).long() - 1
    last_token_positions = torch.clamp(last_token_positions, min=0, max=responses.shape[-1] - 1)
    rewards = torch.as_tensor(sequence_rewards, device=responses.device, dtype=torch.float32)
    reward_tensor[torch.arange(responses.shape[0], device=responses.device), last_token_positions] = rewards
    return reward_tensor


def _compute_fallback_reward(data: DataProto, reward_fn):
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        if os.environ.get("AGZ_REQUIRE_TRAJECTORY_REWARD", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            raise
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict or {}


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    trajectory_rewards, trajectory_sources = _extract_level1_trajectory_rewards(data)
    has_trajectory_reward = [reward is not None for reward in trajectory_rewards]
    require_trajectory_reward = os.environ.get(
        "AGZ_REQUIRE_TRAJECTORY_REWARD", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if require_trajectory_reward and not all(has_trajectory_reward):
        missing = [idx for idx, available in enumerate(has_trajectory_reward) if not available]
        raise RuntimeError(
            "TMCD formal training is missing terminal trajectory rewards for "
            f"batch indices {missing[:16]} (missing={len(missing)}, batch={len(missing) + sum(has_trajectory_reward)})"
        )

    if any(has_trajectory_reward):
        fallback_tensor = None
        fallback_extra_infos = {}
        fallback_scores = [0.0] * len(trajectory_rewards)
        if not all(has_trajectory_reward):
            fallback_tensor, fallback_extra_infos = _compute_fallback_reward(data, reward_fn)
            fallback_scores = fallback_tensor.sum(dim=-1).detach().cpu().tolist()

        sequence_rewards = [
            float(reward) if reward is not None else float(fallback_scores[idx])
            for idx, reward in enumerate(trajectory_rewards)
        ]
        reward_tensor = _token_level_reward_from_sequence_rewards(data, sequence_rewards)
        reward_extra_infos_dict = dict(fallback_extra_infos)
        reward_extra_infos_dict.update(
            {
                "level1_trajectory_reward": sequence_rewards,
                "level1_trajectory_reward_available": [float(x) for x in has_trajectory_reward],
                "level1_trajectory_reward_source": trajectory_sources,
                "single_step_reward_fallback": [0.0 if x else 1.0 for x in has_trajectory_reward],
            }
        )
        return reward_tensor, reward_extra_infos_dict

    reward_tensor, reward_extra_infos_dict = _compute_fallback_reward(data, reward_fn)

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
