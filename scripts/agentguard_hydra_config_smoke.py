#!/usr/bin/env python
"""Smoke-test the Hydra PPO config used by AgentGuard-Zero warmup training."""

from __future__ import annotations

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "executor_train" / "verl_tool" / "trainer" / "config"
    reward_path = root / "curriculum_train" / "examples" / "reward_function" / "vda_reward.py"
    n_gpus = int(os.environ.get("AGZ_N_GPUS_PER_NODE", "1"))
    rollout_n = int(os.environ.get("AGZ_ROLLOUT_N", str(n_gpus)))
    use_kl_loss = os.environ.get("AGZ_USE_KL_LOSS", "false")
    ray_num_cpus = int(os.environ.get("AGZ_RAY_NUM_CPUS", "6"))
    max_prompt_length = int(os.environ.get("AGZ_MAX_PROMPT_LENGTH", "2048"))
    max_response_length = int(os.environ.get("AGZ_MAX_RESPONSE_LENGTH", "512"))

    overrides = [
        f"data.train_files={root / 'data' / 'smoke' / 'vda_train.parquet'}",
        f"data.val_files={root / 'data' / 'smoke' / 'vda_train.parquet'}",
        "data.prompt_key=problem",
        "data.train_batch_size=1",
        "data.val_batch_size=1",
        "data.gen_batch_size=1",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.truncation=right",
        f"ray_init.num_cpus={ray_num_cpus}",
        "reward_model.reward_manager=naive",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        "trainer.logger=['console']",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        "trainer.total_training_steps=2",
        "trainer.val_before_train=False",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
        "algorithm.use_kl_in_reward=False",
    ]

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)

    required = [
        "data.train_files",
        "data.val_files",
        "data.prompt_key",
        "data.reward_fn_key",
        "data.train_batch_size",
        "data.val_batch_size",
        "data.gen_batch_size",
        "data.dataloader_num_workers",
        "data.validation_shuffle",
        "data.sampler.class_path",
        "data.sampler.class_name",
        "data.max_prompt_length",
        "data.max_response_length",
        "reward_model.reward_manager",
        "custom_reward_function.path",
        "actor_rollout_ref.rollout.name",
        "trainer.n_gpus_per_node",
        "ray_init.num_cpus",
    ]
    nullable = {"data.sampler.class_path", "data.sampler.class_name"}
    missing = [key for key in required if OmegaConf.select(cfg, key) is None and key not in nullable]
    if missing:
        raise AssertionError(f"missing required config values: {missing}")

    real_train_batch_size = cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n
    minimal_bsz = cfg.trainer.n_gpus_per_node * cfg.trainer.nnodes
    if real_train_batch_size % minimal_bsz != 0:
        raise AssertionError(
            "invalid FSDP batch geometry: "
            f"train_batch_size * rollout.n = {real_train_batch_size}, "
            f"but must be divisible by {minimal_bsz}"
        )

    print("hydra_config_ok", {
        "prompt_key": cfg.data.prompt_key,
        "reward_fn_key": cfg.data.reward_fn_key,
        "reward_manager": cfg.reward_model.reward_manager,
        "rollout": cfg.actor_rollout_ref.rollout.name,
        "rollout_n": cfg.actor_rollout_ref.rollout.n,
        "train_batch_size": cfg.data.train_batch_size,
        "gen_batch_size": cfg.data.gen_batch_size,
        "dataloader_num_workers": cfg.data.dataloader_num_workers,
        "n_gpus_per_node": cfg.trainer.n_gpus_per_node,
        "ray_num_cpus": cfg.ray_init.num_cpus,
        "use_kl_loss": cfg.actor_rollout_ref.actor.use_kl_loss,
        "use_kl_in_reward": cfg.algorithm.use_kl_in_reward,
    })


if __name__ == "__main__":
    main()
