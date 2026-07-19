#!/usr/bin/env python3
"""Tiny two-rank NCCL/FSDP/CPU-offload regression before loading the 8B model."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy

from verl.utils.torch_functional import AnyPrecisionAdamW
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"GPU preflight requires two ranks, got {world_size}")

    probe = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(probe)
    if probe.item() != 3.0:
        raise RuntimeError(f"NCCL all_reduce returned {probe.item()}, expected 3")

    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    module = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8)).to(torch.bfloat16)
    fsdp = FSDP(
        module,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        device_id=device,
        device_mesh=mesh,
        use_orig_params=False,
    )
    optimizer = AnyPrecisionAdamW(fsdp.parameters(), lr=1e-4, use_kahan_summation=False)
    inputs = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
    loss = fsdp(inputs).float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    for parameter in fsdp.parameters():
        if parameter.dtype != torch.bfloat16:
            raise RuntimeError(f"Expected bf16 FSDP master parameters, got {parameter.dtype}")
    for param_state in optimizer.state.values():
        if "compensation" in param_state:
            raise RuntimeError("Kahan compensation must be disabled for the constrained-memory smoke run")
        for name, value in param_state.items():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise RuntimeError(f"Optimizer state was not CPU-resident: {value.device}")
            if name != "step" and isinstance(value, torch.Tensor) and value.dtype != torch.bfloat16:
                raise RuntimeError(f"AnyPrecision state {name} was not bf16: {value.dtype}")

    model_state, optim_state = get_state_dict(
        fsdp,
        optimizer,
        options=StateDictOptions(cpu_offload=True),
    )
    if not model_state or not optim_state:
        raise RuntimeError("FSDP checkpoint state dict is empty")

    sync_state = get_model_state_dict(fsdp)
    dtensor_count = 0
    for name, value in sync_state.items():
        if isinstance(value, DTensor):
            dtensor_count += 1
            full = FSDPVLLMShardingManager._gather_dtensor(None, value)
            if tuple(full.shape) != tuple(value.shape):
                raise RuntimeError(f"DTensor gather shape mismatch for {name}: {full.shape} vs {value.shape}")
    if dtensor_count == 0:
        raise RuntimeError("Expected FSDP model state to contain DTensors")

    dist.barrier()
    if rank == 0:
        print(
            "Agent0 GPU preflight OK: "
            f"nccl_world_size={world_size} bf16_anyprecision_no_kahan_cpu_optimizer=True "
            f"checkpoint=True dtensors={dtensor_count}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
