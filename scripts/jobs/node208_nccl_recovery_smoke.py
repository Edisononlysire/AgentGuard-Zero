#!/usr/bin/env python3
"""Exercise all four GPUs and NCCL collectives after a node208 GPU reset."""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"expected four ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    payload = torch.full((16 * 1024 * 1024,), float(rank + 1), device="cuda")
    deadline = time.monotonic() + 60.0
    iterations = 0
    started = time.monotonic()
    while time.monotonic() < deadline:
        payload.mul_(1.0000001)
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        payload.div_(world_size)
        torch.cuda.synchronize()
        iterations += 1
    dist.barrier()
    duration = time.monotonic() - started
    checksum = float(payload[0].item())
    device_name = torch.cuda.get_device_name(local_rank)
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "local_rank": local_rank,
            "device": device_name,
            "iterations": iterations,
            "duration_seconds": duration,
            "checksum": checksum,
        },
    )
    if rank == 0:
        report_path = Path(os.environ["AGZ_NCCL_SMOKE_REPORT"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "kind": "node208_nccl_recovery_smoke",
            "host": socket.gethostname(),
            "world_size": world_size,
            "backend": dist.get_backend(),
            "ranks": gathered,
            "status": "passed",
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
