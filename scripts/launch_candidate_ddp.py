#!/usr/bin/env python3
"""Launch candidate training ranks without the cluster-broken elastic parent."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing child command")
    args.log_dir.mkdir(parents=True, exist_ok=False)
    port = _free_port()
    processes: list[tuple[subprocess.Popen[bytes], object]] = []
    for rank in range(args.nproc):
        environment = os.environ.copy()
        environment.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "WORLD_SIZE": str(args.nproc),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
            }
        )
        log_handle = (args.log_dir / f"rank_{rank}.log").open("wb")
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_handle))
    exit_codes = []
    try:
        for process, _ in processes:
            exit_codes.append(process.wait())
    finally:
        for process, handle in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
    failures = [code for code in exit_codes if code != 0]
    if failures:
        for rank, code in enumerate(exit_codes):
            print(f"rank={rank} exit_code={code}", file=sys.stderr)
        return failures[0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
