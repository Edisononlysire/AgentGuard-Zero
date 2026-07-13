#!/usr/bin/env python3
"""Fail-fast per-node gate before loading a TMCD-v2 training model."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, model_identity, sha256_file, utc_now


def _run(command: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["qwen3.5-4b", "qwen3.5-9b"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--variant", choices=["full", "append_only_memory"], required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = socket.gethostname()
    if args.expected_node not in host:
        raise SystemExit(f"wrong execution node: expected {args.expected_node}, got {host}")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "agent0-gpu":
        raise SystemExit(f"wrong conda environment: expected agent0-gpu, got {conda_env!r}")
    gpu_ids = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(gpu_ids) != 4:
        raise SystemExit(f"expected four allocated GPUs, got CUDA_VISIBLE_DEVICES={gpu_ids}")
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise SystemExit(f"invalid model path: {model_path}")
    frozen = ROOT / "data" / "tmcd_v2" / "manifests"
    required = [
        frozen / "protocol.json",
        frozen / "manipulation_families.json",
        frozen / "ood_holdout_families.json",
        frozen / "schema_versions.json",
        frozen / "source_freeze.json",
    ]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing frozen manifest: {path}")
    protocol = json.loads((frozen / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "tmcd-v2":
        raise SystemExit("frozen protocol version mismatch")
    source_freeze = json.loads((frozen / "source_freeze.json").read_text(encoding="utf-8"))
    framework_files = source_freeze.get("training_framework", {})
    if not framework_files:
        raise SystemExit("source freeze does not record the training framework")
    for relative_path, expected_sha256 in framework_files.items():
        framework_path = ROOT / relative_path
        if not framework_path.is_file():
            raise SystemExit(f"missing training framework file: {framework_path}")
        if sha256_file(framework_path) != expected_sha256:
            raise SystemExit(f"training framework hash mismatch: {framework_path}")

    test_output = _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"])
    framework_import = _run(
        [
            sys.executable,
            "-s",
            "-c",
            "import verl; import verl_tool.trainer.main_ppo; print('verl_tool_import_ok')",
        ]
    )
    smoke_path = Path(args.output).resolve().with_name("protocol_smoke.json")
    smoke_output = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "smoke_tmcd_v2_protocol.py"),
            "--count",
            "256",
            "--output",
            str(smoke_path),
        ]
    )
    gpu_query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ]
    )
    report = {
        "protocol_version": "tmcd-v2",
        "kind": "job_preflight",
        "created_at": utc_now(),
        "host": host,
        "expected_node": args.expected_node,
        "backbone": args.backbone,
        "variant": args.variant,
        "conda_env": conda_env,
        "cuda_visible_devices": gpu_ids,
        "gpu_query": gpu_query.splitlines(),
        "model": model_identity(model_path),
        "protocol_manifest_sha256": sha256_file(frozen / "protocol.json"),
        "source_freeze_sha256": sha256_file(frozen / "source_freeze.json"),
        "unit_tests": "passed",
        "unit_test_output_tail": test_output.splitlines()[-5:],
        "training_framework_import": framework_import.splitlines()[-5:],
        "protocol_smoke": json.loads(smoke_path.read_text(encoding="utf-8")),
        "protocol_smoke_stdout_tail": smoke_output.splitlines()[-5:],
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
