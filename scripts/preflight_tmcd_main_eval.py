#!/usr/bin/env python3
"""Fail-fast gate for a frozen-checkpoint TMCD main evaluation."""

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

from agentguard_zero.protocol import TMCD_PROTOCOL_VERSION, TMCD_RELEASE_REVISION  # noqa: E402
from agentguard_zero.inference_contract import (  # noqa: E402
    FORMAL_VDA_MAX_NEW_TOKENS,
    TRAINED_VDA_PROMPT_CONTRACT,
    require_candidate_quality,
)
from agentguard_zero.training.coevolution import (  # noqa: E402
    atomic_write_json,
    model_identity,
    read_json,
    sha256_file,
    sha256_source_tree,
    utc_now,
)


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        detail = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        raise SystemExit(
            f"preflight command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed.stdout.strip()


def parse_source_hashes(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip()
        target = ROOT / relative
        if not target.is_file():
            raise SystemExit(f"evaluation source file is missing: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise SystemExit(
                f"evaluation source hash mismatch: {relative}: {actual} != {expected}"
            )
        rows.append({"path": relative, "sha256": actual})
    if not rows:
        raise SystemExit("evaluation source hash ledger is empty")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["qwen3.5-4b", "qwen3.5-9b"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--vda-manifest", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--ecrg-config", required=True)
    parser.add_argument("--source-hashes", required=True)
    parser.add_argument("--test-python", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = socket.gethostname()
    if args.expected_node not in host:
        raise SystemExit(f"wrong execution node: expected {args.expected_node}, got {host}")
    if os.environ.get("CONDA_DEFAULT_ENV", "") != "agent0-gpu":
        raise SystemExit("formal evaluation requires the agent0-gpu environment")
    gpu_ids = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if len(gpu_ids) != 4:
        raise SystemExit(f"formal evaluation requires four GPUs, got {gpu_ids}")

    model_path = Path(args.model_path).resolve()
    vda_manifest_path = Path(args.vda_manifest).resolve()
    data_manifest_path = Path(args.data_manifest).resolve()
    ecrg_config_path = Path(args.ecrg_config).resolve()
    source_hashes_path = Path(args.source_hashes).resolve()
    test_python = Path(args.test_python).resolve()
    for path in (
        model_path / "config.json",
        vda_manifest_path,
        data_manifest_path,
        ecrg_config_path,
        source_hashes_path,
        test_python,
    ):
        if not path.exists():
            raise SystemExit(f"formal preflight input is missing: {path}")

    vda_manifest = read_json(vda_manifest_path)
    identity = model_identity(model_path)
    if (
        vda_manifest.get("role") != "vda"
        or int(vda_manifest.get("round", -1)) != 3
        or vda_manifest.get("base_model", {}).get("identity_sha256")
        != identity["identity_sha256"]
    ):
        raise SystemExit("base model identity does not match frozen VDA3")

    data_manifest = read_json(data_manifest_path)
    if (
        data_manifest.get("status") != "sealed"
        or data_manifest.get("selected_count") != 2400
        or data_manifest.get("task_counts")
        != {"T1": 600, "T2": 600, "T3": 600, "T4": 600}
    ):
        raise SystemExit("formal TMCD-Test manifest is not sealed and balanced")
    ecrg_config = read_json(ecrg_config_path)
    if (
        ecrg_config.get("status") != "frozen"
        or ecrg_config.get("candidate_count") != 6
        or ecrg_config.get("hidden_state_access") is not False
        or ecrg_config.get("prompt_contract") != TRAINED_VDA_PROMPT_CONTRACT
        or ecrg_config.get("max_new_tokens") != FORMAL_VDA_MAX_NEW_TOKENS
        or ecrg_config.get("vda_manifest_sha256") != sha256_file(vda_manifest_path)
        or ecrg_config.get("vda_adapter_sha256") != vda_manifest.get("adapter_sha256")
    ):
        raise SystemExit("formal ECRG config is not bound to frozen VDA3/K=6")
    try:
        require_candidate_quality(
            ecrg_config.get("candidate_quality", {}), context="formal ECRG config"
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    frozen_dir = ROOT / "data" / "tmcd_v2" / "manifests"
    protocol = read_json(frozen_dir / "protocol.json")
    if (
        protocol.get("protocol_version") != TMCD_PROTOCOL_VERSION
        or protocol.get("release_revision") != TMCD_RELEASE_REVISION
    ):
        raise SystemExit("TMCD protocol freeze mismatch")
    training_freeze_path = frozen_dir / "source_freeze.json"
    training_freeze = read_json(training_freeze_path)
    expected_agentguard = training_freeze.get("source_trees", {}).get("agentguard_zero", "")
    current_agentguard = sha256_source_tree(ROOT / "agentguard_zero")
    evaluation_sources = parse_source_hashes(source_hashes_path)

    test_env = os.environ.copy()
    test_env.pop("PYTHONHOME", None)
    test_env["PYTHONPATH"] = str(ROOT)
    test_output = run(
        [str(test_python), "-m", "pytest", "-q", "tests"],
        env=test_env,
    )
    framework_import = run(
        [
            sys.executable,
            "-s",
            "-c",
            "import verl; import verl_tool.trainer.main_ppo; print('verl_tool_import_ok')",
        ]
    )
    output_path = Path(args.output).resolve()
    smoke_path = output_path.with_name(f"{output_path.stem}.protocol_smoke.json")
    smoke_output = run(
        [
            sys.executable,
            str(ROOT / "scripts" / "smoke_tmcd_v2_protocol.py"),
            "--count",
            "256",
            "--output",
            str(smoke_path),
        ]
    )
    gpu_query = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ]
    )
    report = {
        "schema_version": 1,
        "kind": "tmcd_frozen_checkpoint_evaluation_preflight",
        "status": "passed",
        "created_at": utc_now(),
        "host": host,
        "backbone": args.backbone,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "cuda_visible_devices": gpu_ids,
        "gpu_query": gpu_query.splitlines(),
        "model": identity,
        "vda_manifest": str(vda_manifest_path),
        "vda_manifest_sha256": sha256_file(vda_manifest_path),
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "ecrg_config": str(ecrg_config_path),
        "ecrg_config_sha256": sha256_file(ecrg_config_path),
        "protocol_version": TMCD_PROTOCOL_VERSION,
        "release_revision": TMCD_RELEASE_REVISION,
        "training_source_freeze": str(training_freeze_path),
        "training_source_freeze_sha256": sha256_file(training_freeze_path),
        "training_agentguard_source_sha256": expected_agentguard,
        "evaluation_agentguard_source_sha256": current_agentguard,
        "post_training_runtime_source_delta_recorded": current_agentguard != expected_agentguard,
        "evaluation_source_hashes": evaluation_sources,
        "evaluation_source_hashes_sha256": sha256_file(source_hashes_path),
        "unit_tests": "passed",
        "unit_test_python": str(test_python),
        "unit_test_output_tail": test_output.splitlines()[-5:],
        "training_framework_import": framework_import.splitlines()[-5:],
        "protocol_smoke": read_json(smoke_path),
        "protocol_smoke_stdout_tail": smoke_output.splitlines()[-5:],
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
