#!/usr/bin/env python3
"""Select the passing Gate-A arm and freeze its DAgger parent manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.gates import (
    choose_gate_a_arm,
    evaluate_gate_a,
)
from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    sha256_file,
    sha256_tree,
    utc_now,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gate", type=Path, required=True)
    parser.add_argument("--vda1-gate", type=Path, required=True)
    parser.add_argument("--base-adapter-manifest", type=Path, required=True)
    parser.add_argument("--vda1-adapter-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    gate_payloads = {
        "qwen3.5_base": _load(args.base_gate),
        "vda_1": _load(args.vda1_gate),
    }
    adapter_paths = {
        "qwen3.5_base": args.base_adapter_manifest,
        "vda_1": args.vda1_adapter_manifest,
    }
    adapter_manifests = {arm: _load(path) for arm, path in adapter_paths.items()}
    shared_training_keys = (
        "training_data_manifest_sha256",
        "training_parquet_sha256",
        "training_record_count",
        "training_config",
        "seed",
        "effective_batch_size",
        "review_approval_sha256",
    )
    for key in shared_training_keys:
        values = {
            json.dumps(payload.get(key), sort_keys=True)
            for payload in adapter_manifests.values()
        }
        if len(values) != 1:
            raise RuntimeError(f"Gate-A arms differ on frozen training field: {key}")
    verdicts = [
        evaluate_gate_a(payload["metrics"], arm=arm)
        for arm, payload in gate_payloads.items()
    ]
    selection = choose_gate_a_arm(verdicts)
    if not selection["accepted"]:
        output = {
            "schema_version": 1,
            "kind": "recovery_gate_a_selection",
            "protocol_version": RECOVERY_PROTOCOL_VERSION,
            "status": "rejected",
            "accepted": False,
            "created_at": utc_now(),
            "selection": selection,
            "gate_sha256": {
                "qwen3.5_base": sha256_file(args.base_gate),
                "vda_1": sha256_file(args.vda1_gate),
            },
            "next_stage": "stop_and_repair_bootstrap",
        }
        atomic_write_json(args.output, output)
        return 2

    arm = str(selection["selected_arm"])
    adapter_manifest_path = adapter_paths[arm]
    adapter_manifest = adapter_manifests[arm]
    if adapter_manifest.get("arm") != arm:
        raise RuntimeError("selected adapter manifest arm mismatch")
    adapter_path = Path(str(adapter_manifest.get("adapter_path", "")))
    if not adapter_path.is_dir() or sha256_tree(adapter_path) != adapter_manifest.get(
        "adapter_sha256"
    ):
        raise RuntimeError("selected adapter verification failed")
    output = {
        "schema_version": 1,
        "kind": "recovery_gate_a_selection",
        "protocol_version": RECOVERY_PROTOCOL_VERSION,
        "status": "selected_pending_single_dagger",
        "accepted": True,
        "created_at": utc_now(),
        "selection": selection,
        "selected_arm": arm,
        "selected_gate_metrics": gate_payloads[arm]["metrics"],
        "selected_adapter_manifest": str(adapter_manifest_path.resolve()),
        "selected_adapter_manifest_sha256": sha256_file(adapter_manifest_path),
        "selected_adapter_path": str(adapter_path.resolve()),
        "selected_adapter_sha256": adapter_manifest["adapter_sha256"],
        "gate_sha256": {
            "qwen3.5_base": sha256_file(args.base_gate),
            "vda_1": sha256_file(args.vda1_gate),
        },
        "next_stage": "single_dagger_collection",
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
