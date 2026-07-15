#!/usr/bin/env python3
"""Prune an isolated gate's trainer recovery files after adapter reload succeeds."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    load_checkpoint_manifest,
    sha256_tree,
    utc_now,
)


def prune_gate_recovery(manifest_path: Path) -> dict:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = load_checkpoint_manifest(
        manifest_path,
        role=str(raw.get("role", "")),
        backbone=str(raw.get("backbone", "")),
        round_index=int(raw.get("round", -1)),
    )
    checkpoint_path = Path(str(manifest.get("checkpoint_path", ""))).resolve()
    adapter_path = Path(str(manifest.get("adapter_path", ""))).resolve()
    if not adapter_path.is_dir() or sha256_tree(adapter_path) != manifest.get("adapter_sha256"):
        raise LineageError("stable gate adapter verification failed before recovery pruning")
    if checkpoint_path in adapter_path.parents:
        raise LineageError("refusing to prune a checkpoint that contains the stable adapter")
    trainer_root = checkpoint_path.parent
    if trainer_root.name != "checkpoints":
        raise LineageError(f"gate checkpoint root has an unexpected name: {trainer_root}")

    removed = []
    for step in sorted(trainer_root.glob("global_step_*")):
        if not step.is_dir():
            continue
        size = sum(path.stat().st_size for path in step.rglob("*") if path.is_file())
        removed.append({"path": str(step), "bytes": size})
        shutil.rmtree(step)
    tracker = trainer_root / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        tracker.unlink()
    return {
        "schema_version": 1,
        "kind": "pruned_gate_recovery_checkpoint",
        "pruned_at": utc_now(),
        "checkpoint_manifest": str(manifest_path),
        "adapter_path": str(adapter_path),
        "adapter_sha256": manifest["adapter_sha256"],
        "removed_steps": removed,
        "removed_bytes": sum(item["bytes"] for item in removed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = prune_gate_recovery(Path(args.checkpoint_manifest).resolve())
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
