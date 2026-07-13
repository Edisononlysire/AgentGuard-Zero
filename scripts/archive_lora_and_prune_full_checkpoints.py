#!/usr/bin/env python3
"""Archive LoRA adapters and optionally remove bulky FSDP step directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-tree", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--delete-full-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_tree = Path(args.checkpoint_tree).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    if not checkpoint_tree.is_dir():
        raise SystemExit(f"checkpoint tree does not exist: {checkpoint_tree}")

    steps = sorted(checkpoint_tree.glob("**/global_step_*"))
    if not steps:
        raise SystemExit(f"no global_step directories under {checkpoint_tree}")
    records = []
    for step in steps:
        adapter = step / "actor" / "lora_adapter" / "adapter_model.safetensors"
        config = step / "actor" / "lora_adapter" / "adapter_config.json"
        if not adapter.is_file() or adapter.stat().st_size <= 0 or not config.is_file():
            raise SystemExit(f"refusing to prune step without a complete LoRA adapter: {step}")
        relative = step.relative_to(checkpoint_tree)
        target = archive_dir / relative / "lora_adapter"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter, target / adapter.name)
        shutil.copy2(config, target / config.name)
        copied = target / adapter.name
        if sha256(copied) != sha256(adapter):
            raise SystemExit(f"adapter verification failed: {step}")
        round_manifest = step.parent.parent / "manifest.json"
        if round_manifest.is_file():
            shutil.copy2(round_manifest, target.parent / "round_manifest.json")
        records.append(
            {
                "source_step": str(step),
                "archived_adapter": str(copied),
                "adapter_sha256": sha256(copied),
                "adapter_bytes": copied.stat().st_size,
                "full_step_bytes": sum(path.stat().st_size for path in step.rglob("*") if path.is_file()),
            }
        )

    archive_dir.mkdir(parents=True, exist_ok=True)
    inventory = archive_dir / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint_tree": str(checkpoint_tree),
                "steps": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.delete_full_checkpoints:
        for step in steps:
            shutil.rmtree(step)
    print(inventory)


if __name__ == "__main__":
    main()
