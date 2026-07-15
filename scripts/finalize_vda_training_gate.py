#!/usr/bin/env python3
"""Snapshot and validate an isolated VDA gate adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import write_trained_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", choices=["qwen3.5-4b", "qwen3.5-9b"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--dca-manifest", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    manifest = write_trained_manifest(
        output_dir / "checkpoint_manifest.json",
        role="vda",
        backbone=args.backbone,
        round_index=1,
        model_path=args.model_path,
        seed=args.seed,
        parent_manifest_path=args.parent_manifest,
        training_data_manifest_path=args.pool_manifest,
        checkpoint_root=args.checkpoint_root,
        training_config={
            "kind": "isolated_vda_gate",
            "batch_size": args.batch_size,
            "steps": 1,
        },
    )
    dca = json.loads(Path(args.dca_manifest).read_text(encoding="utf-8"))
    if manifest["adapter_sha256"] == dca.get("adapter_sha256"):
        raise SystemExit("gate VDA and DCA adapters unexpectedly have the same hash")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
