#!/usr/bin/env python3
"""Independently reload one AgentGuard-Zero checkpoint or frozen base model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import (
    atomic_write_json,
    load_checkpoint_manifest,
    sha256_file,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

    manifest_path = Path(args.checkpoint_manifest).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = load_checkpoint_manifest(
        manifest_path,
        role=str(raw.get("role", "")),
        backbone=str(raw.get("backbone", "")),
        round_index=int(raw.get("round", -1)),
    )
    if not manifest.get("adapter_path") and manifest.get("status") != "frozen":
        raise SystemExit("trained adapter path is missing")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    model_path = manifest["base_model"]["path"]
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_class = (
        AutoModelForVision2Seq
        if type(config) in AutoModelForVision2Seq._model_mapping.keys()
        else AutoModelForCausalLM
    )
    base = model_class.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = (
        PeftModel.from_pretrained(base, manifest["adapter_path"])
        if manifest.get("adapter_path")
        else base
    )
    model.to(args.device)
    model.eval()

    report = {
        "schema_version": 1,
        "kind": "adapter_reload",
        "validated_at": utc_now(),
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "role": manifest["role"],
        "backbone": manifest["backbone"],
        "round": int(manifest["round"]),
        "adapter_path": manifest["adapter_path"],
        "adapter_sha256": manifest["adapter_sha256"],
        "model_class": type(base).__name__,
        "reload_mode": "lora_adapter" if manifest.get("adapter_path") else "frozen_base",
        "peft_model_class": type(model).__name__ if manifest.get("adapter_path") else None,
        "active_adapter": (
            str(getattr(model, "active_adapter", "default"))
            if manifest.get("adapter_path")
            else None
        ),
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "reload_ok": True,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
