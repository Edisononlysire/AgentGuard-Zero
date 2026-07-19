#!/usr/bin/env python3
"""Build a tiny Qwen3-compatible checkpoint for end-to-end trainer smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Tokenizer/config source, e.g. Qwen3-8B path")
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--intermediate-size", type=int, default=768)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    config.hidden_size = args.hidden_size
    config.num_hidden_layers = args.layers
    config.num_attention_heads = args.heads
    config.num_key_value_heads = args.kv_heads
    config.intermediate_size = args.intermediate_size
    config.head_dim = args.hidden_size // args.heads
    config.torch_dtype = "bfloat16"
    config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.to(dtype=torch.bfloat16)

    tokenizer.save_pretrained(output)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")

    manifest = {
        "purpose": "AgentGuard-Zero VDA warmup trainer smoke test",
        "base_tokenizer": args.base,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "intermediate_size": args.intermediate_size,
        "dtype": "bfloat16",
    }
    (output / "agentguard_smoke_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved tiny qwen3 smoke checkpoint to {output}")


if __name__ == "__main__":
    main()
