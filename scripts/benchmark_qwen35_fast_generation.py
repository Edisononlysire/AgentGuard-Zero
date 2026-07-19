#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

# Triton cache writes are not process-safe on this cluster's shared home filesystem.
# Set a rank-specific node-local cache before importing torch/FLA/Transformers.
_cache_root = Path(os.environ.get("AGZ_TRITON_CACHE_ROOT", "/tmp/agentguard_zero_triton"))
_cache_dir = _cache_root / f"generation_benchmark_rank_{os.environ.get('LOCAL_RANK', '0')}_{os.getpid()}"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(_cache_dir)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=750)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    return parser.parse_args()


def validate_causal_conv(device: torch.device) -> float:
    from causal_conv1d import causal_conv1d_fn

    torch.manual_seed(17)
    x = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
    weight = torch.randn(32, 4, device=device, dtype=torch.float32)
    bias = torch.randn(32, device=device, dtype=torch.float32)
    fast = causal_conv1d_fn(x, weight, bias, activation="silu")
    slow = F.conv1d(x, weight[:, None, :], bias=bias, padding=3, groups=32)
    slow = F.silu(slow[:, :, : x.shape[-1]])
    return float((fast - slow).abs().max().item())


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

    conv_max_error = validate_causal_conv(device)
    if not is_fast_path_available:
        raise RuntimeError("Qwen3.5 fast path is still unavailable")
    if conv_max_error > 1e-3:
        raise RuntimeError(f"causal_conv1d numerical error is too large: {conv_max_error}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    model_class = (
        AutoModelForVision2Seq
        if type(config) in AutoModelForVision2Seq._model_mapping.keys()
        else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    seed_text = (
        "Generate one compact defensive cyber scenario as valid JSON with abstract evidence, "
        "business constraints, trust uncertainty, and safe verification actions. "
    )
    raw = seed_text
    while len(tokenizer(raw, add_special_tokens=False)["input_ids"]) < args.prompt_tokens:
        raw += seed_text
    token_ids = tokenizer(raw, add_special_tokens=False)["input_ids"][: args.prompt_tokens]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
    encoded = tokenizer(
        [prompt] * args.batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.prompt_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    common = {
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    with torch.inference_mode():
        model.generate(**encoded, max_new_tokens=4, min_new_tokens=4, **common)
    torch.cuda.synchronize()
    dist.barrier()

    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.new_tokens,
            min_new_tokens=args.new_tokens,
            **common,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    input_length = encoded["input_ids"].shape[-1]
    local_new_tokens = int((generated.shape[-1] - input_length) * generated.shape[0])

    elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
    token_tensor = torch.tensor(local_new_tokens, device=device, dtype=torch.long)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
    max_elapsed = float(elapsed_tensor.item())
    total_new_tokens = int(token_tensor.item())
    aggregate_tps = total_new_tokens / max_elapsed

    if rank == 0:
        result = {
            "schema_version": 1,
            "kind": "qwen35_fast_generation_benchmark",
            "model_path": str(Path(args.model_path).resolve()),
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "prompt_tokens": input_length,
            "new_tokens_per_sequence": args.new_tokens,
            "attn_implementation": args.attn_implementation,
            "total_new_tokens": total_new_tokens,
            "elapsed_seconds": max_elapsed,
            "aggregate_tokens_per_second": aggregate_tps,
            "max_memory_allocated_gb_per_gpu": max_memory_allocated,
            "max_memory_reserved_gb_per_gpu": max_memory_reserved,
            "qwen35_fast_path_available": bool(is_fast_path_available),
            "causal_conv1d_max_abs_error": conv_max_error,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "triton_cache_isolated_per_rank": True,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
