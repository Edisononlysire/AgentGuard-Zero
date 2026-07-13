#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


LORA_TARGET_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def check_shards(model_path: Path) -> tuple[list[str], list[str], list[str], int]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (model_path / name).exists()]
    incomplete = [p.name for p in list(model_path.glob("*.incomplete")) + list(model_path.glob("*.tmp"))]
    total_bytes = sum((model_path / name).stat().st_size for name in shards if (model_path / name).exists())
    return shards, missing, incomplete, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only Qwen3.5 smoke check for AgentGuard-Zero.")
    parser.add_argument("--model", action="append", required=True, help="Local Qwen3.5 model directory.")
    parser.add_argument("--skip-processor", action="store_true", help="Skip AutoProcessor loading.")
    args = parser.parse_args()

    import huggingface_hub
    import safetensors
    import transformers
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    print(
        "versions",
        f"transformers={transformers.__version__}",
        f"huggingface_hub={huggingface_hub.__version__}",
        f"safetensors={safetensors.__version__}",
    )

    messages = [
        {"role": "system", "content": "You are a JSON-only cyber defense VDA."},
        {"role": "user", "content": "Return a safe action with tool_call CrossCheck."},
    ]

    for model in args.model:
        model_path = Path(model)
        print(f"MODEL {model_path}")
        shards, missing, incomplete, total_bytes = check_shards(model_path)
        print(f" shards={len(shards)} missing={missing} incomplete={incomplete} bytes={total_bytes}")
        if missing or incomplete:
            raise SystemExit(f"incomplete model snapshot: {model_path}")

        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        print(" config", type(config).__name__, getattr(config, "model_type", None), getattr(config, "architectures", None))

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        print(
            " tokenizer",
            type(tokenizer).__name__,
            f"chat_template={bool(getattr(tokenizer, 'chat_template', None))}",
            f"eos={tokenizer.eos_token_id}",
            f"pad={tokenizer.pad_token_id}",
            f"prompt_tokens={encoded['input_ids'].shape[-1]}",
        )

        if not args.skip_processor:
            processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
            print(" processor", type(processor).__name__)

        with init_empty_weights():
            causal = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        with init_empty_weights():
            conditional = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)

        targets = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, _module in conditional.named_modules()
                if name.rsplit(".", 1)[-1] in LORA_TARGET_NAMES
            }
        )
        print(" causal_empty_model", type(causal).__name__)
        print(" conditional_empty_model", type(conditional).__name__)
        print(" lora_targets", targets)


if __name__ == "__main__":
    main()
