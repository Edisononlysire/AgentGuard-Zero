#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scenario-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--scenarios-per-shard", type=int, default=4)
    parser.add_argument("--budgets", default="384,512,640")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def load_scenarios(path: Path) -> list[dict]:
    scenarios = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("vda_evaluation", {}).get("oracle_solvable", False):
                continue
            scenario = row.get("scenario", {})
            scenario_id = scenario.get("scenario_id")
            if not scenario_id or scenario_id in seen:
                continue
            seen.add(scenario_id)
            scenarios.append(scenario)
    return scenarios


def main() -> None:
    args = parse_args()
    import torch
    from transformers import StoppingCriteriaList

    from agentguard_zero.json_stopping import CompleteJSONObjectCriteria
    from agentguard_zero.schemas.action_schema_v4 import parse_action_json_v4
    from agentguard_zero.training.vda_dataset import scenario_to_training_row
    from eval_level1_select import HFBackend, as_messages, sanitize_initial_messages

    all_scenarios = load_scenarios(Path(args.scenario_jsonl))
    assigned = all_scenarios[args.shard_index :: args.num_shards][: args.scenarios_per_shard]
    if not assigned:
        raise SystemExit("no scenarios assigned")

    backend = HFBackend(
        SimpleNamespace(
            model_path=args.model_path,
            adapter_path="",
            attn_implementation=args.attn_implementation,
            dtype="bf16",
            device_map="",
        )
    )
    messages = []
    for scenario in assigned:
        row = scenario_to_training_row(scenario, split="dca_feedback")
        initial, _ = sanitize_initial_messages(as_messages(row.get("problem", "")))
        messages.append(initial)
    prompts = [backend.format_prompt(value) for value in messages]
    encoded = backend.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    encoded = {key: value.to(backend.device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]
    common = {
        "do_sample": False,
        "pad_token_id": backend.tokenizer.pad_token_id,
        "eos_token_id": backend.tokenizer.eos_token_id,
    }

    rows = []
    for budget in [int(value) for value in args.budgets.split(",") if value.strip()]:
        criteria = CompleteJSONObjectCriteria(backend.tokenizer, batch_size=len(assigned))
        torch.cuda.reset_peak_memory_stats(backend.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = backend.model.generate(
                **encoded,
                max_new_tokens=budget,
                stopping_criteria=StoppingCriteriaList([criteria]),
                **common,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        raw_outputs = [
            backend.tokenizer.decode(value[input_length:], skip_special_tokens=True).strip()
            for value in output
        ]
        parsed = [parse_action_json_v4(value) for value in raw_outputs]
        rows.append(
            {
                "budget": budget,
                "elapsed_s": elapsed,
                "scenario_count": len(assigned),
                "parse_ok": sum(int(value[1]) for value in parsed),
                "parse_messages": [value[2] for value in parsed],
                "raw_chars": [len(value) for value in raw_outputs],
                "raw_tokens": [
                    len(backend.tokenizer.encode(value, add_special_tokens=False))
                    for value in raw_outputs
                ],
                "raw_previews": [value[:240] for value in raw_outputs],
                "raw_tails": [value[-600:] for value in raw_outputs],
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated(backend.device)
                / (1024**3),
                "max_memory_reserved_gb": torch.cuda.max_memory_reserved(backend.device)
                / (1024**3),
            }
        )

    result = {
        "shard_index": args.shard_index,
        "attention": args.attn_implementation,
        "input_tokens": int(encoded["attention_mask"].sum(dim=1).max().item()),
        "scenario_ids": [value["scenario_id"] for value in assigned],
        "results": rows,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
