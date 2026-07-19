#!/usr/bin/env python3
"""Non-Test HF gate for a frozen VDA checkpoint's action contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_tmcd_systems as ev  # noqa: E402
from agentguard_zero.inference_contract import (  # noqa: E402
    FORMAL_VDA_MAX_NEW_TOKENS,
    TRAINED_VDA_PROMPT_CONTRACT,
    summarize_candidate_quality,
)
from agentguard_zero.training.coevolution import (  # noqa: E402
    LineageError,
    atomic_write_json,
    load_checkpoint_manifest,
    read_json,
    sha256_file,
    sha256_tree,
    utc_now,
)
from ecrg_calibration_lib import candidate_features  # noqa: E402


TASK_IDS = ("T1", "T2", "T3", "T4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--vda-manifest", required=True)
    parser.add_argument("--round-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-per-task", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-new-tokens", type=int, default=FORMAL_VDA_MAX_NEW_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()
    if args.candidate_count != 6:
        parser.error("formal checkpoint contract gate requires K=6")
    if args.max_new_tokens != FORMAL_VDA_MAX_NEW_TOKENS:
        parser.error("checkpoint gate must match the 320-token training budget")
    if args.limit_per_task <= 0 or args.batch_size <= 0:
        parser.error("limit-per-task and batch-size must be positive")
    return args


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    data_path = Path(args.data).resolve()
    calibration_manifest_path = Path(args.calibration_manifest).resolve()
    vda_manifest_path = Path(args.vda_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "candidates.jsonl"
    manifest_path = output_dir / "manifest.json"
    if rows_path.exists() or manifest_path.exists():
        raise LineageError(f"refusing to overwrite checkpoint gate: {output_dir}")

    calibration_manifest = read_json(calibration_manifest_path)
    if (
        calibration_manifest.get("status") != "sealed"
        or int(calibration_manifest.get("selected_count", -1)) != 800
        or calibration_manifest.get("task_counts")
        != {"T1": 200, "T2": 200, "T3": 200, "T4": 200}
    ):
        raise LineageError("checkpoint gate requires sealed ECRG-Cal800")

    vda_manifest = load_checkpoint_manifest(
        vda_manifest_path,
        role="vda",
        backbone="qwen3.5-4b",
        round_index=args.round_index,
    )
    adapter_sha_before = str(vda_manifest["adapter_sha256"])
    if sha256_tree(vda_manifest["adapter_path"]) != adapter_sha_before:
        raise LineageError("VDA adapter hash mismatch before checkpoint gate")

    frame = pd.read_parquet(data_path)
    selected = []
    for task_id in TASK_IDS:
        task_frame = frame[frame["task_id"] == task_id].copy()
        task_frame = task_frame.sort_values("scenario_fingerprint", kind="mergesort")
        task_frame = task_frame.head(args.limit_per_task)
        if len(task_frame) != args.limit_per_task:
            raise LineageError(f"insufficient {task_id} calibration rows")
        selected.extend(task_frame.to_dict("records"))

    args.system = "agentguard_zero_train"
    args.model_backend = "hf"
    args.max_turns = 16
    args.model_path = str(vda_manifest["base_model"]["path"])
    args.adapter_path = str(vda_manifest["adapter_path"])
    args.device_map = ""
    args.do_sample = True
    args.stop_on_complete_json = True
    args.api_model = ""
    args.api_base_url = ""
    args.api_key_env = ""
    args.api_timeout = 90
    args.api_retries = 0
    args.api_response_format_json = False
    args.api_disable_thinking = True
    args.api_multi_choice = False
    args.api_system_prompt = ""

    backend = ev.build_backend(args)
    quality_decisions = []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        contexts = [ev.row_context(row, start + index, args) for index, row in enumerate(batch)]
        messages = [item[0] for item in contexts]
        public_contexts = [item[1] for item in contexts]
        if any(
            len(item) != 1 or item[0].get("role") != "user"
            for item in messages
        ):
            raise LineageError("trained VDA prompt was not preserved as one exact user row")
        generated = backend.generate_batch(
            messages, public_contexts, args.candidate_count
        )
        output_rows = []
        for row, prompt, public_context, raw_candidates in zip(
            batch, messages, public_contexts, generated
        ):
            candidate_rows = []
            for index, text in enumerate(raw_candidates):
                features, scored = candidate_features(
                    public_context, text, index=index
                )
                candidate_rows.append(
                    {
                        "index": index,
                        "text": text,
                        "text_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "parse_message": scored.parse_message,
                        "features": features,
                    }
                )
            quality_decisions.append(
                [candidate["features"] for candidate in candidate_rows]
            )
            output_rows.append(
                {
                    "scenario_id": str(row["scenario_id"]),
                    "scenario_fingerprint": str(row["scenario_fingerprint"]),
                    "task_id": str(row["task_id"]),
                    "prompt_contract": TRAINED_VDA_PROMPT_CONTRACT,
                    "prompt_sha256": hashlib.sha256(
                        prompt[0]["content"].encode("utf-8")
                    ).hexdigest(),
                    "candidates": candidate_rows,
                }
            )
        append_jsonl(rows_path, output_rows)
        partial = summarize_candidate_quality(
            quality_decisions, expected_candidates_per_decision=6
        )
        print(
            json.dumps(
                {
                    "round": args.round_index,
                    "completed_scenarios": len(quality_decisions),
                    "candidate_parse_ok_rate": partial[
                        "candidate_parse_ok_rate"
                    ],
                    "decision_base_admissible_coverage": partial[
                        "decision_base_admissible_coverage"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    quality = summarize_candidate_quality(
        quality_decisions, expected_candidates_per_decision=6
    )
    adapter_sha_after = sha256_tree(vda_manifest["adapter_path"])
    if adapter_sha_after != adapter_sha_before:
        raise LineageError("VDA adapter changed during checkpoint gate")
    manifest = {
        "schema_version": 1,
        "kind": "vda_checkpoint_inference_contract_gate",
        "status": "passed" if quality["accepted"] else "failed",
        "created_at": utc_now(),
        "dataset_role": "ecrg_calibration_non_test",
        "tmcd_test_used": False,
        "round": args.round_index,
        "scenario_count": len(selected),
        "task_counts": {task_id: args.limit_per_task for task_id in TASK_IDS},
        "candidate_count_per_decision": 6,
        "prompt_contract": TRAINED_VDA_PROMPT_CONTRACT,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "candidate_quality": quality,
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "calibration_manifest": str(calibration_manifest_path),
        "calibration_manifest_sha256": sha256_file(calibration_manifest_path),
        "vda_manifest": str(vda_manifest_path),
        "vda_manifest_sha256": sha256_file(vda_manifest_path),
        "vda_adapter_sha256_before": adapter_sha_before,
        "vda_adapter_sha256_after": adapter_sha_after,
        "candidates": str(rows_path),
        "candidates_sha256": sha256_file(rows_path),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if not quality["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
