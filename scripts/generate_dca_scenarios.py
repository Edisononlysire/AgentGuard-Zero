#!/usr/bin/env python3
"""Generate a fresh scenario pool from a trained or explicitly frozen DCA."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "third_party" / "verl"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from agentguard_zero.env.checker import full_check, parse_scenario_json
from agentguard_zero.protocol import TMCD_RELEASE_REVISION
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    load_checkpoint_manifest,
    scenario_fingerprint,
    sha256_file,
    utc_now,
)
from agentguard_zero.training.dca_dataset import (
    DCA_PROMPT_VERSION,
    TASK_FOCI,
    build_dca_messages,
)
from agentguard_zero.schemas.scenario_schema_v2 import public_prefix_hash
from agentguard_zero.variants import TRAINING_VARIANTS, experiment_variant


DCA_CANDIDATE_NORMALIZATION_VERSION = 2


def _canonicalize_candidate_identity(
    scenario: dict[str, Any],
    *,
    index: int,
    manifest: dict[str, Any],
) -> None:
    """Assign generator-owned opaque identities that cannot collapse T2 pairs."""

    if scenario.get("protocol_version") != "tmcd-v2":
        return
    scenario["split"] = "train"
    if scenario.get("scenario_family") != "trust_betrayal":
        return
    identity_payload = dict(scenario)
    identity_payload.pop("pair_id", None)
    identity_payload.pop("prefix_hash", None)
    semantic = scenario_fingerprint(identity_payload)
    scenario["pair_id"] = (
        f"pair-{manifest['backbone']}-r{int(manifest['round'])}-"
        f"{index:06d}-{semantic[:12]}"
    )
    scenario["prefix_hash"] = public_prefix_hash(scenario)


def _extract_json_object(text: str) -> tuple[dict[str, Any], bool, str]:
    scenario, ok, message = parse_scenario_json(text)
    if ok and isinstance(scenario, dict):
        return scenario, True, message
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    best_size = -1
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and end > best_size:
            best = value
            best_size = end
    if best is None:
        return {}, False, message
    return best, True, "json_object_extracted"


def _format_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return "\n\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages) + "\n\nASSISTANT:"


def _safe_full_check(scenario: dict[str, Any]) -> dict[str, Any]:
    try:
        return full_check(scenario)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {
            "all_ok": False,
            "format": {"ok": False, "error": message},
            "valid": {"ok": False, "error": message},
            "solvable": {"ok": False, "error": message},
            "safe": {"ok": False, "error": message},
        }


def _failure_reason(checks: dict[str, Any]) -> str:
    for name in ("format", "valid", "solvable", "safe"):
        result = checks.get(name, {})
        if isinstance(result, dict) and not result.get("ok", False):
            return str(result.get("message") or result.get("error") or name)
    return "ok" if checks.get("all_ok", False) else "checks_missing"


def _candidate_record(
    *,
    raw_output: str,
    index: int,
    focus: str,
    nonce: int,
    attempt: int,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_sha: str,
) -> dict[str, Any]:
    scenario, parse_ok, parse_message = _extract_json_object(raw_output)
    if scenario:
        metadata = dict(scenario.get("metadata", {}) or {})
        metadata.update(
            {
                "generator": (
                    "trained_dca_lora"
                    if manifest.get("adapter_path")
                    else "frozen_base_dca"
                ),
                "task_id": focus.split()[0],
                "task_focus": focus,
                "candidate_index": index,
                "generation_seed": args.seed,
                "generation_nonce": nonce,
                "generation_attempt": attempt,
                "generation_prompt_version": DCA_PROMPT_VERSION,
                "generated_at": utc_now(),
                "generation_shard_index": args.shard_index,
                "generation_num_shards": args.num_shards,
                "source_role": "dca",
                "source_dca_round": int(manifest["round"]),
                "source_checkpoint_manifest": str(manifest_path),
                "source_checkpoint_manifest_sha256": manifest_sha,
                "experiment_variant": args.experiment_variant,
                "tmcd_release_revision": TMCD_RELEASE_REVISION,
            }
        )
        scenario["metadata"] = metadata
        _canonicalize_candidate_identity(
            scenario,
            index=index,
            manifest=manifest,
        )
        fingerprint = scenario_fingerprint(scenario)
        scenario["scenario_id"] = (
            f"DCA-{manifest['backbone']}-R{manifest['round']}-{index:06d}-{fingerprint[:8]}"
        )
        checks = _safe_full_check(scenario) if parse_ok else {}
    else:
        fingerprint = scenario_fingerprint({"raw": raw_output})
        checks = {}
    return {
        "candidate_index": index,
        "task_focus": focus,
        "scenario_fingerprint": fingerprint,
        "duplicate": False,
        "parse_ok": parse_ok,
        "parse_message": parse_message,
        "checks": checks,
        "scenario": scenario,
        "raw_output": raw_output,
        "generation_attempt": attempt,
        "generation_prompt_version": DCA_PROMPT_VERSION,
    }


def _load_partial(path: Path, expected_config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        try:
            metadata = json.loads(first)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid candidate partial metadata: {path}") from exc
        if metadata.get("kind") != "meta" or metadata.get("config") != expected_config:
            raise RuntimeError(f"candidate partial config mismatch: {path}")
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = item.get("record") if item.get("kind") == "record" else None
            if isinstance(record, dict) and "candidate_index" in record:
                records[int(record["candidate_index"])] = record
    return records


def _append_partial(
    path: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    fsync: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(json.dumps({"kind": "meta", "config": config}, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps({"kind": "record", "record": record}, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-candidates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--max-input-tokens", type=int, default=896)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=os.environ.get("AGZ_DCA_GENERATION_ATTN_IMPLEMENTATION", "sdpa"),
    )
    parser.add_argument("--partial-fsync-every-batches", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--stop-on-complete-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--experiment-variant",
        choices=TRAINING_VARIANTS,
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_candidates <= 0 or args.batch_size <= 0 or args.num_shards <= 0:
        raise SystemExit("--num-candidates, --batch-size, and --num-shards must be positive")
    if args.partial_fsync_every_batches <= 0:
        raise SystemExit("--partial-fsync-every-batches must be positive")
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

    manifest_path = Path(args.checkpoint_manifest).resolve()
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = load_checkpoint_manifest(
        manifest_path,
        role="dca",
        backbone=str(raw_manifest.get("backbone", "")),
        round_index=int(raw_manifest.get("round", -1)),
    )
    variant = experiment_variant(args.experiment_variant)
    if int(manifest["round"]) <= 0:
        raise SystemExit("fresh VDA candidates require a DCA_{r+1} manifest")
    if str(
        (manifest.get("training_config", {}) or {}).get(
            "tmcd_release_revision", ""
        )
    ) != TMCD_RELEASE_REVISION:
        raise SystemExit("DCA checkpoint does not match the active TMCD release revision")
    if not manifest.get("adapter_path") and variant.train_dca:
        raise SystemExit("trained DCA variant is missing its DCA_{r+1} adapter")
    if not manifest.get("adapter_path") and manifest.get("status") != "frozen":
        raise SystemExit("adapter-free DCA generation requires an explicit frozen manifest")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    shard_seed = args.seed + args.shard_index * 100_003
    random.seed(shard_seed)
    torch.manual_seed(shard_seed)
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["base_model"]["path"], trust_remote_code=True, padding_side="left"
    )
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(
        manifest["base_model"]["path"], trust_remote_code=True
    )
    model_class = (
        AutoModelForVision2Seq
        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys()
        else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(
        manifest["base_model"]["path"],
        config=model_config,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    if manifest.get("adapter_path"):
        model = PeftModel.from_pretrained(model, manifest["adapter_path"])
    model.to(args.device)
    model.eval()

    output_path = Path(args.output).resolve()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.jsonl")
    manifest_sha = sha256_file(manifest_path)
    partial_config = {
        "checkpoint_manifest_sha256": manifest_sha,
        "num_candidates": args.num_candidates,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "seed": args.seed,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "attn_implementation": args.attn_implementation,
        "partial_fsync_every_batches": args.partial_fsync_every_batches,
        "max_attempts": args.max_attempts,
        "generation_prompt_version": DCA_PROMPT_VERSION,
        "candidate_normalization_version": DCA_CANDIDATE_NORMALIZATION_VERSION,
        "tmcd_release_revision": TMCD_RELEASE_REVISION,
        "experiment_variant": args.experiment_variant,
    }
    if not args.resume:
        partial_path.unlink(missing_ok=True)
    records_by_index = _load_partial(partial_path, partial_config) if args.resume else {}
    records: list[dict[str, Any]] = list(records_by_index.values())
    duplicate_count = sum(bool(record.get("duplicate")) for record in records)
    valid_count = sum(bool(record.get("checks", {}).get("all_ok")) for record in records)
    seen: set[str] = {str(record.get("scenario_fingerprint", "")) for record in records}
    assigned_indices = list(range(args.shard_index, args.num_candidates, args.num_shards))
    pending_indices = [index for index in assigned_indices if index not in records_by_index]
    for start in range(0, len(pending_indices), args.batch_size):
        batch_indices = pending_indices[start : start + args.batch_size]
        best_by_index: dict[int, dict[str, Any]] = {}
        attempts_by_index: dict[int, list[dict[str, Any]]] = {
            index: [] for index in batch_indices
        }
        retry_indices = list(batch_indices)
        for attempt in range(1, args.max_attempts + 1):
            descriptors: list[tuple[int, str, int]] = []
            prompt_texts = []
            for index in retry_indices:
                focus = TASK_FOCI[index % len(TASK_FOCI)]
                nonce_seed = args.seed + index * 1_000_003 + attempt * 15_485_863
                nonce = random.Random(nonce_seed).getrandbits(63)
                descriptors.append((index, focus, nonce))
                prompt_texts.append(
                    _format_prompt(tokenizer, build_dca_messages(focus, nonce=nonce))
                )
            encoded = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": max(0, args.top_k),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.stop_on_complete_json:
                from transformers import StoppingCriteriaList
                from agentguard_zero.json_stopping import CompleteJSONObjectCriteria

                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [CompleteJSONObjectCriteria(tokenizer, batch_size=len(retry_indices))]
                )
            with torch.inference_mode():
                generated = model.generate(**encoded, **generate_kwargs)
            input_length = encoded["input_ids"].shape[-1]
            next_retry: list[int] = []
            for descriptor, output_ids in zip(descriptors, generated):
                index, focus, nonce = descriptor
                raw_output = tokenizer.decode(
                    output_ids[input_length:], skip_special_tokens=True
                ).strip()
                candidate = _candidate_record(
                    raw_output=raw_output,
                    index=index,
                    focus=focus,
                    nonce=nonce,
                    attempt=attempt,
                    args=args,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    manifest_sha=manifest_sha,
                )
                checks = candidate.get("checks", {}) or {}
                attempts_by_index[index].append(
                    {
                        "attempt": attempt,
                        "parse_ok": bool(candidate.get("parse_ok", False)),
                        "all_checks_ok": bool(checks.get("all_ok", False)),
                        "failure_reason": _failure_reason(checks),
                        "scenario_fingerprint": candidate["scenario_fingerprint"],
                    }
                )
                previous = best_by_index.get(index)
                if (
                    previous is None
                    or bool(checks.get("all_ok", False))
                    or (
                        bool(candidate.get("parse_ok", False))
                        and not bool(previous.get("parse_ok", False))
                    )
                ):
                    best_by_index[index] = candidate
                if not bool(checks.get("all_ok", False)) and attempt < args.max_attempts:
                    next_retry.append(index)
            retry_indices = next_retry
            if not retry_indices:
                break

        batch_records: list[dict[str, Any]] = []
        for index in batch_indices:
            record = best_by_index[index]
            fingerprint = str(record["scenario_fingerprint"])
            duplicate = fingerprint in seen
            duplicate_count += int(duplicate)
            seen.add(fingerprint)
            record["duplicate"] = duplicate
            record["generation_attempts"] = attempts_by_index[index]
            records_by_index[index] = record
            records.append(record)
            batch_records.append(record)
            valid_count += int(bool((record.get("checks", {}) or {}).get("all_ok")))
        batch_number = start // args.batch_size + 1
        total_batches = (len(pending_indices) + args.batch_size - 1) // args.batch_size
        fsync = batch_number == total_batches or batch_number % args.partial_fsync_every_batches == 0
        _append_partial(partial_path, partial_config, batch_records, fsync=fsync)
        print(
            json.dumps(
                {
                    "generated": len(records),
                    "assigned": len(assigned_indices),
                    "requested_global": args.num_candidates,
                    "shard_index": args.shard_index,
                    "valid": valid_count,
                    "generation_attempts": sum(
                        len(item.get("generation_attempts", [])) for item in records
                    ),
                }
            ),
            flush=True,
        )

    records = [records_by_index[index] for index in assigned_indices]
    output = {
        "schema_version": 1,
        "kind": "dca_candidate_pool" if args.num_shards == 1 else "dca_candidate_pool_shard",
        "created_at": utc_now(),
        "seed": args.seed,
        "backbone": manifest["backbone"],
        "source_dca_round": int(manifest["round"]),
        "source_dca_checkpoint_manifest": str(manifest_path),
        "source_dca_checkpoint_manifest_sha256": manifest_sha,
        "num_candidates_requested": args.num_candidates,
        "num_candidates_assigned": len(assigned_indices),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_candidates_generated": len(records),
        "num_parse_ok": sum(item["parse_ok"] for item in records),
        "num_all_checks_ok": sum(item.get("checks", {}).get("all_ok", False) for item in records),
        "num_duplicates": duplicate_count,
        "generation_prompt_version": DCA_PROMPT_VERSION,
        "candidate_normalization_version": DCA_CANDIDATE_NORMALIZATION_VERSION,
        "tmcd_release_revision": TMCD_RELEASE_REVISION,
        "max_attempts": args.max_attempts,
        "generation_config": partial_config,
        "experiment_variant": args.experiment_variant,
        "candidates": records,
    }
    atomic_write_json(args.output, output)
    partial_path.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in output.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
