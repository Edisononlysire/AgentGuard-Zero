#!/usr/bin/env python3
"""Generate frozen-VDA K=6 decision traces for ECRG calibration.

The VDA and DCA are read-only. Hidden simulator state is used only to attach
offline counterfactual outcome labels; ECRG features are public-state only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_tmcd_systems as ev
from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.governance.v5c import safe_probe_fallback
from agentguard_zero.inference_contract import (
    FORMAL_VDA_MAX_NEW_TOKENS,
    TRAINED_VDA_PROMPT_CONTRACT,
    summarize_candidate_quality,
)
from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    canonical_json,
    load_checkpoint_manifest,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_tree,
    utc_now,
)
from ecrg_calibration_lib import candidate_features
from level1_rollout_server import Level1RolloutStore
from vda_feedback_server import (
    _generation_messages,
    _history_summary,
    _instruction_messages,
)


TASK_IDS = ("T1", "T2", "T3", "T4")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("scenario_id"):
                result[str(row["scenario_id"])] = row
    return result


def _append_progress(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _counterfactual_label(
    env: Any,
    scenario: dict[str, Any],
    packet: dict[str, Any],
    *,
    parse_ok: bool,
    turn: int,
) -> dict[str, Any]:
    """Evaluate one candidate, then use an oracle continuation for labels only."""

    clone = copy.deepcopy(env)
    clone.oracle_mode = False
    _, _, done = clone.step(copy.deepcopy(packet))
    first_step = copy.deepcopy(clone.history[-1] if clone.history else {})
    oracle_steps = 0
    while not done:
        clone.oracle_mode = True
        public_context = {"observation": clone.observe()}
        oracle_packet = ev.oracle_action(public_context, scenario, turn + 1 + oracle_steps)
        _, _, done = clone.step(oracle_packet)
        oracle_steps += 1
    score = score_trajectory_v2(clone)
    steps = len(clone.history)
    budget = float(scenario.get("defense_constraints", {}).get("business_budget", 5.0))
    utility = ev.base.compute_safe_utility(
        score,
        steps=steps,
        max_steps=int(getattr(clone, "max_steps", max(1, steps))),
        budget=budget,
    )
    if not parse_ok:
        utility -= 0.5
    response = first_step.get("response_result", {}) or {}
    proposed = str(response.get("proposed_action", "Observe"))
    high_impact = proposed in {"Isolate", "Restore", "Remove"}
    authorized = bool(response.get("authorized", False))
    evidence_ids = list(response.get("authorization_evidence_ids", []) or [])
    unsupported_high_impact = bool(high_impact and (not authorized or not evidence_ids))
    return {
        "label_access": "hidden_state_offline_only",
        "counterfactual_utility": float(utility),
        "trajectory_reward": float(score.get("reward", 0.0)),
        "attack_mitigated": bool(score.get("attack_mitigated", False)),
        "attack_success": bool(score.get("attack_success", False)),
        "overresponse": bool(score.get("overresponse", False)),
        "business_cost": float(score.get("business_cost", 0.0)),
        "verification_cost": float(score.get("verification_cost", 0.0)),
        "first_response_authorized": authorized,
        "first_response_blocked": bool(response.get("blocked", False)),
        "first_authorization_reason": str(response.get("authorization_reason", "")),
        "evidence_supported_execution": bool(
            authorized and (not high_impact or evidence_ids)
        ),
        "unsupported_high_impact": unsupported_high_impact,
        "oracle_continuation_steps": oracle_steps,
    }


def _candidate_entry(
    public_context: dict[str, Any],
    text: str,
    *,
    index: int,
    env: Any,
    scenario: dict[str, Any],
    turn: int,
) -> dict[str, Any]:
    features, scored = candidate_features(public_context, text, index=index)
    label = _counterfactual_label(
        env,
        scenario,
        scored.packet,
        parse_ok=scored.parse_ok,
        turn=turn,
    )
    return {
        "index": index,
        "text": text,
        "text_sha256": sha256_bytes(text.encode("utf-8")),
        "features": features,
        "label": label,
    }


def _trace_batch(
    indexed_rows: list[tuple[int, dict[str, Any]]],
    backend: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    store = Level1RolloutStore(
        invalid_penalty=args.invalid_penalty,
        max_parallel_trajectories=max(1, len(indexed_rows)),
    )
    states = []
    for row_index, row in indexed_rows:
        messages, public_context, extra, scenario, scenario_id, max_env_steps, budget = ev.row_context(
            row, row_index, args
        )
        trajectory_id = f"{args.run_name}-trace-{row_index}-{scenario_id}"
        rollout_state = store._get_or_create_state(trajectory_id, extra)
        states.append(
            {
                "row_index": row_index,
                "row": row,
                "task_id": str(row.get("task_id", "")),
                "scenario_fingerprint": str(row.get("scenario_fingerprint", "")),
                "scenario_id": scenario_id,
                "scenario": scenario,
                "extra": extra,
                "initial_messages": messages,
                "instruction_messages": _instruction_messages(messages),
                "continuation_prompt_mode": "snapshot",
                "history_window": 8,
                "history": [],
                "public_context": public_context,
                "max_turns": min(args.max_turns, max_env_steps),
                "trajectory_id": trajectory_id,
                "rollout_state": rollout_state,
                "decisions": [],
                "done": False,
            }
        )

    with ThreadPoolExecutor(max_workers=args.label_workers) as executor:
        for turn in range(max(state["max_turns"] for state in states)):
            active = [
                state for state in states if not state["done"] and turn < state["max_turns"]
            ]
            if not active:
                break
            message_batches = [_generation_messages(state) for state in active]
            contexts = [state["public_context"] for state in active]
            if hasattr(backend, "generate_batch"):
                candidate_batches = backend.generate_batch(
                    message_batches, contexts, args.candidate_count
                )
            else:
                candidate_batches = [
                    backend.generate(messages, context, args.candidate_count)
                    for messages, context in zip(message_batches, contexts)
                ]
            selected_texts = []
            selected_packets = []
            for state, raw_candidates in zip(active, candidate_batches):
                public_context = state["public_context"]
                raw_candidates = [
                    ev.postprocess_candidate(args.system, item, public_context)
                    for item in raw_candidates
                ]
                futures = [
                    executor.submit(
                        _candidate_entry,
                        public_context,
                        text,
                        index=index,
                        env=state["rollout_state"].env,
                        scenario=state["scenario"],
                        turn=turn,
                    )
                    for index, text in enumerate(raw_candidates)
                ]
                candidate_rows = [future.result() for future in futures]
                scored = [
                    candidate_features(public_context, text, index=index)[1]
                    for index, text in enumerate(raw_candidates)
                ]
                fallback_packet, fallback_diagnostics = safe_probe_fallback(
                    public_context, scored
                )
                fallback_text = _json(fallback_packet)
                fallback = _candidate_entry(
                    public_context,
                    fallback_text,
                    index=-1,
                    env=state["rollout_state"].env,
                    scenario=state["scenario"],
                    turn=turn,
                )
                fallback["fallback_diagnostics"] = fallback_diagnostics
                selected = ev.select_runtime_candidate(
                    "agentguard_zero_select",
                    public_context,
                    raw_candidates,
                    ev.model_policy(args.system),
                    selector_mode=args.selector_mode,
                    seed=args.seed,
                )
                selected_index = next(
                    (
                        index
                        for index, text in enumerate(raw_candidates)
                        if text == selected.text
                    ),
                    -1,
                )
                state["decisions"].append(
                    {
                        "turn": turn,
                        "public_state_sha256": sha256_bytes(
                            canonical_json(public_context).encode("utf-8")
                        ),
                        "candidate_count": len(candidate_rows),
                        "candidates": candidate_rows,
                        "fallback": fallback,
                        "behavior_policy": "reference_v5c_before_calibration",
                        "behavior_selected_index": selected_index,
                    }
                )
                selected_texts.append(selected.text)
                selected_packets.append(selected.packet)

            response = store.handle(
                {
                    "trajectory_ids": [state["trajectory_id"] for state in active],
                    "actions": selected_texts,
                    "finish": [False] * len(active),
                    "is_last_step": [
                        turn + 1 >= state["max_turns"] for state in active
                    ],
                    "extra_fields": [state["extra"] for state in active],
                }
            )
            for position, state in enumerate(active):
                state["done"] = bool(response["dones"][position])
                if not state["done"]:
                    _, public_context = ev.base.next_user_message(
                        response["observations"][position]
                    )
                    state["history"].append(
                        _history_summary(turn, public_context, selected_packets[position])
                    )
                    state["public_context"] = public_context

    return [
        {
            "schema_version": 1,
            "kind": "ecrg_calibration_scenario_trace",
            "scenario_id": state["scenario_id"],
            "scenario_fingerprint": state["scenario_fingerprint"],
            "task_id": state["task_id"],
            "row_index": state["row_index"],
            "candidate_count": args.candidate_count,
            "decisions": state["decisions"],
            "decision_count": len(state["decisions"]),
            "feature_access": "public_only",
            "label_access": "hidden_state_offline_only",
        }
        for state in states
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--data", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--vda-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--trajectory-batch-size", type=int, default=4)
    parser.add_argument("--label-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--model-backend", choices=["hf", "mock"], default="hf")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-new-tokens", type=int, default=FORMAL_VDA_MAX_NEW_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.candidate_count != 6:
        parser.error("formal ECRG calibration requires K=6")
    if args.max_new_tokens != FORMAL_VDA_MAX_NEW_TOKENS:
        parser.error(
            "formal ECRG calibration must match the trained VDA action budget "
            f"({FORMAL_VDA_MAX_NEW_TOKENS})"
        )
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    # Calibration samples the frozen trained VDA.  AgentGuard-Zero-Full exists
    # only after the resulting ECRG config has been fitted and frozen.
    args.system = "agentguard_zero_train"
    args.selector_mode = "v5_c_evidence_governor"
    args.invalid_penalty = 0.5
    args.run_name = f"ecrg_cal_{args.task_id.lower()}"
    args.do_sample = True
    args.stop_on_complete_json = True
    args.device_map = ""
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    calibration_manifest_path = Path(args.calibration_manifest).resolve()
    calibration_manifest = read_json(calibration_manifest_path)
    if calibration_manifest.get("status") != "sealed" or int(
        calibration_manifest.get("selected_count", -1)
    ) != 800:
        raise LineageError("ECRG-Cal manifest must seal exactly 800 scenarios")
    if calibration_manifest.get("task_counts") != {
        "T1": 200,
        "T2": 200,
        "T3": 200,
        "T4": 200,
    }:
        raise LineageError("ECRG-Cal is not balanced")

    vda_manifest_path = Path(args.vda_manifest).resolve()
    vda_manifest = load_checkpoint_manifest(
        vda_manifest_path,
        role="vda",
        backbone="qwen3.5-4b",
        round_index=3,
    )
    args.model_path = str(vda_manifest["base_model"]["path"])
    args.adapter_path = str(vda_manifest["adapter_path"])
    args.model_backend = args.model_backend
    args.api_model = ""
    args.api_base_url = ""
    args.api_key_env = ""
    args.api_timeout = 90
    args.api_retries = 0
    args.api_response_format_json = False
    args.api_disable_thinking = True
    args.api_multi_choice = False
    args.api_system_prompt = ""

    frame = pd.read_parquet(Path(args.data).resolve())
    frame = frame[frame["task_id"] == args.task_id].copy()
    frame = frame.sort_values("scenario_fingerprint", kind="mergesort")
    if args.limit > 0:
        frame = frame.head(args.limit)
    if len(frame) != args.expected_count:
        raise LineageError(
            f"{args.task_id} calibration rows={len(frame)}, expected={args.expected_count}"
        )
    rows = frame.to_dict("records")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "traces.jsonl"
    manifest_path = output_dir / "manifest.json"
    run_config = {
        "task_id": args.task_id,
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(args.data),
        "calibration_manifest_sha256": sha256_file(calibration_manifest_path),
        "vda_manifest_sha256": sha256_file(vda_manifest_path),
        "vda_adapter_sha256": vda_manifest["adapter_sha256"],
        "candidate_count": args.candidate_count,
        "max_turns": args.max_turns,
        "seed": args.seed,
        "model_backend": args.model_backend,
        "expected_count": args.expected_count,
        "prompt_contract": ev.inference_prompt_contract(args.system),
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "do_sample": bool(args.do_sample),
        "stop_on_complete_json": bool(args.stop_on_complete_json),
    }
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists() and args.resume:
        if read_json(run_config_path) != run_config:
            raise LineageError("ECRG calibration trace resume config mismatch")
    else:
        atomic_write_json(run_config_path, run_config)
        progress_path.unlink(missing_ok=True)
    existing = _load_progress(progress_path) if args.resume else {}
    pending = [
        (index, row)
        for index, row in enumerate(rows)
        if str(row["scenario_id"]) not in existing
    ]
    backend = ev.build_backend(args)
    for start in range(0, len(pending), args.trajectory_batch_size):
        batch = pending[start : start + args.trajectory_batch_size]
        traced = _trace_batch(batch, backend, args)
        _append_progress(progress_path, traced)
        for row in traced:
            existing[row["scenario_id"]] = row
        print(
            _json(
                {
                    "task_id": args.task_id,
                    "completed": len(existing),
                    "expected": args.expected_count,
                    "decision_count": sum(
                        int(row.get("decision_count", 0)) for row in existing.values()
                    ),
                }
            ),
            flush=True,
        )

    if len(existing) != args.expected_count:
        raise LineageError("ECRG trace count incomplete")
    adapter_sha_after = sha256_tree(args.adapter_path)
    if adapter_sha_after != vda_manifest["adapter_sha256"]:
        raise LineageError("frozen VDA adapter changed during ECRG trace generation")
    trace_rows = sorted(existing.values(), key=lambda row: row["scenario_fingerprint"])
    decision_count = sum(int(row.get("decision_count", 0)) for row in trace_rows)
    if any(
        row.get("candidate_count") != 6
        or row.get("feature_access") != "public_only"
        or row.get("label_access") != "hidden_state_offline_only"
        for row in trace_rows
    ):
        raise LineageError("ECRG trace access or K invariant failed")
    candidate_quality = summarize_candidate_quality(
        (
            [candidate.get("features", {}) for candidate in decision.get("candidates", [])]
            for row in trace_rows
            for decision in row.get("decisions", [])
        ),
        expected_candidates_per_decision=6,
    )
    status = "sealed" if candidate_quality["accepted"] else "rejected"
    manifest = {
        "schema_version": 1,
        "kind": "ecrg_calibration_trace_shard",
        "status": status,
        "sealed_at": utc_now(),
        "task_id": args.task_id,
        "scenario_count": len(trace_rows),
        "decision_count": decision_count,
        "candidate_count_per_decision": 6,
        "prompt_contract": TRAINED_VDA_PROMPT_CONTRACT,
        "max_new_tokens": FORMAL_VDA_MAX_NEW_TOKENS,
        "candidate_quality": candidate_quality,
        "feature_access": "public_only",
        "label_access": "hidden_state_offline_only",
        "parameter_training": False,
        "vda_frozen": True,
        "dca_used": False,
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(args.data),
        "calibration_manifest": str(calibration_manifest_path),
        "calibration_manifest_sha256": sha256_file(calibration_manifest_path),
        "vda_manifest": str(vda_manifest_path),
        "vda_manifest_sha256": sha256_file(vda_manifest_path),
        "vda_adapter_sha256_before": vda_manifest["adapter_sha256"],
        "vda_adapter_sha256_after": adapter_sha_after,
        "trace": str(progress_path),
        "trace_sha256": sha256_file(progress_path),
        "run_config": str(run_config_path),
        "run_config_sha256": sha256_file(run_config_path),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if status != "sealed":
        raise LineageError(
            "ECRG trace rejected by candidate-quality gate: "
            f"{candidate_quality['failures']}"
        )


if __name__ == "__main__":
    main()
