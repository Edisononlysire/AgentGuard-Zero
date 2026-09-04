#!/usr/bin/env python3
"""Run one frozen V11 expert through the common learned hierarchical interface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.expert_router import (
    EXPERT_NAMES,
    load_expert_policy,
)
from agentguard_zero.recovery.public_teacher import public_state_digest
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now
from scripts import eval_candidate_policy as frozen_eval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expert-name", choices=EXPERT_NAMES, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-source", type=Path, required=True)
    parser.add_argument("--scenario-count", type=int, default=200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--protocol-role", default="fair_trajectory")
    parser.add_argument("--post-mitigation-audit-step", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    policy = load_expert_policy(
        model_path=args.model_path,
        manifest_path=args.expert_manifest,
        device=args.device,
        max_length=args.max_length,
        score_batch_size=args.score_batch_size,
    )
    latencies: dict[str, list[float]] = defaultdict(list)
    original_decide = policy.decide

    def instrumented_decide(
        observation: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        started = time.perf_counter()
        decision = original_decide(observation, **kwargs)
        latencies[public_state_digest(observation)].append(
            (time.perf_counter() - started) * 1000.0
        )
        return decision

    policy.decide = instrumented_decide  # type: ignore[method-assign]
    original_factory = frozen_eval.CandidateRankerPolicy
    frozen_eval.CandidateRankerPolicy = lambda **_kwargs: policy
    base_output = args.output.with_suffix(args.output.suffix + ".base.json")
    if base_output.exists():
        raise FileExistsError(f"refusing to overwrite {base_output}")
    namespace = SimpleNamespace(
        model_path=args.model_path,
        ranker_manifest=args.expert_manifest,
        policy_mode="candidate_ranker",
        ecrg_config=None,
        progressive_arm=None,
        selector_temperature=1.0,
        output=base_output,
        scenario_count=args.scenario_count,
        scenario_source=args.scenario_source,
        task_filter=None,
        active_probe_budget=None,
        group_offset=0,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        device=args.device,
        max_length=args.max_length,
        score_batch_size=args.score_batch_size,
        evaluation_seed=args.evaluation_seed,
        memory_counterfactual="none",
        probe_counterfactual="none",
        intervention_consistent_candidates=False,
        skip_teacher_audit=True,
        post_mitigation_audit_step=args.post_mitigation_audit_step,
        protocol_role=args.protocol_role,
    )
    try:
        status = frozen_eval.run(namespace)
    finally:
        frozen_eval.CandidateRankerPolicy = original_factory
    if status != 0:
        return int(status)
    payload = json.loads(base_output.read_text(encoding="utf-8"))
    values = []
    for trace in payload.get("traces", []):
        queue = latencies.get(str(trace.get("public_state_digest", "")), [])
        if not queue:
            raise RuntimeError("single-expert trace instrumentation mismatch")
        latency = queue.pop(0)
        trace["expert_inference_latency_ms"] = latency
        values.append(latency)
    if any(latencies.values()):
        raise RuntimeError("unused single-expert latency records")
    payload.update(
        {
            "schema_version": 1,
            "kind": "v11_single_expert_policy_evaluation_shard",
            "created_at": utc_now(),
            "policy_mode": f"v11_single_expert_{args.expert_name}",
            "expert_name": args.expert_name,
            "expert_manifest": str(args.expert_manifest.resolve()),
            "expert_manifest_sha256": sha256_file(args.expert_manifest),
            "historical_capability_chain_used": False,
            "forced_common_selection_mode": (
                "learned_hierarchical_family_then_utility"
            ),
            "base_evaluation_audit": str(base_output.resolve()),
            "base_evaluation_audit_sha256": sha256_file(base_output),
            "hidden_state_in_model_input": False,
            "teacher_controls_actions": False,
            "dca_frozen": True,
            "ecrg_enabled": False,
            "formal_three_rounds_started": False,
        }
    )
    payload["metrics"]["decoding"] = (
        "v11_single_expert_learned_hierarchical_family_then_utility"
    )
    payload["metrics"]["expert_inference_latency_ms_mean"] = sum(values) / max(
        1, len(values)
    )
    payload["metrics"]["expert_inference_latency_ms_max"] = max(
        values, default=0.0
    )
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
