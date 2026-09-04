#!/usr/bin/env python3
"""Merge V11 single/composed shards with the unchanged frozen metric engine."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.candidate.expert_router import EXPERT_NAMES, FAMILY_NAMES
from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now
from scripts import eval_candidate_policy as frozen_eval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not shards:
        raise ValueError("no V11 evaluation shards")
    kinds = {str(shard.get("kind", "")) for shard in shards}
    if len(kinds) != 1 or kinds.pop() not in {
        "v11_single_expert_policy_evaluation_shard",
        "v11_unified_router_policy_evaluation_shard",
    }:
        raise RuntimeError("V11 shard kind mismatch")
    invariant = (
        "policy_mode",
        "scenario_source_sha256",
        "evaluation_seed",
        "suite_semantic_sha256",
        "protocol_role",
    )
    for key in invariant:
        if len({json.dumps(row.get(key), sort_keys=True) for row in shards}) != 1:
            raise RuntimeError(f"V11 policy shard invariant mismatch: {key}")

    base_output = args.output.with_suffix(args.output.suffix + ".base.json")
    if base_output.exists():
        raise FileExistsError(f"refusing to overwrite {base_output}")
    frozen_eval.merge(SimpleNamespace(inputs=args.inputs, output=base_output))
    payload = json.loads(base_output.read_text(encoding="utf-8"))
    traces = payload.get("traces", [])
    first = shards[0]
    is_router = str(first["kind"]).startswith("v11_unified_router")
    route_sums: dict[str, dict[str, float]] = {
        expert: {family: 0.0 for family in FAMILY_NAMES}
        for expert in EXPERT_NAMES
    }
    route_count = 0
    latencies = []
    for trace in traces:
        if is_router:
            weights = trace.get("router_weights") or {}
            if set(weights) != set(EXPERT_NAMES):
                raise RuntimeError("Router trace lacks 3-expert weights")
            for expert in EXPERT_NAMES:
                if set(weights[expert]) != set(FAMILY_NAMES):
                    raise RuntimeError("Router trace lacks 6-family weights")
                for family in FAMILY_NAMES:
                    route_sums[expert][family] += float(weights[expert][family])
            route_count += 1
            latencies.append(float(trace["router_inference_latency_ms"]))
        else:
            latencies.append(float(trace["expert_inference_latency_ms"]))
    payload.update(
        {
            "schema_version": 1,
            "kind": (
                "v11_unified_router_policy_evaluation"
                if is_router
                else "v11_single_expert_policy_evaluation"
            ),
            "created_at": utc_now(),
            "policy_mode": first["policy_mode"],
            "base_evaluation_audit": str(base_output.resolve()),
            "base_evaluation_audit_sha256": sha256_file(base_output),
            "source_shard_sha256": {
                path.name: sha256_file(path) for path in args.inputs
            },
            "hidden_state_in_model_input": False,
            "teacher_controls_actions": False,
            "dca_frozen": True,
            "ecrg_enabled": False,
            "formal_three_rounds_started": False,
        }
    )
    for key in (
        "router_manifest",
        "router_manifest_sha256",
        "expert_manifests",
        "expert_manifest_sha256",
        "routing_mode",
        "expert_name",
        "expert_manifest",
        "historical_capability_chain_used",
        "forced_common_selection_mode",
    ):
        if key in first:
            payload[key] = first[key]
    payload["metrics"]["inference_latency_ms_mean"] = sum(latencies) / max(
        1, len(latencies)
    )
    payload["metrics"]["inference_latency_ms_max"] = max(latencies, default=0.0)
    if is_router:
        payload["metrics"]["mean_router_weights"] = {
            expert: {
                family: route_sums[expert][family] / max(1, route_count)
                for family in FAMILY_NAMES
            }
            for expert in EXPERT_NAMES
        }
        dominant = Counter()
        for trace in traces:
            weights = trace["router_weights"]
            dominant[
                max(
                    (
                        (float(weights[expert][family]), expert, family)
                        for expert in EXPERT_NAMES
                        for family in FAMILY_NAMES
                    )
                )[1:]
            ] += 1
        payload["metrics"]["dominant_route_cell_counts"] = {
            f"{expert}:{family}": count
            for (expert, family), count in sorted(dominant.items())
        }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
