#!/usr/bin/env python3
"""Bind the accepted T1/T2 data, Dev suite, D0, and retention reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--epoch-dev-protocol", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--dev-scenarios", type=Path, required=True)
    parser.add_argument("--dev-scenarios-manifest", type=Path, required=True)
    parser.add_argument("--d0-manifest", type=Path, required=True)
    parser.add_argument("--old-v9-manifest", type=Path, required=True)
    parser.add_argument("--frozen-suite-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-extension", type=Path, required=True)
    parser.add_argument("--evaluator-source", type=Path, required=True)
    parser.add_argument("--public-legality-gate-source", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    parent = _read(args.parent_protocol)
    epoch_dev_protocol = _read(args.epoch_dev_protocol)
    data_audit = _read(args.data_audit)
    train = _read(args.train_manifest)
    dev = _read(args.dev_manifest)
    dev_suite = _read(args.dev_scenarios_manifest)
    d0 = _read(args.d0_manifest)
    old_v9 = _read(args.old_v9_manifest)
    frozen_suite = _read(args.frozen_suite_manifest)
    evaluator_extension = _read(args.evaluator_extension)
    checks = {
        "parent_protocol": parent.get("kind")
        == "v11_three_expert_router_preregistration",
        "parent_frozen": parent.get("status") == "frozen_before_data_generation",
        "epoch_dev_protocol": epoch_dev_protocol.get("kind")
        == "v11_t12_epoch_selection_dev_preregistration",
        "epoch_dev_parent_binding": epoch_dev_protocol.get(
            "parent_protocol_sha256"
        )
        == sha256_file(args.parent_protocol),
        "data_audit_accepted": data_audit.get("accepted") is True,
        "train_accepted": train.get("accepted") is True,
        "dev_accepted": dev.get("accepted") is True,
        "train_exact": train.get("task_record_counts")
        == {"T1": 2000, "T2": 2000},
        "dev_exact": dev.get("task_record_counts") == {"T1": 200, "T2": 200},
        "epoch_dev_accepted": dev_suite.get("accepted") is True,
        "epoch_dev_exact": dev_suite.get("task_counts")
        == {"T1": 200, "T2": 200},
        "epoch_dev_not_formal": dev_suite.get("formal_frozen_test") is False,
        "d0_kind": d0.get("kind") == "candidate_ranker_checkpoint",
        "d0_adapter_hash_present": bool(d0.get("adapter_sha256")),
        "d0_heads_hash_present": bool(d0.get("heads_sha256")),
        "old_v9_reference": old_v9.get("kind") == "candidate_ranker_checkpoint",
        "old_v9_same_d0_adapter": old_v9.get("initialization_adapter_sha256")
        == d0.get("adapter_sha256"),
        "old_v9_same_d0_heads": old_v9.get("initialization_heads_sha256")
        == d0.get("heads_sha256"),
        "frozen_suite_accepted": frozen_suite.get("accepted") is True,
        "frozen_suite_hash": sha256_file(args.frozen_suite_manifest)
        == (parent.get("final_evaluation") or {}).get(
            "frozen_suite_manifest_sha256"
        ),
        "evaluator_extension_frozen": evaluator_extension.get("kind")
        == "v11_evaluator_protocol_role_extension"
        and evaluator_extension.get("status") == "frozen_before_t12_training",
        "evaluator_extension_is_protocol_only": evaluator_extension.get(
            "scoring_functions_changed"
        )
        is False
        and evaluator_extension.get("terminal_metrics_changed") is False
        and evaluator_extension.get("frozen_600_role_changed") is False,
        "evaluator_source_binding": sha256_file(args.evaluator_source)
        == evaluator_extension.get("source_sha256_after_extension"),
        "public_legality_gate_source_binding": sha256_file(
            args.public_legality_gate_source
        )
        == evaluator_extension.get("public_legality_gate_source_sha256"),
    }
    payload = {
        "schema_version": 1,
        "kind": "v11_t12_expert_training_protocol",
        "created_at": utc_now(),
        "status": "frozen_after_data_before_training",
        "accepted": all(checks.values()),
        "checks": checks,
        "failures": [key for key, passed in checks.items() if not passed],
        "artifacts": {
            "parent_protocol": str(args.parent_protocol.resolve()),
            "parent_protocol_sha256": sha256_file(args.parent_protocol),
            "epoch_dev_protocol": str(args.epoch_dev_protocol.resolve()),
            "epoch_dev_protocol_sha256": sha256_file(args.epoch_dev_protocol),
            "data_audit": str(args.data_audit.resolve()),
            "data_audit_sha256": sha256_file(args.data_audit),
            "train_jsonl": str(args.train_jsonl.resolve()),
            "train_jsonl_sha256": sha256_file(args.train_jsonl),
            "train_manifest": str(args.train_manifest.resolve()),
            "train_manifest_sha256": sha256_file(args.train_manifest),
            "dev_jsonl": str(args.dev_jsonl.resolve()),
            "dev_jsonl_sha256": sha256_file(args.dev_jsonl),
            "dev_manifest": str(args.dev_manifest.resolve()),
            "dev_manifest_sha256": sha256_file(args.dev_manifest),
            "dev_scenarios": str(args.dev_scenarios.resolve()),
            "dev_scenarios_sha256": sha256_file(args.dev_scenarios),
            "dev_scenarios_manifest": str(
                args.dev_scenarios_manifest.resolve()
            ),
            "dev_scenarios_manifest_sha256": sha256_file(
                args.dev_scenarios_manifest
            ),
            "d0_manifest": str(args.d0_manifest.resolve()),
            "d0_manifest_sha256": sha256_file(args.d0_manifest),
            "d0_adapter_sha256": d0.get("adapter_sha256"),
            "d0_heads_sha256": d0.get("heads_sha256"),
            "old_v9_manifest": str(args.old_v9_manifest.resolve()),
            "old_v9_manifest_sha256": sha256_file(args.old_v9_manifest),
            "frozen_suite_manifest": str(args.frozen_suite_manifest.resolve()),
            "frozen_suite_manifest_sha256": sha256_file(
                args.frozen_suite_manifest
            ),
            "evaluator_extension": str(args.evaluator_extension.resolve()),
            "evaluator_extension_sha256": sha256_file(
                args.evaluator_extension
            ),
            "evaluator_source": str(args.evaluator_source.resolve()),
            "evaluator_source_sha256": sha256_file(args.evaluator_source),
            "public_legality_gate_source": str(
                args.public_legality_gate_source.resolve()
            ),
            "public_legality_gate_source_sha256": sha256_file(
                args.public_legality_gate_source
            ),
        },
        "training": {
            "initialization": "common_d0_only",
            "inherits_v9_checkpoint": False,
            "backbone": "Qwen3.5-4B",
            "lora_rank": 16,
            "architecture": "shared_local_long_v1",
            "objective": "branched_defense",
            "selection_mode": "hierarchical_family_then_utility",
            "memory_tier_encoding": True,
            "local_heads_initialized_from_d0_legacy_heads": True,
            "epochs": 4,
            "global_batch": 16,
            "optimizer_steps": 1000,
            "training_examples_seen": 16000,
            "save_each_epoch": True,
            "formal_frozen_600_during_training": False,
        },
        "epoch_selection": {
            "source": "t12_epoch_selection_dev_400_only",
            "scenario_count": 400,
            "evaluation_seed": 2026072303,
            "lexicographic": [
                "t12_macro_safe_success_desc",
                "attack_mitigation_desc",
                "attack_success_asc",
                "intent_accuracy_desc",
                "overresponse_asc",
                "business_cost_asc"
            ],
            "formal_frozen_600_visible": False,
        },
        "retention_gate": {
            "reference": "old_v9_hierarchical_distill",
            "suite": "fair_s1_fair_s2_fair_s3_total_600",
            "evaluated_after_epoch_selection": True,
            "test_result_used_for_retraining": False,
            "failure_stops_router": True,
        },
        "controls": {
            "teacher_directory_read_only": True,
            "ecrg_enabled": False,
            "dagger_enabled": False,
            "formal_three_rounds_started": False,
            "router_training_started": False,
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
