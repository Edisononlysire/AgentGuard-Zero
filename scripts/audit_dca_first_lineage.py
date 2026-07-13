#!/usr/bin/env python3
"""Audit a complete DCA-first checkpoint/data lineage from disk."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    canonical_json,
    load_checkpoint_manifest,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    validate_round_lineage,
)


def audit_lineage(
    root: Path,
    backbone: str,
    artifact_scope: str,
    max_round: int,
    expected_host: str | None = None,
) -> dict:
    checkpoint_root = root / ("checkpoints" if artifact_scope == "formal" else "checkpoints_pilot")
    data_root = root / "data" / (
        "co_evolution" if artifact_scope == "formal" else "co_evolution_pilot"
    )
    role_manifests: dict[str, list[dict]] = {"dca": [], "vda": []}
    for role in ("dca", "vda"):
        base_path = checkpoint_root / backbone / role / "round_0" / "manifest.json"
        role_manifests[role].append(
            load_checkpoint_manifest(base_path, role=role, backbone=backbone, round_index=0)
        )

    rounds = []
    for round_index in range(1, max_round + 1):
        manifests = {}
        manifest_paths = {}
        for role in ("dca", "vda"):
            path = checkpoint_root / backbone / role / f"round_{round_index}" / "manifest.json"
            manifest = load_checkpoint_manifest(
                path, role=role, backbone=backbone, round_index=round_index
            )
            parent_path = (
                checkpoint_root / backbone / role / f"round_{round_index - 1}" / "manifest.json"
            ).resolve()
            if Path(manifest.get("parent_manifest", "")).resolve() != parent_path:
                raise LineageError(f"{role} round {round_index} parent path mismatch")
            if manifest.get("parent_manifest_sha256") != sha256_file(parent_path):
                raise LineageError(f"{role} round {round_index} parent hash mismatch")
            config = manifest.get("training_config", {}) or {}
            expected_config_hash = sha256_bytes(canonical_json(config).encode("utf-8"))
            if manifest.get("training_config_sha256") != expected_config_hash:
                raise LineageError(f"{role} round {round_index} training config hash mismatch")
            if int(config.get("world_size", -1)) != 4:
                raise LineageError(f"{role} round {round_index} was not trained with four ranks")
            if expected_host and config.get("execution_host") != expected_host:
                raise LineageError(
                    f"{role} round {round_index} ran on {config.get('execution_host')}, "
                    f"expected {expected_host}"
                )
            manifests[role] = manifest
            manifest_paths[role] = path
            role_manifests[role].append(manifest)
        if manifests["dca"].get("adapter_sha256") == manifests["vda"].get("adapter_sha256"):
            raise LineageError(f"round {round_index} DCA/VDA adapters have the same hash")

        round_dir = data_root / backbone / f"round_{round_index}"
        feedback_log = round_dir / "dca_feedback" / "feedback.jsonl"
        feedback_manifest_path = round_dir / "dca_feedback" / "manifest.json"
        pool_manifest_path = round_dir / "vda_pool_manifest.json"
        split_paths = [
            round_dir / "vda_train" / "train.parquet",
            round_dir / "vda_dev" / "dev.parquet",
            round_dir / "vda_xplay" / "xplay.parquet",
        ]
        lineage = validate_round_lineage(
            dca_manifest_path=str(manifest_paths["dca"]),
            feedback_log_path=str(feedback_log),
            pool_manifest_path=str(pool_manifest_path),
            split_paths=[str(path) for path in split_paths],
            backbone=backbone,
            target_round=round_index,
        )
        feedback_manifest = read_json(feedback_manifest_path)
        expected_parent_paths = {
            role: checkpoint_root
            / backbone
            / role
            / f"round_{round_index - 1}"
            / "manifest.json"
            for role in ("dca", "vda")
        }
        for role in ("dca", "vda"):
            key = f"source_{role}_manifest_sha256"
            if feedback_manifest.get(key) != sha256_file(expected_parent_paths[role]):
                raise LineageError(
                    f"round {round_index} feedback did not use current {role.upper()}_{round_index - 1}"
                )
        if Path(manifests["dca"].get("training_data_manifest", "")).resolve() != feedback_manifest_path.resolve():
            raise LineageError(f"round {round_index} DCA checkpoint does not cite its feedback manifest")
        if manifests["dca"].get("training_data_manifest_sha256") != sha256_file(
            feedback_manifest_path
        ):
            raise LineageError(f"round {round_index} DCA feedback manifest hash mismatch")
        pool_manifest = read_json(pool_manifest_path)
        if Path(manifests["vda"].get("training_data_manifest", "")).resolve() != pool_manifest_path.resolve():
            raise LineageError(f"round {round_index} VDA checkpoint does not cite its fresh pool")
        if manifests["vda"].get("training_data_manifest_sha256") != sha256_file(
            pool_manifest_path
        ):
            raise LineageError(f"round {round_index} VDA pool manifest hash mismatch")
        expected_splits = manifests["vda"]["training_config"]
        expected_counts = {
            "train": int(expected_splits["train_size"]),
            "dev": int(expected_splits["dev_size"]),
            "xplay": int(expected_splits["xplay_size"]),
        }
        if pool_manifest.get("split_counts") != expected_counts:
            raise LineageError(f"round {round_index} VDA split counts mismatch")

        reload_reports = {}
        for role in ("dca", "vda"):
            report_path = round_dir / f"adapter_reload_{role}.json"
            report = read_json(report_path)
            if not report.get("reload_ok", False):
                raise LineageError(f"round {round_index} {role} reload report failed")
            if report.get("adapter_sha256") != manifests[role].get("adapter_sha256"):
                raise LineageError(f"round {round_index} {role} reload adapter hash mismatch")
            reload_reports[role] = {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            }
        round_report_path = round_dir / "round_report.json"
        round_report = read_json(round_report_path)
        if expected_host and round_report.get("execution_host") != expected_host:
            raise LineageError(
                f"round {round_index} report did not originate on {expected_host}"
            )
        rounds.append(
            {
                "round": round_index,
                "dca_manifest": str(manifest_paths["dca"]),
                "dca_manifest_sha256": sha256_file(manifest_paths["dca"]),
                "vda_manifest": str(manifest_paths["vda"]),
                "vda_manifest_sha256": sha256_file(manifest_paths["vda"]),
                "lineage": lineage,
                "reload_reports": reload_reports,
                "round_report_sha256": sha256_file(round_report_path),
            }
        )

    result = {
        "schema_version": 1,
        "kind": "dca_first_lineage_audit",
        "status": "passed",
        "audited_at": utc_now(),
        "backbone": backbone,
        "artifact_scope": artifact_scope,
        "max_round": max_round,
        "execution_host_required": expected_host,
        "dca_chain": [manifest["adapter_sha256"] for manifest in role_manifests["dca"]],
        "vda_chain": [manifest["adapter_sha256"] for manifest in role_manifests["vda"]],
        "rounds": rounds,
    }
    if artifact_scope == "formal" and max_round == 3:
        heldout_manifest_path = root / "data" / "final_heldout" / backbone / "manifest.json"
        heldout = read_json(heldout_manifest_path)
        if heldout.get("status") != "sealed" or int(heldout.get("selected_count", -1)) != 800:
            raise LineageError("formal final-heldout is not sealed at 800 scenarios")
        result["final_heldout_manifest"] = str(heldout_manifest_path)
        result["final_heldout_manifest_sha256"] = sha256_file(heldout_manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--backbone", choices=["qwen3.5-4b", "qwen3.5-9b"], required=True)
    parser.add_argument("--artifact-scope", choices=["formal", "pilot"], required=True)
    parser.add_argument("--max-round", type=int, required=True)
    parser.add_argument(
        "--expected-host",
        default=os.environ.get("AGZ_EXPECTED_HOST", ""),
        help="optionally require every round to originate on this hostname",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.max_round <= 0 or args.max_round > 3:
        raise SystemExit("--max-round must be between 1 and 3")
    root = Path(args.root).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else root
        / "data"
        / ("co_evolution" if args.artifact_scope == "formal" else "co_evolution_pilot")
        / args.backbone
        / f"lineage_audit_round_{args.max_round}.json"
    )
    result = audit_lineage(
        root,
        args.backbone,
        args.artifact_scope,
        args.max_round,
        expected_host=args.expected_host or None,
    )
    atomic_write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
