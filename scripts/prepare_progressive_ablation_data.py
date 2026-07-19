#!/usr/bin/env python3
"""Re-encode the sealed formal VDA train/dev scenarios for one ablation variant.

No scenario is generated, selected, removed, reordered, or duplicated here.
Only the variant-specific prompt and runtime metadata are rebuilt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import (
    LineageError,
    atomic_write_json,
    canonical_json,
    read_json,
    scenario_fingerprint,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from agentguard_zero.training.vda_dataset import scenario_to_training_row
from agentguard_zero.variants import PROGRESSIVE_VARIANTS


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rebuild_split(
    source_path: Path,
    output_path: Path,
    *,
    variant: str,
    split: str,
    round_index: int,
) -> dict[str, Any]:
    source = pd.read_parquet(source_path)
    rebuilt_rows = []
    source_fingerprints = []
    rebuilt_fingerprints = []
    task_counts: Counter[str] = Counter()
    for row_index, source_row in enumerate(source.to_dict(orient="records")):
        scenario = _as_dict(source_row.get("scenario"))
        if not scenario:
            raise LineageError(f"missing scenario in {source_path} row {row_index}")
        source_extra = _as_dict(source_row.get("extra_info"))
        fingerprint = scenario_fingerprint(scenario)
        declared = str(source_extra.get("scenario_fingerprint", fingerprint))
        if declared != fingerprint:
            raise LineageError(
                f"source scenario fingerprint mismatch in {source_path} row {row_index}"
            )
        source_fingerprints.append(fingerprint)

        variant_row = scenario_to_training_row(
            scenario,
            split=split,
            experiment_variant=variant,
        )
        rebuilt = dict(source_row)
        rebuilt["problem"] = variant_row["problem"]
        extra = dict(source_extra)
        task_id = str(
            source_row.get("task_id")
            or source_extra.get("task_id")
            or _as_dict(variant_row.get("extra_info")).get("task_id")
        )
        extra.update(
            {
                "scenario_fingerprint": fingerprint,
                "source_formal_parquet": str(source_path.resolve()),
                "source_formal_parquet_sha256": sha256_file(source_path),
                "source_formal_row_index": row_index,
                "source_dca_round": round_index,
                "experiment_variant": variant,
                "progressive_ablation": True,
            }
        )
        rebuilt["extra_info"] = extra
        rebuilt["scenario_fingerprint"] = fingerprint
        rebuilt["task_id"] = task_id
        rebuilt_rows.append(rebuilt)
        rebuilt_scenario = _as_dict(rebuilt["scenario"])
        if canonical_json(rebuilt_scenario) != canonical_json(scenario):
            raise LineageError("progressive re-encoding changed formal scenario content")
        rebuilt_fingerprints.append(scenario_fingerprint(rebuilt_scenario))
        task_counts[task_id] += 1

    if source_fingerprints != rebuilt_fingerprints:
        raise LineageError(f"rebuilt {split} scenario order or fingerprints changed")
    if len(set(source_fingerprints)) != len(source_fingerprints):
        raise LineageError(f"formal {split} contains duplicate scenario fingerprints")
    _atomic_parquet(pd.DataFrame(rebuilt_rows), output_path)
    return {
        "split": split,
        "source": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "rows": len(rebuilt_rows),
        "task_counts": dict(sorted(task_counts.items())),
        "ordered_scenario_fingerprints_sha256": sha256_bytes(
            canonical_json(source_fingerprints).encode("utf-8")
        ),
        "scenario_content_and_order_identical": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--formal-scope", default="tmcd_v242")
    parser.add_argument("--backbone", choices=["qwen3.5-4b", "qwen3.5-9b"], required=True)
    parser.add_argument("--variant", choices=PROGRESSIVE_VARIANTS, required=True)
    parser.add_argument("--round", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_round = root / "data" / args.formal_scope / args.backbone / f"round_{args.round}"
    source_pool_manifest = source_round / "vda_pool_manifest.json"
    source_train = source_round / "vda_train" / "train.parquet"
    source_dev = source_round / "vda_dev" / "dev.parquet"
    for path in (source_pool_manifest, source_train, source_dev):
        if not path.is_file():
            raise SystemExit(f"missing sealed formal VDA artifact: {path}")
    formal_manifest = read_json(source_pool_manifest)
    if formal_manifest.get("kind") != "vda_pool":
        raise LineageError(f"formal source manifest is not a VDA pool: {source_pool_manifest}")
    for split, path in (("train", source_train), ("dev", source_dev)):
        if (formal_manifest.get("sha256", {}) or {}).get(split) != sha256_file(path):
            raise LineageError(f"formal {split} parquet no longer matches its sealed pool manifest")

    manifest_path = output_dir / "manifest.json"
    train_path = output_dir / "vda_train" / "train.parquet"
    dev_path = output_dir / "vda_dev" / "dev.parquet"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if (
            existing.get("status") == "sealed"
            and existing.get("variant") == args.variant
            and existing.get("round") == args.round
            and existing.get("source_pool_manifest_sha256") == sha256_file(source_pool_manifest)
            and train_path.exists()
            and dev_path.exists()
            and existing.get("splits", {}).get("train", {}).get("output_sha256")
            == sha256_file(train_path)
            and existing.get("splits", {}).get("dev", {}).get("output_sha256")
            == sha256_file(dev_path)
        ):
            print(json.dumps(existing, ensure_ascii=False, indent=2), flush=True)
            return
        raise LineageError(f"incompatible existing progressive data: {output_dir}")

    train = rebuild_split(
        source_train,
        train_path,
        variant=args.variant,
        split="train",
        round_index=args.round,
    )
    dev = rebuild_split(
        source_dev,
        dev_path,
        variant=args.variant,
        split="dev",
        round_index=args.round,
    )
    if train["rows"] != 2400 or dev["rows"] != 400:
        raise LineageError(f"unexpected formal split sizes: train={train['rows']} dev={dev['rows']}")
    expected_task_counts = {"T1": 600, "T2": 600, "T3": 600, "T4": 600}
    if train["task_counts"] != expected_task_counts:
        raise LineageError(f"formal train task quotas changed: {train['task_counts']}")

    manifest = {
        "schema_version": 1,
        "kind": "progressive_ablation_round_data",
        "status": "sealed",
        "created_at": utc_now(),
        "formal_scope": args.formal_scope,
        "backbone": args.backbone,
        "variant": args.variant,
        "round": args.round,
        "data_policy": "exact_formal_filtered_vda_scenarios_variant_reencoded",
        "dca_generation": False,
        "frontier_filtering": False,
        "source_pool_manifest": str(source_pool_manifest.resolve()),
        "source_pool_manifest_sha256": sha256_file(source_pool_manifest),
        "splits": {"train": train, "dev": dev},
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
