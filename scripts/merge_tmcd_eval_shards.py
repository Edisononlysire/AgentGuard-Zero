#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import eval_tmcd_systems as tmcd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge resumable TMCD evaluation shards.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    shard_dirs = sorted(path for path in run_dir.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise SystemExit(f"no shard directories found in {run_dir}")

    common_config = None
    results_by_id = {}
    for shard_dir in shard_dirs:
        config = json.loads((shard_dir / "run_config.json").read_text(encoding="utf-8"))
        comparable = {key: value for key, value in config.items() if key != "shard_index"}
        if common_config is None:
            common_config = comparable
        elif comparable != common_config:
            raise RuntimeError(f"evaluation shard config mismatch: {shard_dir}")
        result_path = shard_dir / "results.jsonl"
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                scenario_id = str(item["scenario_id"])
                if scenario_id in results_by_id:
                    raise RuntimeError(f"duplicate scenario across shards: {scenario_id}")
                results_by_id[scenario_id] = item

    results = sorted(results_by_id.values(), key=lambda item: int(item.get("row_index", 0)))
    if args.expected_count and len(results) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} results, got {len(results)}")
    summary_args = SimpleNamespace(**common_config)
    summary = tmcd.summarize(results, summary_args)
    tmcd.write_outputs(results, summary, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "num_results": len(results), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
