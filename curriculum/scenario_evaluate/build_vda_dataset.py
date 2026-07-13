from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Iterable, List

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agentguard_zero.training.vda_dataset import scenario_to_training_row


def _read_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = [data]
    scenarios = []
    for item in data:
        scenario = item.get("scenario", item)
        if isinstance(scenario, str):
            scenario = json.loads(scenario)
        if scenario:
            scenarios.append(scenario)
    return scenarios


def _read_parquet_file(path: str) -> List[Dict[str, Any]]:
    frame = pd.read_parquet(path)
    scenarios = []
    for _, row in frame.iterrows():
        scenario = row.get("scenario")
        if isinstance(scenario, str):
            scenario = json.loads(scenario)
        if isinstance(scenario, dict):
            scenarios.append(scenario)
    return scenarios


def load_scenarios(inputs: Iterable[str]) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []
    for pattern in inputs:
        for path in sorted(glob.glob(pattern)):
            if path.endswith(".parquet"):
                scenarios.extend(_read_parquet_file(path))
            elif path.endswith(".json"):
                scenarios.extend(_read_json_file(path))
            elif path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            item = json.loads(line)
                            scenario = item.get("scenario", item)
                            if isinstance(scenario, str):
                                scenario = json.loads(scenario)
                            scenarios.append(scenario)
            else:
                raise ValueError(f"unsupported input file: {path}")
    return scenarios


def main(args: argparse.Namespace) -> None:
    scenarios = load_scenarios(args.inputs)
    if args.limit:
        scenarios = scenarios[: args.limit]
    rows = [scenario_to_training_row(s, split=args.split) for s in scenarios]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"saved {len(rows)} VDA training rows to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="JSON/JSONL/parquet scenario files or glob patterns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    main(parser.parse_args())
