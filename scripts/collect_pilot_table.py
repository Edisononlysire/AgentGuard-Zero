#!/usr/bin/env python3
"""Collect AgentGuard-Zero pilot summaries into one table."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

METRIC_COLUMNS = [
    "safe_utility",
    "trajectory_reward",
    "safe_success_rate",
    "attack_mitigation",
    "attack_success",
    "intent_accuracy",
    "business_cost",
    "verification_cost",
    "overresponse_rate",
    "json_parse_failure_rate",
    "raw_candidate_json_parse_failure_rate",
    "selector_fallback_rate",
    "selector_governor_override_rate",
    "invalid_tool_call_rate",
    "invalid_response_action_rate",
    "avg_steps",
]

BASE_COLUMNS = [
    "run_name",
    "method",
    "model",
    "model_backend",
    "policy",
    "num_scenarios",
    "candidate_count",
    "selector_mode",
    "offset",
]


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def format_value(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "" if value is None else str(value)
    return f"{number:.6f}"


def model_label(summary: dict[str, Any]) -> str:
    backend = summary.get("model_backend", "")
    if backend == "api":
        return f"api:{summary.get('api_model', 'unknown')}"
    if backend == "mock":
        return "mock"
    model_path = str(summary.get("model_path", "") or "")
    if model_path:
        return "hf:" + os.path.basename(model_path.rstrip("/"))
    return str(summary.get("model", "") or backend or "unknown")


def method_label(summary: dict[str, Any]) -> str:
    policy = str(summary.get("policy", ""))
    backend = str(summary.get("model_backend", ""))
    if policy == "agentguard_zero_select":
        return "AgentGuard-Zero-Select"
    if policy == "zero_shot_vda":
        return "Zero-shot VDA"
    if policy == "base_tools":
        return "Base+Tools"
    return policy or backend or "unknown"


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    summary = dict(summary)
    summary["source_dir"] = str(path.parent)
    summary["method"] = method_label(summary)
    summary["model"] = model_label(summary)
    return summary


def collect(paths: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in paths:
        if root.is_file() and root.name == "summary.json":
            candidates = [root]
        else:
            candidates = sorted(root.rglob("summary.json")) if root.exists() else []
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                summaries.append(load_summary(path))
            except Exception as exc:
                print(f"skip {path}: {exc}")
    summaries.sort(key=lambda row: (str(row.get("method", "")), str(row.get("model", "")), str(row.get("run_name", ""))))
    return summaries


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = BASE_COLUMNS + METRIC_COLUMNS + ["source_dir"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "method",
        "model",
        "num_scenarios",
        "candidate_count",
        "selector_mode",
        "offset",
        "safe_utility",
        "trajectory_reward",
        "attack_mitigation",
        "safe_success_rate",
        "intent_accuracy",
        "business_cost",
        "overresponse_rate",
        "json_parse_failure_rate",
        "raw_candidate_json_parse_failure_rate",
        "selector_fallback_rate",
        "selector_governor_override_rate",
        "invalid_tool_call_rate",
        "avg_steps",
    ]
    labels = {
        "method": "Method",
        "model": "Model",
        "num_scenarios": "N",
        "candidate_count": "K",
        "selector_mode": "Selector",
        "offset": "Offset",
        "safe_utility": "Safe Utility",
        "trajectory_reward": "Reward",
        "attack_mitigation": "Mitigation",
        "safe_success_rate": "Safe Success",
        "intent_accuracy": "Intent Acc.",
        "business_cost": "Business Cost",
        "overresponse_rate": "Overresponse",
        "json_parse_failure_rate": "JSON Fail",
        "raw_candidate_json_parse_failure_rate": "Raw JSON Fail",
        "selector_fallback_rate": "Fallback",
        "selector_governor_override_rate": "Gov. Override",
        "invalid_tool_call_rate": "Invalid Tool",
        "avg_steps": "Avg Steps",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(labels[col] for col in columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            cells: list[str] = []
            for col in columns:
                value = row.get(col, "")
                cells.append(format_value(value) if col in METRIC_COLUMNS or col in {"num_scenarios", "candidate_count"} else str(value))
            f.write("| " + " | ".join(cells) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            str(ROOT / "outputs" / "eval_api_select"),
            str(ROOT / "outputs" / "eval_select"),
            str(ROOT / "outputs" / "eval_select_smoke"),
            str(ROOT / "outputs" / "eval_api_select_smoke"),
        ],
        help="Directories or summary.json files to collect.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "pilot_tables"))
    parser.add_argument("--prefix", default="agentguard_pilot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(path) for path in args.roots]
    rows = collect(roots)
    output_dir = Path(args.output_dir)
    write_markdown(rows, output_dir / f"{args.prefix}.md")
    write_csv(rows, output_dir / f"{args.prefix}.csv")
    write_json(rows, output_dir / f"{args.prefix}.json")
    print(json.dumps({"rows": len(rows), "output_dir": str(output_dir), "prefix": args.prefix}, ensure_ascii=False))


if __name__ == "__main__":
    main()
