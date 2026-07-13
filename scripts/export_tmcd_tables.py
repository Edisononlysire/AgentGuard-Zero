#!/usr/bin/env python3
"""Export paper-ready TMCD tables from AgentGuard-Zero eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


MAIN_ORDER = [
    "Rule-based SOC",
    "ReAct / Base+Tools",
    "Memory Agent",
    "Trust-score Agent",
    "Cyber LLM VDA",
    "Qwen Zero-shot VDA",
    "AgentGuard-Zero-Select",
    "AgentGuard-Zero-Train",
    "Oracle Defender",
]

TASK_ORDER = [
    "T1 Active Probing Defense",
    "T2 Trust-Building Betrayal",
    "T3 Profile / Memory Poisoning",
    "T4 Business-Constrained Overreaction",
]

CAGE_ORDER = [
    "Rule-based SOC",
    "ReAct / Base+Tools",
    "Memory Agent",
    "Trust-score Agent",
    "Cyber LLM VDA",
    "AgentGuard-Zero-Select",
    "AgentGuard-Zero-Train",
]

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.3f}"


def collect_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = load_json(summary_path)
        results_path = run_dir / "results.jsonl"
        results = read_jsonl(results_path) if results_path.exists() else []
        runs.append({"dir": run_dir, "summary": summary, "results": results})
    return runs


def is_cage_run(run: dict[str, Any]) -> bool:
    for item in run.get("results", []):
        if str(item.get("task", "")).startswith("CAGE-"):
            return True
    task_summary = run.get("summary", {}).get("task_safe_utility", {})
    if isinstance(task_summary, dict):
        return any(str(key).startswith("CAGE-") for key in task_summary)
    return False


def by_display_latest(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for run in runs:
        display = str(run["summary"].get("system_display") or run["summary"].get("system") or run["dir"].name)
        previous = latest.get(display)
        if previous is not None:
            prev_mtime = (previous["dir"] / "summary.json").stat().st_mtime
            this_mtime = (run["dir"] / "summary.json").stat().st_mtime
            if this_mtime <= prev_mtime:
                continue
        latest[display] = run
    return latest


def normalization_bounds(runs: list[dict[str, Any]]) -> tuple[float, float]:
    random_values = [
        float(run["summary"].get("safe_utility", 0.0))
        for run in runs
        if str(run["summary"].get("system")) == "random_policy"
    ]
    oracle_values = [
        float(run["summary"].get("safe_utility", 0.0))
        for run in runs
        if str(run["summary"].get("system")) == "oracle_defender"
    ]
    lower = sum(random_values) / len(random_values) if random_values else -0.45
    upper = sum(oracle_values) / len(oracle_values) if oracle_values else 0.55
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def nsu(value: Any, lower: float, upper: float) -> float:
    try:
        raw = float(value)
    except Exception:
        raw = lower
    return max(0.0, min(1.0, (raw - lower) / (upper - lower)))


def write_table1(runs: list[dict[str, Any]], output: Path) -> None:
    latest = by_display_latest(runs)
    lower, upper = normalization_bounds(runs)
    rows = []
    for name in MAIN_ORDER:
        run = latest.get(name)
        if not run:
            continue
        s = run["summary"]
        rows.append(
            {
                "System": name,
                "NSU ↑": fmt(nsu(s.get("safe_utility"), lower, upper)),
                "Attack Mitigation ↑": fmt(s.get("attack_mitigation")),
                "Betrayal Detection ↑": fmt(s.get("betrayal_detection")),
                "Poison Success ↓": fmt(s.get("poison_success")),
                "Overresponse ↓": fmt(s.get("overresponse_rate")),
                "Business Cost ↓": fmt(s.get("business_cost")),
            }
        )
    write_markdown_table(output / "table1_overall_results.md", rows)
    write_csv(output / "table1_overall_results.csv", rows)


def task_safe_utility(run: dict[str, Any], lower: float, upper: float) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for item in run["results"]:
        task = str(item.get("task", "unknown"))
        value = item.get("tmcd_metrics", {}).get("safe_utility")
        try:
            buckets.setdefault(task, []).append(float(value))
        except Exception:
            pass
    return {
        task: nsu(sum(values) / len(values), lower, upper)
        for task, values in buckets.items()
        if values
    }


def write_heatmap(runs: list[dict[str, Any]], output: Path) -> None:
    latest = by_display_latest(runs)
    lower, upper = normalization_bounds(runs)
    rows = []
    for name in MAIN_ORDER:
        if name in {"Rule-based SOC", "Oracle Defender"}:
            continue
        run = latest.get(name)
        if not run:
            continue
        by_task = task_safe_utility(run, lower, upper)
        row = {"System": name}
        for task in TASK_ORDER:
            row[task] = fmt(by_task.get(task))
        rows.append(row)
    write_csv(output / "figure1_task_heatmap.csv", rows)
    write_markdown_table(output / "figure1_task_heatmap.md", rows)


def write_cage_table(runs: list[dict[str, Any]], output: Path) -> None:
    latest = by_display_latest(runs)
    lower, upper = normalization_bounds(runs)
    rows = []
    for name in CAGE_ORDER:
        run = latest.get(name)
        if not run:
            continue
        t3_poison = []
        t3_su = []
        t4_over = []
        t4_cost = []
        all_su = []
        for item in run["results"]:
            task = str(item.get("task", ""))
            metrics = item.get("tmcd_metrics", {})
            if task.startswith("CAGE-T3"):
                t3_poison.append(float(metrics.get("poison_success", 0.0)))
                t3_su.append(float(metrics.get("safe_utility", 0.0)))
            if task.startswith("CAGE-T4"):
                t4_over.append(float(metrics.get("overresponse", 0.0)))
                t4_cost.append(float(metrics.get("business_cost", 0.0)))
            if task.startswith("CAGE-"):
                all_su.append(float(metrics.get("safe_utility", 0.0)))
        if not all_su:
            continue
        rows.append(
            {
                "System": name,
                "CAGE-T3 Poison Success ↓": fmt(sum(t3_poison) / len(t3_poison) if t3_poison else None),
                "CAGE-T3 NSU ↑": fmt(nsu(sum(t3_su) / len(t3_su), lower, upper) if t3_su else None),
                "CAGE-T4 Overresponse ↓": fmt(sum(t4_over) / len(t4_over) if t4_over else None),
                "CAGE-T4 Business Cost ↓": fmt(sum(t4_cost) / len(t4_cost) if t4_cost else None),
                "Avg. NSU ↑": fmt(nsu(sum(all_su) / len(all_su), lower, upper)),
            }
        )
    write_markdown_table(output / "table3_cage_transfer.md", rows)
    write_csv(output / "table3_cage_transfer.csv", rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("_No rows available yet._\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="outputs/tmcd_eval")
    parser.add_argument("--output_dir", default="outputs/paper_tables")
    args = parser.parse_args()
    runs = collect_runs(Path(args.input_dir))
    main_runs = [run for run in runs if not is_cage_run(run)]
    cage_runs = [run for run in runs if is_cage_run(run)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_table1(main_runs, output)
    write_heatmap(main_runs, output)
    write_cage_table(cage_runs, output)
    print(json.dumps({"runs": len(runs), "main_runs": len(main_runs), "cage_runs": len(cage_runs), "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
