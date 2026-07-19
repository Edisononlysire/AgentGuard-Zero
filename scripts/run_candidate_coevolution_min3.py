#!/usr/bin/env python3
"""Run a resumable three-round capability-gated candidate co-evolution pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, utc_now


def run(
    command: list[str],
    *,
    check: bool = True,
    output: Path | None = None,
) -> int:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    if output is None:
        result = subprocess.run(command, check=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command, check=False, stdout=handle, stderr=subprocess.STDOUT, text=True
            )
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return int(result.returncode)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_policy(
    *,
    python: str,
    model_path: Path,
    manifest: Path,
    output_dir: Path,
    group_offset: int,
    scenario_count: int,
) -> Path:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        return metrics_path
    output_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[str], object]] = []
    shards: list[Path] = []
    for index in range(4):
        shard = output_dir / f"shard_{index}.json"
        shards.append(shard)
        log_handle = (output_dir / f"shard_{index}.log").open("w", encoding="utf-8")
        command = [
            python,
            "-s",
            str(ROOT / "scripts/eval_candidate_policy.py"),
            "run",
            "--model-path",
            str(model_path),
            "--ranker-manifest",
            str(manifest),
            "--output",
            str(shard),
            "--scenario-count",
            str(scenario_count),
            "--group-offset",
            str(group_offset),
            "--shard-index",
            str(index),
            "--shard-count",
            "4",
            "--device",
            f"cuda:{index}",
            "--max-length",
            "1024",
            "--score-batch-size",
            "4",
        ]
        processes.append(
            (
                subprocess.Popen(
                    command, stdout=log_handle, stderr=subprocess.STDOUT, text=True
                ),
                log_handle,
            )
        )
    failures = []
    for process, handle in processes:
        code = process.wait()
        handle.close()
        if code:
            failures.append(code)
    if failures:
        raise RuntimeError(f"candidate policy evaluation shards failed: {failures}")
    run(
        [
            python,
            "-s",
            str(ROOT / "scripts/eval_candidate_policy.py"),
            "merge",
            "--inputs",
            *map(str, shards),
            "--output",
            str(metrics_path),
        ]
    )
    return metrics_path


def evaluate_offline(
    *,
    python: str,
    model_path: Path,
    manifest: Path,
    candidate_sets: Path,
    output: Path,
    device: str,
) -> Path:
    if output.exists():
        return output
    run(
        [
            python,
            "-s",
            str(ROOT / "scripts/eval_candidate_ranker_offline.py"),
            "--model-path",
            str(model_path),
            "--ranker-manifest",
            str(manifest),
            "--candidate-sets",
            str(candidate_sets),
            "--output",
            str(output),
            "--device",
            device,
            "--max-length",
            "1024",
            "--score-batch-size",
            "4",
        ]
    )
    return output


def require_or_override(
    path: Path, *, diagnostic_override: bool, label: str
) -> bool:
    accepted = bool(read(path).get("accepted", False))
    if not accepted and not diagnostic_override:
        raise RuntimeError(f"{label} rejected; use --diagnostic-override-gates only for a pilot")
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--warmstart-manifest", type=Path, required=True)
    parser.add_argument("--canonical-replay", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--records-per-round", type=int, default=16)
    parser.add_argument("--fixed-scenario-count", type=int, default=8)
    parser.add_argument("--fixed-group-offset", type=int, default=60000)
    parser.add_argument("--fresh-group-offset", type=int, default=70000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnostic-override-gates", action="store_true")
    args = parser.parse_args()
    if args.rounds != 3:
        raise ValueError("this minimum protocol is frozen to exactly three rounds")
    if args.fixed_scenario_count <= 0 or args.fixed_scenario_count % 8:
        raise ValueError("fixed scenario count must be a positive multiple of eight")
    for path in (
        args.model_path,
        args.warmstart_manifest,
        args.canonical_replay,
        args.canonical_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    canonical_manifest = read(args.canonical_manifest)
    if canonical_manifest.get("accepted") is not True:
        raise RuntimeError("canonical replay manifest is not accepted")
    if sha256_file(args.canonical_replay) != canonical_manifest.get(
        "candidate_sets_sha256"
    ):
        raise RuntimeError("canonical replay hash mismatch")

    data_root = ROOT / "data/candidate_coevolution_min3/pilot3"
    output_root = ROOT / "outputs/candidate_min3/pilot3"
    evaluation_root = ROOT / "evaluations/candidate_coevolution_min3/pilot3"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    active_manifest = args.warmstart_manifest.resolve()
    active_evaluation = evaluate_policy(
        python=args.python,
        model_path=args.model_path,
        manifest=active_manifest,
        output_dir=evaluation_root / "round_0/fixed_active",
        group_offset=args.fixed_group_offset,
        scenario_count=args.fixed_scenario_count,
    )
    gate_a = evaluation_root / "round_0/gate_a.json"
    if not gate_a.exists():
        run(
            [
                args.python,
                "-s",
                str(ROOT / "scripts/eval_candidate_gates.py"),
                "gate-a",
                "--evaluation",
                str(active_evaluation),
                "--output",
                str(gate_a),
            ],
            check=False,
        )
    gate_a_accepted = require_or_override(
        gate_a,
        diagnostic_override=args.diagnostic_override_gates,
        label="Gate A",
    )

    previous_fresh: list[Path] = []
    previous_fresh_start_evaluations: list[Path] = []
    rounds: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        round_data = data_root / f"round_{round_index}"
        round_eval = evaluation_root / f"round_{round_index}"
        round_output = output_root / f"ranker_round_{round_index}"
        round_eval.mkdir(parents=True, exist_ok=True)
        curriculum = round_eval / "dca_curriculum.json"
        if not curriculum.exists():
            run(
                [
                    args.python,
                    "-s",
                    str(ROOT / "scripts/update_candidate_curriculum.py"),
                    "--evaluation",
                    str(active_evaluation),
                    "--output",
                    str(curriculum),
                    "--round-index",
                    str(round_index),
                    "--seed",
                    str(args.seed),
                    "--groups",
                    "8",
                ],
                check=False,
            )
        require_or_override(
            curriculum,
            diagnostic_override=args.diagnostic_override_gates,
            label=f"round {round_index} source capability gate",
        )
        curriculum_payload = read(curriculum)
        fresh_dir = round_data / "fresh"
        if not (fresh_dir / "manifest.json").exists():
            run(
                [
                    args.python,
                    "-s",
                    str(ROOT / "scripts/build_candidate_dataset.py"),
                    "--output-dir",
                    str(fresh_dir),
                    "--task-schedule",
                    ",".join(curriculum_payload["task_schedule"]),
                    "--group-offset",
                    str(args.fresh_group_offset + round_index * 100),
                    "--max-records",
                    str(args.records_per_round),
                    "--max-records-per-group",
                    "2",
                    "--trajectory-policy",
                    "teacher",
                    "--data-source",
                    f"dca_frontier_round_{round_index}",
                    "--permutation-seed",
                    str(args.seed + round_index * 1000),
                ]
            )
        audit_path = round_eval / "candidate_audit.json"
        if not audit_path.exists():
            run(
                [
                    args.python,
                    "-s",
                    str(ROOT / "scripts/audit_candidate_dataset.py"),
                    "--candidate-sets",
                    str(fresh_dir / "candidate_sets.jsonl"),
                    "--output",
                    str(audit_path),
                ]
            )
        fresh_start = evaluate_offline(
            python=args.python,
            model_path=args.model_path,
            manifest=active_manifest,
            candidate_sets=fresh_dir / "candidate_sets.jsonl",
            output=round_eval / "fresh_start.json",
            device=args.device,
        )
        feedback_gate = round_eval / "dca_feedback_gate.json"
        if not feedback_gate.exists():
            run(
                [
                    args.python,
                    "-s",
                    str(ROOT / "scripts/summarize_candidate_curriculum_feedback.py"),
                    "--offline-evaluation",
                    str(fresh_start),
                    "--curriculum",
                    str(curriculum),
                    "--output",
                    str(feedback_gate),
                ],
                check=False,
            )
        feedback_accepted = require_or_override(
            feedback_gate,
            diagnostic_override=args.diagnostic_override_gates,
            label=f"round {round_index} DCA feedback gate",
        )

        mix_dir = round_data / "mix"
        if not (mix_dir / "manifest.json").exists():
            mix_command = [
                args.python,
                "-s",
                str(ROOT / "scripts/mix_candidate_replay.py"),
                "--fresh",
                str(fresh_dir / "candidate_sets.jsonl"),
                "--canonical",
                str(args.canonical_replay),
                "--probe-chain",
                str(args.canonical_replay),
                "--output-dir",
                str(mix_dir),
                "--total-records",
                str(args.records_per_round),
                "--seed",
                str(args.seed + round_index),
            ]
            if previous_fresh:
                mix_command.extend(["--past", *map(str, previous_fresh)])
                mix_command.extend(["--dagger", str(previous_fresh[-1])])
            run(mix_command)

        if not (round_output / "manifest.json").exists():
            source = read(active_manifest)
            log_dir = output_root / "launcher_logs" / f"round_{round_index}"
            run(
                [
                    args.python,
                    "-s",
                    str(ROOT / "scripts/launch_candidate_ddp.py"),
                    "--nproc",
                    "4",
                    "--log-dir",
                    str(log_dir),
                    "--",
                    args.python,
                    "-s",
                    str(ROOT / "scripts/train_candidate_ranker.py"),
                    "--model-path",
                    str(args.model_path),
                    "--train-jsonl",
                    str(mix_dir / "candidate_sets.jsonl"),
                    "--data-manifest",
                    str(mix_dir / "manifest.json"),
                    "--output-dir",
                    str(round_output),
                    "--init-adapter",
                    str(source["adapter_path"]),
                    "--init-heads",
                    str(source.get("heads_path", source["score_head_path"])),
                    "--objective",
                    "joint",
                    "--learning-rate",
                    "3e-6",
                    "--epochs",
                    "1",
                    "--seed",
                    str(args.seed + round_index),
                    "--max-length",
                    "1024",
                    "--per-device-batch",
                    "1",
                    "--gradient-accumulation",
                    "1",
                    "--warmup-ratio",
                    "0",
                ]
            )
        target_manifest = round_output / "manifest.json"
        training = read(target_manifest)
        training_summary = {
            "schema_version": 1,
            "kind": "candidate_round_training_log_summary",
            "created_at": utc_now(),
            "round_index": round_index,
            "objective": training["objective"],
            "learning_rate": training["learning_rate"],
            "train_metrics": training["train_metrics"],
            "adapter_sha256": training["adapter_sha256"],
            "heads_sha256": training["heads_sha256"],
            "source_data_manifest_sha256": training["source_data_manifest_sha256"],
        }
        training_summary_path = round_eval / "training_summary.json"
        if not training_summary_path.exists():
            atomic_write_json(training_summary_path, training_summary)

        fixed_end = evaluate_policy(
            python=args.python,
            model_path=args.model_path,
            manifest=target_manifest,
            output_dir=round_eval / "fixed_end",
            group_offset=args.fixed_group_offset,
            scenario_count=args.fixed_scenario_count,
        )
        fresh_end = evaluate_offline(
            python=args.python,
            model_path=args.model_path,
            manifest=target_manifest,
            candidate_sets=fresh_dir / "candidate_sets.jsonl",
            output=round_eval / "fresh_end.json",
            device=args.device,
        )
        retention = None
        if previous_fresh:
            retention = evaluate_offline(
                python=args.python,
                model_path=args.model_path,
                manifest=target_manifest,
                candidate_sets=previous_fresh[-1],
                output=round_eval / "previous_round_retention.json",
                device=args.device,
            )
        round_gate = round_eval / "round_gate.json"
        if not round_gate.exists():
            command = [
                args.python,
                "-s",
                str(ROOT / "scripts/eval_candidate_min3_round.py"),
                "--fixed-start",
                str(active_evaluation),
                "--fixed-end",
                str(fixed_end),
                "--fresh-start",
                str(fresh_start),
                "--fresh-end",
                str(fresh_end),
                "--round-index",
                str(round_index),
                "--output",
                str(round_gate),
            ]
            if retention is not None:
                command.extend(["--retention", str(retention)])
                command.extend(
                    ["--retention-start", str(previous_fresh_start_evaluations[-1])]
                )
            run(command, check=False)
        gate = read(round_gate)
        accepted = bool(
            gate_a_accepted and gate["accepted"] and feedback_accepted
        )
        source_before = str(active_manifest)
        if accepted:
            active_manifest = target_manifest.resolve()
            active_evaluation = fixed_end
        previous_fresh.append((fresh_dir / "candidate_sets.jsonl").resolve())
        previous_fresh_start_evaluations.append(fresh_start.resolve())
        round_summary = {
            "round_index": round_index,
            "accepted": accepted,
            "rollback_required": not accepted,
            "source_manifest": source_before,
            "candidate_manifest": str(target_manifest.resolve()),
            "active_manifest_after_gate": str(active_manifest),
            "curriculum": str(curriculum),
            "dca_feedback_gate": str(feedback_gate),
            "round_gate": str(round_gate),
            "training_summary": str(training_summary_path),
        }
        atomic_write_json(round_eval / "summary.json", round_summary)
        rounds.append(round_summary)

    summary = {
        "schema_version": 1,
        "kind": "candidate_coevolution_min3_pilot",
        "created_at": utc_now(),
        "round_count": len(rounds),
        "completed": len(rounds) == 3,
        "diagnostic_override_gates": args.diagnostic_override_gates,
        "initial_gate_a_accepted": gate_a_accepted,
        "rounds_accepted": sum(row["accepted"] for row in rounds),
        "rounds": rounds,
        "final_active_manifest": str(active_manifest),
        "warmstart_manifest_sha256": sha256_file(args.warmstart_manifest),
    }
    atomic_write_json(evaluation_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
