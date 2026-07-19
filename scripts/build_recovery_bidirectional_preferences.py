#!/usr/bin/env python3
"""Build conservative action/Observe preferences from audited Teacher Q values."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.public_teacher import (
    ActionCandidate,
    compact_wire_json,
    enumerate_public_candidates,
)
from agentguard_zero.schemas.action_schema_v4 import parse_action_json_v4
from agentguard_zero.world.public_projector import assert_public


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def public_observation(prompt: str) -> dict:
    marker = "\nCurrent decision instance:"
    if marker not in prompt:
        raise ValueError("preference prompt is missing its public context")
    context = json.loads(prompt.split(marker, 1)[1])
    assert_public(context)
    return dict(context["observation"])


def packet_candidate_id(packet_text: str) -> str:
    packet, valid, reason = parse_action_json_v4(packet_text)
    if not valid:
        raise ValueError(f"invalid Teacher target: {reason}")
    raw = json.dumps(
        packet,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-core-margin", type=float, default=0.01)
    parser.add_argument("--min-pairs", type=int, default=1000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    parent_path = args.teacher_dir / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("accepted") is not True:
        raise RuntimeError("Teacher dataset is not accepted")
    rows = pd.read_parquet(args.teacher_dir / "bootstrap_sft.parquet")
    audits = load_jsonl(args.teacher_dir / "teacher_selection_audit.jsonl")
    audit_by_id = {str(row.get("record_id", "")): row for row in audits}
    if len(audit_by_id) != len(audits):
        raise RuntimeError("Teacher audit contains duplicate record IDs")

    pairs: list[dict] = []
    directions: Counter[str] = Counter()
    chosen_categories: Counter[str] = Counter()
    rejected_categories: Counter[str] = Counter()
    for row in rows.to_dict(orient="records"):
        record_id = str(row["record_id"])
        audit = audit_by_id.get(record_id)
        if audit is None:
            raise RuntimeError("Teacher row has no matching Q audit")
        candidates = enumerate_public_candidates(
            public_observation(str(row["prompt"])), max_candidates=96
        )
        by_id: dict[str, ActionCandidate] = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        observe = [candidate for candidate in candidates if candidate.category == "observe"]
        if len(observe) != 1:
            raise RuntimeError("public state does not have exactly one Observe")
        observe_candidate = observe[0]
        selected_id = str(audit.get("selected_candidate_id", ""))
        if packet_candidate_id(str(row["target"])) != selected_id:
            raise RuntimeError("Teacher target disagrees with selected candidate ID")
        if selected_id not in by_id or observe_candidate.candidate_id not in by_id:
            raise RuntimeError("audited candidate is absent from public enumeration")
        q_values = {
            str(key): float(value)
            for key, value in (audit.get("core_q_audit", {}) or {}).items()
        }
        if selected_id not in q_values or observe_candidate.candidate_id not in q_values:
            raise RuntimeError("candidate is absent from core-Q audit")
        selected = by_id[selected_id]
        observe_value = q_values[observe_candidate.candidate_id]
        selected_value = q_values[selected_id]
        margin = float(args.minimum_core_margin)

        chosen: ActionCandidate
        rejected: ActionCandidate
        direction: str
        if selected_id != observe_candidate.candidate_id and selected_value > observe_value + margin:
            chosen, rejected = selected, observe_candidate
            direction = "action_over_observe"
        elif selected_id != observe_candidate.candidate_id and observe_value > selected_value + margin:
            chosen, rejected = observe_candidate, selected
            direction = "observe_over_action"
        elif selected_id == observe_candidate.candidate_id:
            inferior = [
                candidate
                for candidate in candidates
                if candidate.candidate_id != observe_candidate.candidate_id
                and candidate.candidate_id in q_values
                and observe_value > q_values[candidate.candidate_id] + margin
            ]
            if not inferior:
                continue
            # The closest inferior action is the most informative false-positive
            # boundary, with candidate ID as the deterministic tie breaker.
            rejected = sorted(
                inferior,
                key=lambda candidate: (
                    -q_values[candidate.candidate_id],
                    candidate.candidate_id,
                ),
            )[0]
            chosen = observe_candidate
            direction = "observe_over_hard_negative"
        else:
            continue

        chosen_text = compact_wire_json(chosen.packet)
        rejected_text = compact_wire_json(rejected.packet)
        pair_id = hashlib.sha256(
            f"bipref:{record_id}:{chosen.candidate_id}:{rejected.candidate_id}".encode(
                "utf-8"
            )
        ).hexdigest()
        pairs.append(
            {
                "record_id": pair_id,
                "source_record_id": record_id,
                "prompt": str(row["prompt"]),
                "chosen": chosen_text,
                "rejected": rejected_text,
                "chosen_action_category": chosen.category,
                "rejected_action_category": rejected.category,
                "preference_direction": direction,
                "chosen_core_q": q_values[chosen.candidate_id],
                "rejected_core_q": q_values[rejected.candidate_id],
                "core_q_margin": q_values[chosen.candidate_id]
                - q_values[rejected.candidate_id],
            }
        )
        directions[direction] += 1
        chosen_categories[chosen.category] += 1
        rejected_categories[rejected.category] += 1

    unique = len({row["record_id"] for row in pairs}) == len(pairs)
    all_positive = all(
        row["core_q_margin"] > args.minimum_core_margin for row in pairs
    )
    accepted = len(pairs) >= args.min_pairs and unique and all_positive
    manifest = {
        "schema_version": 1,
        "kind": "recovery_bidirectional_core_q_preferences",
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "pair_count": len(pairs),
        "minimum_pair_count": args.min_pairs,
        "minimum_core_q_margin": args.minimum_core_margin,
        "preference_direction_counts": dict(directions),
        "chosen_action_category_counts": dict(chosen_categories),
        "rejected_action_category_counts": dict(rejected_categories),
        "unique_record_ids": unique,
        "all_core_q_margins_strictly_positive": all_positive,
        "source_scenario_count": parent.get("source_scenario_count"),
        "source_scenarios_sha256": parent.get("source_scenarios_sha256"),
        "teacher_manifest_sha256": sha256(parent_path),
        "counterfactual_worlds_training_visible": False,
        "hidden_state_in_prompt_or_response": False,
        "preference_semantics": (
            "prefer audited higher core utility in both action and Observe directions"
        ),
    }
    args.output_dir.mkdir(parents=True)
    parquet = args.output_dir / "preferences.parquet"
    manifest_path = args.output_dir / "manifest.json"
    pd.DataFrame(pairs).to_parquet(parquet, index=False)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256(path) for path in (parquet, manifest_path)}
    (args.output_dir / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
