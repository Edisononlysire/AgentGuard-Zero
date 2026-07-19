#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    sha256_source_tree,
    utc_now,
)


def main() -> None:
    target = ROOT / "data" / "tmcd_v2" / "manifests" / "source_freeze.json"
    manifest = json.loads(target.read_text(encoding="utf-8"))
    previous_source_trees = dict(manifest.get("source_trees", {}))
    previous_framework = dict(manifest.get("training_framework", {}))

    source_trees = {
        relative_path: sha256_source_tree(ROOT / relative_path)
        for relative_path in previous_source_trees
    }
    framework = {
        relative_path: sha256_file(ROOT / relative_path)
        for relative_path in previous_framework
    }
    revision = {
        "created_at": utc_now(),
        "incident_job": "343569",
        "reason": (
            "Restore the stable 32-trajectory/75-step VDA schedule after the "
            "160-trajectory schedule exposed an ignored HF rollout sequence limit "
            "and exhausted A100 memory late in multi-turn generation."
        ),
        "changed_files": [
            "third_party/verl/verl/workers/rollout/hf_rollout.py",
            "scripts/train_vda_qwen35_lora.sh",
            "scripts/jobs/tmcd_v2_4b_full_optimized_node175.dsub.sh",
            "scripts/jobs/tmcd_v2_4b_vda_oom_gate_node175.dsub.sh",
            "scripts/record_hotfix_343569.py",
        ],
        "behavioral_scope": (
            "HF rollout micro-batch scheduling and 4B VDA execution schedule only; "
            "training data, trajectories, rewards, schemas, and protocol remain unchanged"
        ),
        "previous_source_trees": previous_source_trees,
        "source_trees": source_trees,
        "previous_training_framework": previous_framework,
        "training_framework": framework,
        "validation": {
            "chunk_planner_gate": "passed",
            "full_unittest_discovery": "59/59 passed",
            "gpu_gate_job": "343578",
            "gpu_gate_schedule": "global batch 32; 8 trajectories per GPU",
            "gpu_gate_status": "SUCCEEDED",
        },
    }
    manifest.setdefault("deployment_revisions", []).append(revision)
    manifest["source_trees"] = source_trees
    manifest["source_tree_sha256"] = source_trees["agentguard_zero"]
    manifest["scripts_tree_sha256"] = source_trees["scripts"]
    manifest["training_framework"] = framework
    manifest["updated_at"] = utc_now()
    manifest["git_dirty"] = True
    atomic_write_json(target, manifest)
    print(json.dumps(revision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
