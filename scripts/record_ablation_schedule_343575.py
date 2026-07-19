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
        path: sha256_source_tree(ROOT / path) for path in previous_source_trees
    }
    framework = {
        path: sha256_file(ROOT / path) for path in previous_framework
    }
    revision = {
        "created_at": utc_now(),
        "superseded_job": "343575",
        "reason": (
            "Apply the GPU-validated 32-trajectory/75-step VDA schedule to the "
            "4B append-only-memory ablation before resubmission."
        ),
        "changed_files": [
            "scripts/jobs/tmcd_v2_4b_append_only_optimized_node208.dsub.sh",
            "scripts/record_ablation_schedule_343575.py",
        ],
        "behavioral_scope": (
            "ablation execution schedule only; data volume, variant semantics, "
            "rewards, and three-round co-evolution remain unchanged"
        ),
        "previous_source_trees": previous_source_trees,
        "source_trees": source_trees,
        "previous_training_framework": previous_framework,
        "training_framework": framework,
        "validation": {
            "bash_syntax": "passed",
            "reference_gpu_gate_job": "343578",
            "reference_gpu_gate_status": "SUCCEEDED",
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
