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
        "incident_job": "343982",
        "failed_preflight_job": "344013",
        "reason": (
            "Bound the persistent 9B DCA VDA-feedback server generation batch after "
            "a large active-feedback batch shared each A100 with the DCA actor and "
            "caused CUDA OOM."
        ),
        "changed_files": [
            "scripts/vda_feedback_server.py",
            "tests/test_vda_feedback_microbatch.py",
            "scripts/record_hotfix_343982.py",
        ],
        "behavioral_scope": (
            "Opt-in inference micro-batching only; prompt order, feedback records, "
            "formal data volumes, rewards, LoRA schedule, and scientific protocol "
            "are unchanged. The default batch size remains legacy-compatible zero."
        ),
        "previous_source_trees": previous_source_trees,
        "source_trees": source_trees,
        "previous_training_framework": previous_framework,
        "training_framework": framework,
        "validation": {
            "feedback_microbatch_regression": "2/2 passed; exact input order preserved",
            "full_unittest_discovery": "119/119 passed",
            "python_compile": "passed",
            "recovery_generation_batch_size": 16,
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
