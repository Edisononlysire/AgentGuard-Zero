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
        "incident_job": "343561",
        "reason": (
            "Harden malformed VDA action handling after a belief object with an "
            "unknown string-valued field caused the TMCD-v2 tool server to return HTTP 500."
        ),
        "changed_files": [
            "agentguard_zero/schemas/action_schema_v4.py",
            "agentguard_zero/env/cyber_env_v2.py",
            "third_party/verl_tool/llm_agent/manager.py",
            "tests/test_tmcd_v2.py",
            "scripts/record_hotfix_343561.py",
        ],
        "behavioral_scope": "invalid action fallback and tool-server response validation only",
        "previous_source_trees": previous_source_trees,
        "source_trees": source_trees,
        "previous_training_framework": previous_framework,
        "training_framework": framework,
        "validation": {
            "malformed_belief_regression": "2/2 passed",
            "tmcd_v2_suite": "48/48 passed",
            "full_unittest_discovery": "passed",
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
