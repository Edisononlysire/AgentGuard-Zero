#!/usr/bin/env python3
"""Record a fail-closed TMCD-v2 release revision after server validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.protocol import TMCD_PROTOCOL_VERSION, TMCD_RELEASE_REVISION
from agentguard_zero.training.coevolution import (
    atomic_write_json,
    sha256_file,
    sha256_source_tree,
    utc_now,
)


def main() -> None:
    manifests_dir = ROOT / "data" / "tmcd_v2" / "manifests"
    target = manifests_dir / "source_freeze.json"
    protocol_path = manifests_dir / "protocol.json"
    active_protocol_path = ROOT / "configs" / "tmcd_v2" / "protocol.json"
    manifest = json.loads(target.read_text(encoding="utf-8"))
    protocol = json.loads(active_protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != TMCD_PROTOCOL_VERSION:
        raise SystemExit("protocol version mismatch")
    if protocol.get("release_revision") != TMCD_RELEASE_REVISION:
        raise SystemExit("protocol release revision mismatch")
    atomic_write_json(protocol_path, protocol)

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
        "release_revision": TMCD_RELEASE_REVISION,
        "reason": (
            "Close the teacher-reviewed v2.4 protocol with action-semantic response "
            "authorization, runtime-only oracle privilege, exact task-family lineage, "
            "and positive support-only authorization evidence."
        ),
        "changed_files": [
            "agentguard_zero/protocol.py",
            "agentguard_zero/runtime_policy.py",
            "agentguard_zero/defender_state/append_only_memory.py",
            "agentguard_zero/defender_state/evidence_store.py",
            "agentguard_zero/defender_state/memory_fsm.py",
            "agentguard_zero/defender_state/retriever.py",
            "agentguard_zero/world/public_projector.py",
            "agentguard_zero/world/hidden_world.py",
            "agentguard_zero/defender_state/trust_manager.py",
            "agentguard_zero/defender_state/evidence_signals.py",
            "agentguard_zero/env/cyber_env_v2.py",
            "agentguard_zero/env/scenario_instantiator.py",
            "agentguard_zero/env/oracle_v2.py",
            "agentguard_zero/evaluation/rq3_memory.py",
            "agentguard_zero/governance/v5c.py",
            "agentguard_zero/governance/authorization.py",
            "agentguard_zero/schemas/action_schema_v4.py",
            "agentguard_zero/schemas/observation_schema_v4.py",
            "agentguard_zero/schemas/scenario_schema_v2.py",
            "agentguard_zero/training/vda_dataset.py",
            "agentguard_zero/training/coevolution.py",
            "agentguard_zero/training/dca_dataset.py",
            "scripts/generate_dca_scenarios.py",
            "scripts/level1_rollout_server.py",
            "scripts/vda_feedback_server.py",
            "scripts/train_dca_qwen35_lora.sh",
            "scripts/train_vda_qwen35_lora.sh",
            "scripts/env.sh",
            "scripts/jobs/tmcd_v2_4b_full_node175.dsub.sh",
            "scripts/jobs/tmcd_v2_4b_append_only_node208.dsub.sh",
            "scripts/jobs/tmcd_v2_9b_full_node217.dsub.sh",
            "curriculum/reward_function/dca_online_reward.py",
            "scripts/merge_dca_candidate_shards.py",
            "scripts/build_vda_round_pool.py",
            "scripts/eval_tmcd_systems.py",
            "scripts/eval_level1_select.py",
            "scripts/audit_tmcd_v2_release.py",
            "scripts/run_dca_first_round.py",
            "scripts/preflight_tmcd_v2_job.py",
            "scripts/prepare_tmcd_v2_run.py",
            "tests/test_tmcd_v2.py",
            "configs/tmcd_v2/protocol.json",
            "data/tmcd_v2/manifests/protocol.json",
            "README.md",
            "docs/SELECT_V5C.md",
        ],
        "behavioral_scope": (
            "Fail-closed scientific protocol hardening; no new model component and no "
            "change to formal data volume, LoRA training schedule, or rollout backend"
        ),
        "previous_source_trees": previous_source_trees,
        "source_trees": source_trees,
        "previous_training_framework": previous_framework,
        "training_framework": framework,
        "validation": {
            "full_unittest_discovery": "110/110 passed",
            "protocol_smoke": "256 scenarios; T1-T4 64 each; passed",
            "protocol_smoke_digest": "7fb4e0d9fcb24a6ba862e0d19d38b441f749688a8d0f0512e6207521bdaaa870",
            "formal_pool_status": "all pre-v2.4.1 pools invalidated; v2.4.1 pool must be regenerated",
            "legacy_pool_rejection": "enforced by release revision",
        },
    }
    revisions = [
        item
        for item in manifest.get("deployment_revisions", [])
        if item.get("release_revision") != TMCD_RELEASE_REVISION
    ]
    revisions.append(revision)
    manifest["deployment_revisions"] = revisions
    manifest["protocol_version"] = TMCD_PROTOCOL_VERSION
    manifest["release_revision"] = TMCD_RELEASE_REVISION
    manifest["source_trees"] = source_trees
    manifest["source_tree_sha256"] = source_trees["agentguard_zero"]
    manifest["scripts_tree_sha256"] = source_trees["scripts"]
    manifest["training_framework"] = framework
    manifest["manifests"]["protocol.json"]["sha256"] = sha256_file(protocol_path)
    manifest["updated_at"] = utc_now()
    manifest["git_dirty"] = True
    atomic_write_json(target, manifest)
    print(json.dumps(revision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
