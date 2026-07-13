#!/usr/bin/env python3
"""Freeze TMCD-v2 protocol manifests and source identity before GPU submission."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.training.coevolution import atomic_write_json, sha256_file, sha256_tree, utc_now


MANIFEST_NAMES = (
    "protocol.json",
    "manipulation_families.json",
    "ood_holdout_families.json",
    "schema_versions.json",
)


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--source-branch", default="tmcd-protocol-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    source = root / "configs" / "tmcd_v2"
    destination = root / "data" / "tmcd_v2" / "manifests"
    destination.mkdir(parents=True, exist_ok=True)
    if (root / ".git").exists():
        dirty = _git(["status", "--porcelain"])
        git_commit = _git(["rev-parse", "HEAD"])
        git_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        if dirty and not args.allow_dirty:
            raise SystemExit("refusing to freeze TMCD-v2 from a dirty git worktree")
    else:
        if not args.source_commit:
            raise SystemExit("--source-commit is required when the deployed tree has no .git directory")
        dirty = ""
        git_commit = args.source_commit
        git_branch = args.source_branch
    copied = {}
    for name in MANIFEST_NAMES:
        source_path = source / name
        if not source_path.is_file():
            raise SystemExit(f"missing protocol manifest: {source_path}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != "tmcd-v2":
            raise SystemExit(f"wrong protocol version in {source_path}")
        target = destination / name
        if target.exists() and sha256_file(target) != sha256_file(source_path):
            raise SystemExit(f"frozen manifest differs from source: {target}")
        shutil.copy2(source_path, target)
        copied[name] = {"path": str(target), "sha256": sha256_file(target)}
    source_manifest = {
        "protocol_version": "tmcd-v2",
        "kind": "source_freeze",
        "created_at": utc_now(),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": bool(dirty),
        "source_tree_sha256": sha256_tree(root / "agentguard_zero"),
        "scripts_tree_sha256": sha256_tree(root / "scripts"),
        "manifests": copied,
    }
    target = destination / "source_freeze.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("created_at", None)
        current = dict(source_manifest)
        current.pop("created_at", None)
        if comparable != current:
            raise SystemExit(f"existing source freeze differs: {target}")
    else:
        atomic_write_json(target, source_manifest)
    print(json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
