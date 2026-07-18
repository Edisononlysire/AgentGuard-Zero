#!/usr/bin/env python3
"""Generate deterministic public-equivalent canonical recovery worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.recovery.canonical_scenarios import canonical_recovery_suite
from agentguard_zero.recovery.protocol import RECOVERY_PROTOCOL_VERSION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-count", type=int, required=True)
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    groups = canonical_recovery_suite(
        scenario_count=args.scenario_count,
        group_offset=args.group_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": RECOVERY_PROTOCOL_VERSION,
        "scenario_count": sum(len(group) for group in groups),
        "public_world_group_count": len(groups),
        "human_action_labels": 0,
        "groups": groups,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "sealed",
        "protocol_version": RECOVERY_PROTOCOL_VERSION,
        "scenario_count": payload["scenario_count"],
        "public_world_group_count": payload["public_world_group_count"],
        "sha256": _sha256(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
