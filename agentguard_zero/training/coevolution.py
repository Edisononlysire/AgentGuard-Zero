from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agentguard_zero.protocol import TMCD_RELEASE_REVISION


SCHEMA_VERSION = 2
PROTOCOL_VERSION = "tmcd-v2"
ROLES = {"dca", "vda"}


class LineageError(RuntimeError):
    """Raised when a co-evolution artifact violates the frozen protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | os.PathLike[str]) -> str:
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    digest = hashlib.sha256()
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(root)).encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def sha256_source_tree(path: str | os.PathLike[str]) -> str:
    """Hash source artifacts while ignoring interpreter and editor by-products."""

    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    digest = hashlib.sha256()
    children = []
    for child in root.rglob("*"):
        if not child.is_file():
            continue
        relative = child.relative_to(root)
        if any(part in {"__pycache__", ".git", ".pytest_cache"} for part in relative.parts):
            continue
        if child.suffix in {".pyc", ".pyo", ".swp", ".tmp"} or child.name == ".DS_Store":
            continue
        children.append(child)
    for child in sorted(children):
        digest.update(str(child.relative_to(root)).encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def scenario_fingerprint(scenario: dict[str, Any]) -> str:
    payload = dict(scenario)
    payload.pop("scenario_id", None)
    # Metadata is provenance, not scenario semantics. Excluding it catches a
    # feedback scenario even when a later generator assigns new lineage fields.
    payload.pop("metadata", None)
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise LineageError(f"{source}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def model_identity(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(model_path).resolve()
    files = []
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        candidate = root / name
        if candidate.exists():
            files.append({"path": name, "sha256": sha256_file(candidate)})
    return {
        "path": str(root),
        "identity_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
        "identity_files": files,
    }


def write_base_manifest(
    path: str | os.PathLike[str],
    *,
    role: str,
    backbone: str,
    model_path: str,
    seed: int,
) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError(f"unsupported role: {role}")
    target = Path(path)
    if target.exists():
        return load_checkpoint_manifest(target, role=role, backbone=backbone, round_index=0)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "kind": "checkpoint",
        "role": role,
        "backbone": backbone,
        "round": 0,
        "created_at": utc_now(),
        "seed": int(seed),
        "base_model": model_identity(model_path),
        "parent_manifest": None,
        "parent_manifest_sha256": None,
        "training_data_manifest": None,
        "training_data_manifest_sha256": None,
        "checkpoint_path": None,
        "adapter_path": None,
        "adapter_sha256": None,
        "status": "base",
    }
    atomic_write_json(target, manifest)
    return manifest


def write_frozen_manifest(
    path: str | os.PathLike[str],
    *,
    role: str,
    backbone: str,
    round_index: int,
    model_path: str,
    seed: int,
    parent_manifest_path: str,
    training_data_manifest_path: str,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance lineage without changing parameters for a training ablation."""

    if round_index <= 0:
        raise ValueError("frozen checkpoints must have round_index > 0")
    parent = load_checkpoint_manifest(
        parent_manifest_path,
        role=role,
        backbone=backbone,
        round_index=round_index - 1,
    )
    data_manifest = Path(training_data_manifest_path).resolve()
    if not data_manifest.is_file():
        raise LineageError(f"training data manifest is missing: {data_manifest}")
    frozen_training_config = dict(training_config or {})
    frozen_training_config["parameter_update"] = False
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "kind": "checkpoint",
        "role": role,
        "backbone": backbone,
        "round": int(round_index),
        "created_at": utc_now(),
        "seed": int(seed),
        "base_model": model_identity(model_path),
        "parent_manifest": str(Path(parent_manifest_path).resolve()),
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "training_data_manifest": str(data_manifest),
        "training_data_manifest_sha256": sha256_file(data_manifest),
        "training_config": frozen_training_config,
        "training_config_sha256": sha256_bytes(
            canonical_json(frozen_training_config).encode("utf-8")
        ),
        "checkpoint_path": parent.get("checkpoint_path"),
        "adapter_path": parent.get("adapter_path"),
        "adapter_sha256": parent.get("adapter_sha256"),
        "status": "frozen",
    }
    atomic_write_json(path, manifest)
    return manifest


def load_checkpoint_manifest(
    path: str | os.PathLike[str],
    *,
    role: str | None = None,
    backbone: str | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    source = Path(path).resolve()
    manifest = read_json(source)
    if manifest.get("kind") != "checkpoint":
        raise LineageError(f"not a checkpoint manifest: {source}")
    actual_role = manifest.get("role")
    if actual_role not in ROLES:
        raise LineageError(f"invalid checkpoint role {actual_role!r}: {source}")
    if role is not None and actual_role != role:
        raise LineageError(f"role mismatch: expected {role}, got {actual_role}: {source}")
    if backbone is not None and manifest.get("backbone") != backbone:
        raise LineageError(
            f"backbone mismatch: expected {backbone}, got {manifest.get('backbone')}: {source}"
        )
    if round_index is not None and int(manifest.get("round", -1)) != int(round_index):
        raise LineageError(
            f"round mismatch: expected {round_index}, got {manifest.get('round')}: {source}"
        )
    adapter_path = manifest.get("adapter_path")
    adapter_sha = manifest.get("adapter_sha256")
    if adapter_path:
        if not Path(adapter_path).exists():
            raise LineageError(f"adapter path is missing: {adapter_path}")
        actual_sha = sha256_tree(adapter_path)
        if adapter_sha and actual_sha != adapter_sha:
            raise LineageError(f"adapter hash mismatch: {adapter_path}")
    return manifest


def latest_global_step(checkpoint_root: str | os.PathLike[str]) -> Path:
    root = Path(checkpoint_root).resolve()
    tracker = root / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        step = int(tracker.read_text(encoding="utf-8").strip())
        candidate = root / f"global_step_{step}"
        if candidate.is_dir():
            return candidate
    candidates = []
    for candidate in root.glob("global_step_*"):
        try:
            step = int(candidate.name.split("global_step_", 1)[1])
        except (IndexError, ValueError):
            continue
        if candidate.is_dir():
            candidates.append((step, candidate))
    if not candidates:
        raise LineageError(f"no global_step checkpoint under {root}")
    return max(candidates)[1]


def snapshot_lora_adapter(source: Path, destination: Path) -> Path:
    if not (source / "adapter_model.safetensors").is_file():
        raise LineageError(f"LoRA adapter was not saved: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    backup = destination.with_name(f".{destination.name}.{os.getpid()}.bak")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    if sha256_tree(temporary) != sha256_tree(source):
        shutil.rmtree(temporary)
        raise LineageError(f"LoRA adapter snapshot verification failed: {source}")
    if destination.exists():
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(destination, backup)
    os.replace(temporary, destination)
    if backup.exists():
        shutil.rmtree(backup)
    return destination


def write_trained_manifest(
    path: str | os.PathLike[str],
    *,
    role: str,
    backbone: str,
    round_index: int,
    model_path: str,
    seed: int,
    parent_manifest_path: str,
    training_data_manifest_path: str,
    checkpoint_root: str,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if round_index <= 0:
        raise ValueError("trained checkpoints must have round_index > 0")
    parent = load_checkpoint_manifest(
        parent_manifest_path,
        role=role,
        backbone=backbone,
        round_index=round_index - 1,
    )
    step_path = latest_global_step(checkpoint_root)
    step_adapter_path = step_path / "actor" / "lora_adapter"
    adapter_path = snapshot_lora_adapter(
        step_adapter_path,
        Path(path).resolve().parent / "adapter",
    )
    data_manifest = Path(training_data_manifest_path).resolve()
    if not data_manifest.exists():
        raise LineageError(f"training data manifest is missing: {data_manifest}")
    frozen_training_config = dict(training_config or {})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "kind": "checkpoint",
        "role": role,
        "backbone": backbone,
        "round": int(round_index),
        "created_at": utc_now(),
        "seed": int(seed),
        "base_model": model_identity(model_path),
        "parent_manifest": str(Path(parent_manifest_path).resolve()),
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "training_data_manifest": str(data_manifest),
        "training_data_manifest_sha256": sha256_file(data_manifest),
        "training_config": frozen_training_config,
        "training_config_sha256": sha256_bytes(
            canonical_json(frozen_training_config).encode("utf-8")
        ),
        "checkpoint_path": str(step_path),
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_tree(adapter_path),
        "status": "trained",
    }
    if parent.get("role") != manifest["role"]:
        raise LineageError("parent role changed during manifest creation")
    atomic_write_json(path, manifest)
    return manifest


def feedback_fingerprints(path: str | os.PathLike[str]) -> set[str]:
    values = set()
    for row in read_jsonl(path):
        fingerprint = row.get("scenario_fingerprint")
        if fingerprint:
            values.add(str(fingerprint))
    return values


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def parquet_lineage(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    values = []
    for row in frame.to_dict(orient="records"):
        extra = _as_dict(row.get("extra_info"))
        scenario = _as_dict(row.get("scenario"))
        values.append(
            {
                "scenario_id": str(row.get("scenario_id") or extra.get("scenario_id") or scenario.get("scenario_id")),
                "task_id": str(row.get("task_id") or extra.get("task_id") or ""),
                "scenario_fingerprint": str(
                    extra.get("scenario_fingerprint") or scenario_fingerprint(scenario)
                ),
                "source_dca_round": int(extra.get("source_dca_round", -1)),
                "source_checkpoint_manifest_sha256": str(
                    extra.get("source_checkpoint_manifest_sha256", "")
                ),
            }
        )
    return values


def validate_round_lineage(
    *,
    dca_manifest_path: str,
    feedback_log_path: str,
    pool_manifest_path: str,
    split_paths: Iterable[str],
    backbone: str,
    target_round: int,
) -> dict[str, Any]:
    dca_manifest = load_checkpoint_manifest(
        dca_manifest_path,
        role="dca",
        backbone=backbone,
        round_index=target_round,
    )
    if str(
        (dca_manifest.get("training_config", {}) or {}).get(
            "tmcd_release_revision", ""
        )
    ) != TMCD_RELEASE_REVISION:
        raise LineageError("DCA checkpoint release revision mismatch")
    pool_manifest = read_json(pool_manifest_path)
    if pool_manifest.get("kind") != "vda_pool":
        raise LineageError("pool manifest kind must be vda_pool")
    if str(pool_manifest.get("tmcd_release_revision", "")) != TMCD_RELEASE_REVISION:
        raise LineageError("VDA pool release revision mismatch")
    expected_manifest_sha = sha256_file(dca_manifest_path)
    if pool_manifest.get("source_dca_checkpoint_manifest_sha256") != expected_manifest_sha:
        raise LineageError("VDA pool was not generated from the declared DCA checkpoint")
    if parse_utc(pool_manifest["created_at"]) < parse_utc(dca_manifest["created_at"]):
        raise LineageError("VDA candidates predate DCA checkpoint creation")

    feedback = feedback_fingerprints(feedback_log_path)
    declared_paths = {
        str(Path(path).resolve()): split
        for split, path in (pool_manifest.get("paths", {}) or {}).items()
        if split in {"train", "dev", "xplay"}
    }
    split_lineage: list[dict[str, Any]] = []
    actual_split_counts: dict[str, int] = {}
    actual_split_task_counts: dict[str, dict[str, int]] = {}
    for path in split_paths:
        resolved = str(Path(path).resolve())
        split = declared_paths.get(resolved)
        if split is None:
            raise LineageError(f"undeclared VDA split path: {resolved}")
        rows = parquet_lineage(path)
        split_lineage.extend(rows)
        actual_split_counts[split] = len(rows)
        task_counts: dict[str, int] = {}
        for row in rows:
            task_id = row["task_id"]
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        actual_split_task_counts[split] = task_counts
    declared_split_counts = {
        str(split): int(count)
        for split, count in (pool_manifest.get("split_counts", {}) or {}).items()
    }
    if actual_split_counts != declared_split_counts:
        raise LineageError(
            f"VDA split counts differ from manifest: {actual_split_counts} != {declared_split_counts}"
        )
    declared_task_counts = {
        str(split): {str(task): int(count) for task, count in counts.items()}
        for split, counts in (pool_manifest.get("split_task_counts", {}) or {}).items()
    }
    if actual_split_task_counts != declared_task_counts:
        raise LineageError(
            "VDA split task counts differ from manifest: "
            f"{actual_split_task_counts} != {declared_task_counts}"
        )
    if len(split_lineage) != int(pool_manifest.get("selected_count", -1)):
        raise LineageError("VDA selected_count differs from split row count")
    split_fingerprints = {row["scenario_fingerprint"] for row in split_lineage}
    if len(split_fingerprints) != len(split_lineage):
        raise LineageError("VDA train/dev/xplay contain duplicate scenario fingerprints")
    overlap = sorted(feedback & split_fingerprints)
    if overlap:
        raise LineageError(f"DCA feedback leaked into VDA data: {overlap[:5]}")
    for row in split_lineage:
        if row["source_dca_round"] != int(target_round):
            raise LineageError(
                f"scenario {row['scenario_id']} came from DCA round {row['source_dca_round']}, "
                f"expected {target_round}"
            )
        if row["source_checkpoint_manifest_sha256"] != expected_manifest_sha:
            raise LineageError(f"scenario {row['scenario_id']} has the wrong DCA checkpoint hash")
    return {
        "ok": True,
        "backbone": backbone,
        "target_round": int(target_round),
        "feedback_count": len(feedback),
        "vda_scenario_count": len(split_lineage),
        "feedback_vda_overlap_count": 0,
        "dca_checkpoint_manifest_sha256": expected_manifest_sha,
        "validated_at": utc_now(),
    }


@dataclass(frozen=True)
class RoundLayout:
    root: Path
    backbone: str
    source_round: int
    artifact_scope: str = "formal"
    experiment_variant: str = "full"

    def __post_init__(self) -> None:
        if self.artifact_scope not in {
            "formal",
            "pilot",
            "tmcd_v2",
            "tmcd_v2_pilot",
            "tmcd_v24",
        }:
            raise ValueError(f"unsupported artifact scope: {self.artifact_scope}")
        if not self.experiment_variant or "/" in self.experiment_variant or ".." in self.experiment_variant:
            raise ValueError(f"invalid experiment variant: {self.experiment_variant}")

    @property
    def target_round(self) -> int:
        return self.source_round + 1

    @property
    def data_dir(self) -> Path:
        tree = {
            "formal": "co_evolution",
            "pilot": "co_evolution_pilot",
            "tmcd_v2": "tmcd_v2",
            "tmcd_v2_pilot": "tmcd_v2_pilot",
            "tmcd_v24": "tmcd_v24",
        }[self.artifact_scope]
        base = self.root / "data" / tree
        if self.experiment_variant != "full":
            base = base / "ablations" / self.experiment_variant
        return base / self.backbone / f"round_{self.target_round}"

    def checkpoint_dir(self, role: str, round_index: int | None = None) -> Path:
        if role not in ROLES:
            raise ValueError(f"unsupported role: {role}")
        index = self.target_round if round_index is None else int(round_index)
        if self.artifact_scope in {"tmcd_v2", "tmcd_v2_pilot", "tmcd_v24"}:
            tree = self.artifact_scope
            base = self.root / "checkpoints" / tree
            if self.experiment_variant != "full":
                base = base / "ablations" / self.experiment_variant
            return base / self.backbone / role / f"round_{index}"
        tree = "checkpoints" if self.artifact_scope == "formal" else "checkpoints_pilot"
        return self.root / tree / self.backbone / role / f"round_{index}"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "round_state.json"


def load_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"schema_version": SCHEMA_VERSION, "stages": {}, "created_at": utc_now()}
    return read_json(target)


def mark_stage(path: str | os.PathLike[str], stage: str, status: str, **details: Any) -> None:
    state = load_state(path)
    stages = state.setdefault("stages", {})
    stages[stage] = {"status": status, "updated_at": utc_now(), **details}
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)


def stage_complete(path: str | os.PathLike[str], stage: str) -> bool:
    state = load_state(path)
    return state.get("stages", {}).get(stage, {}).get("status") == "completed"
