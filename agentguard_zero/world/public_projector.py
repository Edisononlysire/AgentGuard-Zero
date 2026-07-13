from __future__ import annotations

import copy
from typing import Any, Iterable


FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "is_fake",
        "truth_value",
        "true_objective",
        "oracle",
        "oracle_label",
        "oracle_objective",
        "hidden_source_behavior",
        "hidden_behavior_state",
        "behavior_schedule",
        "future_schedule",
        "spoofability",
    }
)


def assurance_level(spoofability: Any) -> str:
    """Convert a hidden continuous spoofability value to a coarse public prior."""
    try:
        value = float(spoofability)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 0.33:
        return "high"
    if value <= 0.66:
        return "medium"
    return "low"


def project_public(value: Any, forbidden_keys: Iterable[str] = FORBIDDEN_PUBLIC_KEYS) -> Any:
    """Recursively project internal state into a public, oracle-free representation."""
    forbidden = set(forbidden_keys)
    if isinstance(value, list):
        return [project_public(item, forbidden) for item in value]
    if isinstance(value, tuple):
        return [project_public(item, forbidden) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)

    projected: dict[str, Any] = {}
    if "spoofability" in value and "source_assurance_level" not in value:
        projected["source_assurance_level"] = assurance_level(value.get("spoofability"))
    for key, item in value.items():
        if str(key) in forbidden:
            continue
        projected[str(key)] = project_public(item, forbidden)
    return projected


def forbidden_public_paths(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key) in FORBIDDEN_PUBLIC_KEYS:
                findings.append(child)
            findings.extend(forbidden_public_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(forbidden_public_paths(item, f"{path}[{index}]"))
    return findings


def assert_public(value: Any) -> None:
    findings = forbidden_public_paths(value)
    if findings:
        raise ValueError(f"hidden fields leaked into public state: {', '.join(findings[:8])}")


def project_event(internal_event: dict[str, Any]) -> dict[str, Any]:
    public = project_public(internal_event)
    assert_public(public)
    return public

