"""Batch reward for DCA training using live VDA rollout services."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.env.checker import full_check, parse_scenario_json
from agentguard_zero.rewards.dca_reward import compute_dca_reward
from agentguard_zero.schemas.scenario_schema_v2 import public_prefix_hash
from agentguard_zero.training.coevolution import scenario_fingerprint, utc_now


_FINGERPRINT_CACHE_LOCK = threading.Lock()
_FINGERPRINT_CACHE: dict[str, tuple[int, int, set[str]]] = {}
_APPEND_COUNTS: dict[str, int] = {}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_json_object(text: str) -> tuple[dict[str, Any], bool, str]:
    scenario, ok, message = parse_scenario_json(text)
    if ok and isinstance(scenario, dict):
        return scenario, True, message

    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    best_size = -1
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and end > best_size:
            best = value
            best_size = end
    if best is None:
        return {}, False, message
    return best, True, "json_object_extracted"


def _service_urls() -> list[str]:
    raw = os.environ.get("AGZ_VDA_FEEDBACK_URLS", "")
    urls = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not urls:
        raise RuntimeError("AGZ_VDA_FEEDBACK_URLS is required for DCA online reward")
    return urls


def _estimated_rollout_cost(scenario: dict[str, Any]) -> int:
    """Estimate VDA work from the task horizon for four-way load balancing."""
    metadata = scenario.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    task_id = str(metadata.get("task_id", "")).upper()
    if task_id not in {"T1", "T2", "T3", "T4"}:
        task_focus = str(metadata.get("task_focus", "")).upper()
        task_id = next((task for task in ("T1", "T2", "T3", "T4") if task in task_focus), "")
    return {"T1": 10, "T2": 16, "T3": 14, "T4": 10}.get(task_id, 16)


def _balance_feedback_groups(
    items: list[tuple[int, dict[str, Any]]], urls: list[str]
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """Assign longest trajectories first to the currently lightest service."""
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {url: [] for url in urls}
    loads = {url: 0 for url in urls}
    order = {url: index for index, url in enumerate(urls)}
    for item in sorted(items, key=lambda value: (-_estimated_rollout_cost(value[1]), value[0])):
        url = min(urls, key=lambda value: (loads[value], len(groups[value]), order[value]))
        groups[url].append(item)
        loads[url] += _estimated_rollout_cost(item[1])
    return groups


def _evaluate_batch(
    url: str, scenarios: list[dict[str, Any]], timeout: float
) -> list[dict[str, Any]]:
    payload = json.dumps({"scenarios": scenarios}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/evaluate_batch",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict) or not body.get("ok", False):
        raise RuntimeError(f"VDA feedback service rejected scenario: {body}")
    results = body.get("results")
    if not isinstance(results, list) or len(results) != len(scenarios):
        raise RuntimeError("VDA feedback service returned the wrong batch size")
    if not all(isinstance(result, dict) for result in results):
        raise RuntimeError("VDA feedback service returned a non-object result")
    return results


def _existing_fingerprints(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            stat = os.fstat(handle.fileno())
            signature = (stat.st_size, stat.st_mtime_ns)
            key = str(path)
            with _FINGERPRINT_CACHE_LOCK:
                cached = _FINGERPRINT_CACHE.get(key)
                if cached and cached[:2] == signature:
                    return set(cached[2])

            values: set[str] = set()
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fingerprint = (
                    value.get("scenario_fingerprint")
                    if isinstance(value, dict)
                    else None
                )
                if fingerprint:
                    values.add(str(fingerprint))
            with _FINGERPRINT_CACHE_LOCK:
                _FINGERPRINT_CACHE[key] = (*signature, set(values))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return values


def _safe_full_check(scenario: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    try:
        checks = full_check(scenario)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {
            "all_ok": False,
            "format": {"ok": False, "error": message},
            "valid": {"ok": False, "error": message},
            "solvable": {"ok": False, "error": message},
            "safe": {"ok": False, "error": message},
        }, message
    return checks, None


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            key = str(path)
            before = os.fstat(handle.fileno())
            before_signature = (before.st_size, before.st_mtime_ns)
            with _FINGERPRINT_CACHE_LOCK:
                cached = _FINGERPRINT_CACHE.get(key)
                cached_values = (
                    set(cached[2])
                    if cached and cached[:2] == before_signature
                    else None
                )
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush()
            with _FINGERPRINT_CACHE_LOCK:
                append_count = _APPEND_COUNTS.get(key, 0) + 1
                _APPEND_COUNTS[key] = append_count
            fsync_interval = max(
                1, int(os.environ.get("AGZ_DCA_REWARD_FSYNC_EVERY_BATCHES", "1"))
            )
            if append_count % fsync_interval == 0:
                os.fsync(handle.fileno())

            after = os.fstat(handle.fileno())
            with _FINGERPRINT_CACHE_LOCK:
                if cached_values is None:
                    _FINGERPRINT_CACHE.pop(key, None)
                else:
                    cached_values.update(
                        str(row.get("scenario_fingerprint"))
                        for row in rows
                        if row.get("scenario_fingerprint")
                    )
                    _FINGERPRINT_CACHE[key] = (
                        after.st_size,
                        after.st_mtime_ns,
                        cached_values,
                    )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def compute_score(
    data_sources: list[Any] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[Any] | None = None,
    extra_infos: list[Any] | None = None,
    **_: Any,
) -> list[dict[str, float]]:
    del data_sources, ground_truths
    solutions = [] if solution_strs is None else list(solution_strs)
    extras = (
        [{} for _ in solutions]
        if extra_infos is None
        else list(extra_infos)
    )
    if len(extras) < len(solutions):
        extras.extend({} for _ in range(len(solutions) - len(extras)))

    urls = _service_urls()
    timeout = float(os.environ.get("AGZ_VDA_FEEDBACK_TIMEOUT", "300"))
    log_path = Path(os.environ.get("AGZ_DCA_FEEDBACK_LOG", "")).resolve()
    if not str(log_path) or str(log_path) == "/":
        raise RuntimeError("AGZ_DCA_FEEDBACK_LOG is required for DCA online reward")

    existing = _existing_fingerprints(log_path)
    parsed: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any] | None] = [None] * len(solutions)
    errors: list[str | None] = [None] * len(solutions)

    for index, (solution, raw_extra) in enumerate(zip(solutions, extras)):
        extra = _as_dict(raw_extra)
        scenario, parse_ok, parse_message = _extract_json_object(str(solution))
        metadata = _as_dict(scenario.get("metadata")) if scenario else {}
        metadata.update(
            {
                "feedback_candidate": True,
                "source_dca_round": int(extra.get("source_dca_round", -1)),
                "source_vda_round": int(extra.get("source_vda_round", -1)),
                "task_focus": str(extra.get("task_focus", "")),
                "prompt_index": extra.get("index"),
                "prompt_nonce": extra.get("prompt_nonce"),
                "experiment_variant": str(extra.get("experiment_variant", "full")),
            }
        )
        if scenario:
            scenario["metadata"] = metadata
            if scenario.get("protocol_version") == "tmcd-v2" and scenario.get("scenario_family") == "trust_betrayal":
                scenario["prefix_hash"] = public_prefix_hash(scenario)
            fingerprint = scenario_fingerprint(scenario)
            scenario["scenario_id"] = (
                f"FB-D{metadata['source_dca_round']}-V{metadata['source_vda_round']}-"
                f"{index:04d}-{fingerprint[:12]}"
            )
        else:
            fingerprint = scenario_fingerprint({"raw": str(solution)})
        parsed.append(
            {
                "scenario": scenario,
                "fingerprint": fingerprint,
                "parse_ok": parse_ok,
                "parse_message": parse_message,
                "extra": extra,
                "raw_solution": str(solution),
            }
        )

    with ThreadPoolExecutor(max_workers=max(1, len(urls))) as executor:
        valid_items: list[tuple[int, dict[str, Any]]] = []
        for index, item in enumerate(parsed):
            checks, checker_error = (
                _safe_full_check(item["scenario"]) if item["parse_ok"] else ({}, None)
            )
            if checker_error:
                errors[index] = checker_error
            if not item["parse_ok"] or not checks.get("all_ok", False):
                evaluations[index] = {
                    "checks": checks,
                    "oracle_solvable": False,
                    "ambiguity_penalty": 1.0 if item["parse_ok"] else 0.0,
                }
                continue
            valid_items.append((index, item["scenario"]))
        grouped = _balance_feedback_groups(valid_items, urls)
        futures = {
            executor.submit(
                _evaluate_batch,
                url,
                [scenario for _, scenario in items],
                timeout,
            ): items
            for url, items in grouped.items()
            if items
        }
        for future in as_completed(futures):
            items = futures[future]
            try:
                results = future.result()
                for (index, _), result in zip(items, results):
                    evaluations[index] = result
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                for index, _ in items:
                    errors[index] = message
                    evaluations[index] = {
                        "checks": _safe_full_check(parsed[index]["scenario"])[0],
                        "oracle_solvable": False,
                        "feedback_service_error": message,
                    }

    scores: list[dict[str, float]] = []
    log_rows: list[dict[str, Any]] = []
    batch_seen = set(existing)
    for index, item in enumerate(parsed):
        evaluation = evaluations[index] or {"checks": {}, "oracle_solvable": False}
        checks = evaluation.get("checks", {}) or {}
        reward_scenario = item["scenario"] if checks.get("all_ok", False) else {}
        try:
            components = compute_dca_reward(
                reward_scenario,
                evaluation,
                seen_fingerprints=batch_seen,
                task_focus=str(item["extra"].get("task_focus", "")),
            )
        except Exception as exc:
            errors[index] = errors[index] or f"{type(exc).__name__}: {exc}"
            components = compute_dca_reward(
                {},
                {"checks": {}, "oracle_solvable": False},
                seen_fingerprints=batch_seen,
                task_focus=str(item["extra"].get("task_focus", "")),
            )
        score = {"score": float(components["overall"]), **components}
        scores.append(score)
        log_rows.append(
            {
                "logged_at": utc_now(),
                "scenario_fingerprint": item["fingerprint"],
                "scenario": item["scenario"],
                "raw_solution": item["raw_solution"],
                "parse_ok": item["parse_ok"],
                "parse_message": item["parse_message"],
                "extra_info": item["extra"],
                "vda_evaluation": evaluation,
                "reward": score,
                "feedback_service_error": errors[index],
            }
        )
        batch_seen.add(item["fingerprint"])

    _append_rows(log_path, log_rows)
    require_valid = os.environ.get("AGZ_DCA_REQUIRE_VALID_BATCH", "0").strip().lower()
    if require_valid in {"1", "true", "yes", "on"} and not any(
        bool((evaluation or {}).get("oracle_solvable", False))
        and "current_vda_safe_success" in (evaluation or {})
        for evaluation in evaluations
    ):
        raise RuntimeError("DCA batch produced no valid scenario that reached current-VDA feedback")
    return scores
