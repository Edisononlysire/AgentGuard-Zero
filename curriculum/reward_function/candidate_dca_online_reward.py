"""VerL-compatible online DCA reward using candidate-ranker feedback services."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.env.checker import parse_scenario_json
from agentguard_zero.training.coevolution import scenario_fingerprint, utc_now


def _service_urls() -> list[str]:
    urls = [
        item.strip().rstrip("/")
        for item in os.environ.get("AGZ_VDA_FEEDBACK_URLS", "").split(",")
        if item.strip()
    ]
    if not urls:
        raise RuntimeError("AGZ_VDA_FEEDBACK_URLS is required")
    return urls


def _evaluate(url: str, scenario: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}/evaluate",
        data=json.dumps({"scenario": scenario}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok") or not isinstance(payload.get("result"), dict):
        raise RuntimeError(f"candidate feedback service rejected scenario: {payload}")
    return payload["result"]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def compute_score(
    data_sources: list[Any] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[Any] | None = None,
    extra_infos: list[Any] | None = None,
    **_: Any,
) -> list[dict[str, float]]:
    del data_sources, ground_truths, extra_infos
    urls = _service_urls()
    log_path = Path(os.environ["AGZ_DCA_FEEDBACK_LOG"])
    results: list[dict[str, float]] = []
    for index, text in enumerate(solution_strs or []):
        scenario, parse_ok, parse_message = parse_scenario_json(str(text))
        if parse_ok:
            feedback = _evaluate(urls[index % len(urls)], scenario)
            reward = float(feedback.get("reward", -1.0))
        else:
            feedback = {
                "parser_failure": True,
                "parse_message": parse_message,
                "reward": -1.0,
            }
            reward = -1.0
        _append(
            log_path,
            {
                "created_at": utc_now(),
                "parse_ok": parse_ok,
                "scenario_fingerprint": scenario_fingerprint(
                    scenario if parse_ok else {"raw": str(text)}
                ),
                "feedback": feedback,
            },
        )
        results.append({"score": reward, "acc": reward})
    return results
