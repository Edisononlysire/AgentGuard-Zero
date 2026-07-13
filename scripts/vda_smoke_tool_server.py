#!/usr/bin/env python3
"""Minimal VDA tool server for AgentGuard-Zero warmup smoke training.

This server intentionally does not execute code, payloads, network calls, or
cyber operations. It only satisfies the verl-tool `/get_observation` contract
so the VDA warmup path can exercise agent rollout, reward, PPO update, and
checkpointing end to end before the full Level-1 simulator server is enabled.
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


LOGGER = logging.getLogger("vda_smoke_tool_server")


def _as_list(value: Any, n: int, default: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return [default for _ in range(n)]
    return [value for _ in range(n)]


def _build_response(payload: dict[str, Any]) -> dict[str, Any]:
    trajectory_ids = payload.get("trajectory_ids") or []
    actions = payload.get("actions") or []
    n = max(len(trajectory_ids), len(actions), 1)
    actions = _as_list(actions, n, "")
    finish = _as_list(payload.get("finish"), n, False)
    is_last_step = _as_list(payload.get("is_last_step"), n, False)

    observations = []
    dones = []
    valids = []
    for idx in range(n):
        action = actions[idx] if idx < len(actions) else ""
        action_text = action if isinstance(action, str) else json.dumps(action, ensure_ascii=False)
        valid = bool(action_text.strip()) or bool(finish[idx])
        observations.append(
            {
                "obs": "",
                "reward": None,
                "tool": "VDA-SMOKE",
                "action_preview": action_text[:160],
                "invalid_reason": None if valid else "empty_action",
                "smoke_done": True,
                "is_last_step": bool(is_last_step[idx]),
            }
        )
        dones.append(1)
        valids.append(1 if valid else 0)

    return {"observations": observations, "dones": dones, "valids": valids}


class Handler(BaseHTTPRequestHandler):
    server_version = "VDASmokeToolServer/0.1"

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/get_observation":
            self._send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            response = _build_response(payload)
            LOGGER.info(
                "handled batch trajectories=%s actions=%s",
                len(payload.get("trajectory_ids") or []),
                len(payload.get("actions") or []),
            )
            self._send_json(200, response)
        except Exception as exc:  # pragma: no cover - defensive server path
            LOGGER.exception("request failed")
            self._send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30150)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOGGER.info("VDA smoke tool server listening on http://%s:%s/get_observation", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
