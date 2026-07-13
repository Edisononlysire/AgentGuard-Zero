from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any


class OracleLedger:
    """Append-only hidden labels used only by terminal reward and evaluation."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self._rows.append(copy.deepcopy(row))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(copy.deepcopy(self._rows))

    def __len__(self) -> int:
        return len(self._rows)

    def snapshot(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._rows)
