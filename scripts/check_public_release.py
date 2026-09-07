#!/usr/bin/env python3
"""Validate the focused source release without models, GPUs, or network access."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = {"PLAN.md", "ARCHITECTURE.md", "RESULTS.md", "STATUS.md"}


def check_release(root: Path = ROOT) -> dict[str, int]:
    actual_docs = {path.relative_to(root / "docs").as_posix() for path in (root / "docs").rglob("*.md")}
    if actual_docs != CURRENT_DOCS:
        raise ValueError(f"unexpected_document_set: {sorted(actual_docs)}")
    for relative in ("scripts/jobs", "artifacts", "curriculum", "third_party/verl", "third_party/verl_tool"):
        path = root / relative
        if path.exists() and any(item.is_file() and "__pycache__" not in item.parts for item in path.rglob("*")):
            raise ValueError(f"retired_release_content: {relative}")

    documents = [root / "README.md", root / "THIRD_PARTY_NOTICES.md", *sorted((root / "docs").glob("*.md"))]
    link_count = 0
    for document in documents:
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            path = unquote(target.split("#", 1)[0])
            if not (document.parent / path).exists():
                raise ValueError(f"broken_document_link: {document.relative_to(root)} -> {path}")
            link_count += 1

    source_count = 0
    for directory in ("agentguard_zero", "scripts", "tests"):
        for path in sorted((root / directory).rglob("*.py")):
            ast.parse(path.read_text(), filename=str(path))
            source_count += 1

    contract = json.loads((root / "configs/t12/t12_ranker_public.json").read_text())
    for field in ("design", "implementation_status"):
        if not (root / contract["vnext_active_probing"][field]).is_file():
            raise ValueError(f"broken_contract_link: {field}")
    results = json.loads((root / "results/t12_legacy_retention_20260907.json").read_text())
    if results["legacy_retention_gate"]["accepted"] is not False:
        raise ValueError("historical_failed_gate_must_be_preserved")
    current = [row for row in results["evaluations"] if row["system"] == "new_t12"]
    if len(current) != 3 or any(row["ecrg_enabled"] for row in current):
        raise ValueError("unexpected_historical_result_scope")
    for task, successes in (("T1", 50), ("T2", 100)):
        if sum(row["tasks"][task]["scenario_count"] for row in current) != 150:
            raise ValueError(f"unexpected_historical_denominator: {task}")
        if sum(row["tasks"][task]["safe_success_count"] for row in current) != successes:
            raise ValueError(f"unexpected_historical_success_count: {task}")
    return {"documents": len(documents), "local_links": link_count, "python_sources": source_count}


if __name__ == "__main__":
    print(json.dumps(check_release(), indent=2))
