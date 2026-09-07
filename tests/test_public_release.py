from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from agentguard_zero.defender_state.evidence_store import EvidenceStore
from scripts.check_public_release import check_release


def test_current_release_documents_and_result_contract() -> None:
    assert check_release()["documents"] == 6


def test_audited_historical_results_are_unchanged() -> None:
    path = Path(__file__).resolve().parents[1] / "results/t12_legacy_retention_20260907.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "d1cfa50ec1990a23579e46aeeb98eb714c52aa1d26c313c8621c60f597e43dda"
    )


def test_current_t12_entrypoints_still_import() -> None:
    for module in (
        "scripts.build_v11_t12_native_v2",
        "scripts.train_candidate_ranker",
        "scripts.launch_candidate_ddp",
        "scripts.freeze_v11_t12_training_protocol",
        "scripts.eval_candidate_ranker_offline",
        "scripts.eval_v11_single_expert_policy",
        "scripts.eval_v11_t12_retention_gate",
        "scripts.select_v11_t12_epoch",
        "scripts.merge_v11_policy_shards",
        "scripts.smoke_t12_scenarios",
    ):
        importlib.import_module(module)


def test_delayed_probe_is_indexed_and_can_parent_followup_evidence() -> None:
    store = EvidenceStore()
    event = {
        "event_id": "probe-event",
        "source_id": "decoy-sensor",
        "type": "decoy_probe_result",
        "probe_id": "p-1",
    }
    evidence_id = store.add_probe_result(event, time=1, tool="DecoyProbe")
    assert store.evidence_for_event("probe-event") == evidence_id
    assert store.validate_refs([evidence_id], time=0)[0] is False
    assert store.validate_refs([evidence_id], time=1) == (True, "ok")
    followup = store.add_tool_result(
        {"tool": "CrossCheck", "event_id": "probe-event"},
        time=1,
        parent_evidence_ids=[store.evidence_for_event("probe-event")],
    )
    record = store.get(followup, time=2)
    assert record["parent_evidence_ids"] == [evidence_id]
    assert record["root_source_ids"] == ["decoy-sensor"]
