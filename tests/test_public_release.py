from __future__ import annotations

import hashlib
import importlib
import json
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


def test_release_metadata_separates_existing_results_from_proposed_methods() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "configs/t12/t12_ranker_public.json").read_text())
    assert contract["status"] == "audited_historical_result_architecture"
    assert contract["scope"]["workflow_requirement_hints_in_model_input"] is True
    assert contract["scope"]["complete_leakage_freedom_established"] is False
    assert contract["runtime"]["ecrg_during_parameter_training"] is False
    assert contract["runtime"]["ecrg_in_audited_evaluation"] is False
    assert contract["runtime"]["selected_checkpoint_epoch"] == 2
    assert contract["runtime"]["selected_checkpoint_optimizer_steps"] == 500
    assert contract["data"]["dev_records"] == 400
    assert contract["data"]["epoch_selection_trajectories"] == 400
    assert contract["vnext_active_probing"]["status"] == "design_only_not_current_result"
    proposed = contract["vnext_active_probing"]
    assert proposed["plan_revision"] == "2026-09-07-r2"
    assert proposed["core_training_arm"] == "public_value_policy_supervision"
    assert proposed["voi_auxiliary_required_for_core_method"] is False
    assert proposed["ecrg_comparison"] == "same_checkpoint_inference_only"
    assert proposed["implementation_gate_status"] == "pending"
    assert proposed["plan_revision"] in (root / "docs/PLAN.md").read_text()
    assert proposed["plan_revision"] in (root / "docs/STATUS.md").read_text()
    assert proposed["plan_revision"] in (root / "README.md").read_text()


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
