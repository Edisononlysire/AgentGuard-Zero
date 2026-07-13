from __future__ import annotations

import copy
import unittest
from pathlib import Path

from agentguard_zero.defender_state.evidence_store import EvidenceStore
from agentguard_zero.defender_state.memory_fsm import EvidenceStateMemory
from agentguard_zero.defender_state.trust_manager import ContextualTrustManager
from agentguard_zero.env.checker import full_check
from agentguard_zero.env.cyber_env_v2 import CyberDefenseEnvV2
from agentguard_zero.env.oracle_v2 import score_trajectory_v2
from agentguard_zero.governance.v5c import select_v5c
from agentguard_zero.schemas.action_schema_v4 import DEFAULT_ACTION_PACKET_V4
from agentguard_zero.schemas.scenario_schema_v2 import (
    OOD_FAMILIES,
    minimal_example_v2,
    paired_counterpart_v2,
    paired_minimal_examples_v2,
    public_prefix_hash,
    validate_pair_v2,
    validate_scenario_v2,
)
from agentguard_zero.world.public_projector import assert_public, forbidden_public_paths, project_public


def _action(**updates):
    packet = copy.deepcopy(DEFAULT_ACTION_PACKET_V4)
    packet.update(updates)
    return packet


def _has_key(value, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_has_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_has_key(item, target) for item in value)
    return False


class TMCDV2IsolationTests(unittest.TestCase):
    def test_public_projection_is_recursive(self) -> None:
        internal = {
            "outer": [{"truth_value": False, "nested": {"is_fake": True, "spoofability": 0.9}}],
            "oracle": {"true_objective": "exfiltration"},
        }
        public = project_public(internal)
        self.assertEqual(forbidden_public_paths(public), [])
        self.assertEqual(public["outer"][0]["nested"]["source_assurance_level"], "low")

    def test_same_step_tool_evidence_is_unavailable(self) -> None:
        store = EvidenceStore()
        event = {"event_id": "e0", "source_id": "A", "type": "alert"}
        parent = store.add_event(event, time=0)
        tool = store.add_tool_result(
            {"tool": "SourceChallenge", "event_id": "e0", "verdict": "challenge_passed"},
            time=0,
            parent_evidence_ids=[parent],
        )
        self.assertEqual(store.validate_refs([tool], time=0)[0], False)
        self.assertEqual(store.validate_refs([tool], time=1), (True, "ok"))

    def test_circular_confirmation_is_not_independent(self) -> None:
        store = EvidenceStore()
        parent = store.add_event({"event_id": "e0", "source_id": "A", "type": "alert"}, time=0)
        derived = store.add_tool_result(
            {"tool": "CrossCheck", "event_id": "e0", "verdict": "supported"},
            time=0,
            parent_evidence_ids=[parent],
        )
        self.assertEqual(store.independent_count([parent, derived], time=1), 1)

    def test_reward_never_enters_public_state(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=2)
        initial = env.observe()
        next_obs, tool_result, _done = env.step(_action())
        for value in (initial, next_obs, tool_result, env.history):
            assert_public(value)
            self.assertFalse(_has_key(value, "reward"))

    def test_v5c_rejects_hidden_input(self) -> None:
        context = {"observation": {"protocol_version": "tmcd-v2", "oracle": {"x": 1}}}
        with self.assertRaises(ValueError):
            select_v5c(context, ["{}"])


class TMCDV2StateTests(unittest.TestCase):
    def _state_fixture(self):
        store = EvidenceStore()
        claim = {"entity_id": "asset-A", "predicate": "risk_level", "object": "high", "scope": "exfiltration"}
        first = store.add_event(
            {"event_id": "claim-1", "source_id": "A", "type": "alert", "verdict": "supported", "claim_semantics": claim},
            time=0,
        )
        second = store.add_event(
            {"event_id": "claim-2", "source_id": "B", "type": "alert", "verdict": "supported", "claim_semantics": claim},
            time=0,
        )
        trust = ContextualTrustManager()
        trust.register_claim(
            {"event_id": "claim-1", "source_id": "A", "source_assurance_level": "high", "claim_semantics": claim},
            time=0,
        )
        trust.apply(
            [{"op": "support", "source_id": "A", "event_id": "claim-1", "evidence_refs": [first, second]}],
            evidence_store=store,
            time=0,
        )
        memory = EvidenceStateMemory()
        ingest = memory.apply(
            [{"op": "ingest", "claim": claim, "source_ids": ["A"], "evidence_refs": [first]}],
            evidence_store=store,
            trust_manager=trust,
            time=0,
        )[0]
        return store, trust, memory, first, second, ingest["memory_id"]

    def test_direct_confirm_falls_back_to_quarantine(self) -> None:
        store, trust, memory, _first, _second, _memory_id = self._state_fixture()
        claim = {"entity_id": "asset-B", "predicate": "risk_level", "object": "low", "scope": "exfiltration"}
        matching = store.add_event(
            {"event_id": "claim-B", "source_id": "A", "type": "alert", "claim_semantics": claim},
            time=0,
        )
        result = memory.apply(
            [{"op": "ingest", "claim": claim, "source_ids": ["A"], "evidence_refs": [matching], "target_status": "confirmed"}],
            evidence_store=store,
            trust_manager=trust,
            time=0,
        )[0]
        self.assertTrue(result["fallback"])
        self.assertEqual(memory.records[result["memory_id"]]["status"], "quarantined")

    def test_promotion_requires_two_independent_roots(self) -> None:
        store, trust, memory, first, second, memory_id = self._state_fixture()
        denied = memory.apply(
            [{"op": "promote", "memory_id": memory_id, "event_id": "claim-1", "evidence_refs": [first]}],
            evidence_store=store,
            trust_manager=trust,
            time=0,
        )[0]
        allowed = memory.apply(
            [{"op": "promote", "memory_id": memory_id, "event_id": "claim-1", "evidence_refs": [first, second]}],
            evidence_store=store,
            trust_manager=trust,
            time=0,
        )[0]
        self.assertFalse(denied["committed"])
        self.assertTrue(allowed["committed"])
        self.assertEqual(memory.records[memory_id]["status"], "confirmed")

    def test_unrelated_evidence_cannot_satisfy_promotion(self) -> None:
        store, trust, memory, first, _second, memory_id = self._state_fixture()
        unrelated = store.add_event(
            {
                "event_id": "unrelated",
                "source_id": "C",
                "type": "alert",
                "verdict": "supported",
                "claim_semantics": {
                    "entity_id": "asset-Z",
                    "predicate": "owner",
                    "object": "team-Z",
                    "scope": "inventory",
                },
            },
            time=0,
        )
        result = memory.apply(
            [{"op": "promote", "memory_id": memory_id, "event_id": "claim-1", "evidence_refs": [first, unrelated]}],
            evidence_store=store,
            trust_manager=trust,
            time=0,
        )[0]
        self.assertFalse(result["committed"])
        self.assertIn("irrelevant_claim_evidence", result["reason"])

    def test_illegal_transition_and_exclusivity(self) -> None:
        store, trust, memory, first, _second, memory_id = self._state_fixture()
        result = memory.apply(
            [{"op": "reopen", "memory_id": memory_id, "event_id": "claim-1", "evidence_refs": [first]}],
            evidence_store=store,
            trust_manager=trust,
            time=0,
        )[0]
        self.assertFalse(result["committed"])
        partitions = memory.partition_view()
        appearances = sum(memory_id in values for values in partitions.values())
        self.assertEqual(appearances, 1)

    def test_rejected_recovery_requires_new_independent_evidence(self) -> None:
        store, trust, memory, first, second, memory_id = self._state_fixture()
        memory.records[memory_id]["status"] = "rejected"
        trust.claim_trust["claim-1"]["status"] = "challenged"
        denied = memory.apply(
            [{"op": "reopen", "memory_id": memory_id, "event_id": "claim-1", "evidence_refs": [first]}],
            evidence_store=store,
            trust_manager=trust,
            time=0,
        )[0]
        allowed = memory.apply(
            [{"op": "reopen", "memory_id": memory_id, "event_id": "claim-1", "evidence_refs": [first, second]}],
            evidence_store=store,
            trust_manager=trust,
            time=0,
        )[0]
        self.assertFalse(denied["committed"])
        self.assertTrue(allowed["committed"])
        self.assertEqual(memory.records[memory_id]["status"], "quarantined")

    def test_trust_manager_has_no_oracle_state(self) -> None:
        manager = ContextualTrustManager()
        self.assertFalse(hasattr(manager, "oracle_ledger"))
        self.assertFalse(_has_key(manager.public_snapshot(), "truth_value"))


class TMCDV2DatasetTests(unittest.TestCase):
    def test_t2_pair_prefixes_match(self) -> None:
        first, second = paired_minimal_examples_v2()
        self.assertEqual(validate_pair_v2(first, second), (True, "ok"))
        self.assertEqual(public_prefix_hash(first), public_prefix_hash(second))
        generated = paired_counterpart_v2(first)
        self.assertEqual(validate_pair_v2(first, generated), (True, "ok"))

    def test_t2_prefix_hash_is_total_for_malformed_candidates(self) -> None:
        scenario = minimal_example_v2(trajectory_type="betrayal")
        scenario["source_profiles"][0]["public_prior"] = [0.5]
        scenario["event_schedule"][0]["time"] = [0]
        value = public_prefix_hash(scenario)
        self.assertEqual(len(value), 64)
        self.assertEqual(value, public_prefix_hash(scenario))

    def test_ood_family_is_blocked_from_training(self) -> None:
        scenario = minimal_example_v2()
        scenario["scenario_family"] = sorted(OOD_FAMILIES)[0]
        scenario["distribution"] = "ood"
        valid, reason = validate_scenario_v2(scenario)
        self.assertFalse(valid)
        self.assertEqual(reason, "ood_family_in_training_split")

    def test_v2_minimal_scenario_passes_hard_gate(self) -> None:
        self.assertTrue(full_check(minimal_example_v2())["all_ok"])

    def test_noop_and_isolate_extremes_have_expected_costs(self) -> None:
        no_op = CyberDefenseEnvV2(minimal_example_v2(), max_steps=1)
        no_op.observe()
        no_op.step(_action())
        no_op_score = score_trajectory_v2(no_op)
        isolate = CyberDefenseEnvV2(minimal_example_v2(), max_steps=1)
        isolate.observe()
        response = {"tier": "L3", "action": "Isolate", "target": "database"}
        belief = {"exfiltration": 0.85, "sabotage": 0.05, "persistence": 0.05, "credential_theft": 0.05}
        isolate.step(_action(response=response, belief=belief, uncertainty=0.8))
        isolate_score = score_trajectory_v2(isolate)
        self.assertTrue(no_op_score["attack_success"])
        self.assertGreater(isolate_score["business_cost"], no_op_score["business_cost"])
        self.assertGreaterEqual(isolate_score["unauthorized_high_impact"], 1)


class TMCDV2ReleaseTests(unittest.TestCase):
    def test_training_feedback_does_not_call_v5c(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "vda_feedback_server.py").read_text(encoding="utf-8")
        self.assertNotIn("select_candidate(", source)
        self.assertIn('"v5c_used": False', source)

    def test_rq6_same_state_layer_controls_are_registered(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "eval_tmcd_systems.py").read_text(encoding="utf-8")
        for system in (
            "react_base_tools",
            "react_state_layer",
            "trajectory_react_state_layer",
            "agentguard_zero_train",
            "agentguard_zero_full",
        ):
            self.assertIn(f'"{system}"', source)

    def test_formal_jobs_are_single_node_four_gpu_jobs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        jobs = root / "scripts" / "jobs"
        expected = {
            "tmcd_v2_4b_full_node175.dsub.sh": "cyclone001-agent-175",
            "tmcd_v2_4b_append_only_node208.dsub.sh": "cyclone001-agent-208",
            "tmcd_v2_9b_full_node217.dsub.sh": "cyclone001-agent-217",
        }
        for name, node in expected.items():
            source = (jobs / name).read_text(encoding="utf-8")
            self.assertIn('#DSUB -R "cpu=64;gpu=4;mem=230000"', source)
            self.assertIn(node, source)
            self.assertIn('source "${ROOT}/scripts/qwen35_env.sh"', source)
            self.assertIn('source "${ROOT}/scripts/env.sh"', source)
            self.assertIn("--dca-steps 50", source)
            self.assertIn("--vda-steps 75", source)

    def test_preflight_freezes_and_imports_training_framework(self) -> None:
        root = Path(__file__).resolve().parents[1]
        prepare_source = (root / "scripts" / "prepare_tmcd_v2_run.py").read_text(
            encoding="utf-8"
        )
        preflight_source = (root / "scripts" / "preflight_tmcd_v2_job.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("third_party/verl_tool/trainer/main_ppo.py", prepare_source)
        self.assertIn("third_party/verl/verl/trainer/main_ppo.py", prepare_source)
        self.assertIn("import verl; import verl_tool.trainer.main_ppo", preflight_source)
        self.assertIn("training framework hash mismatch", preflight_source)


if __name__ == "__main__":
    unittest.main()
