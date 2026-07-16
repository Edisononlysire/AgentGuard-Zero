from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agentguard_zero.defender_state.evidence_store import EvidenceStore
from agentguard_zero.defender_state.memory_fsm import EvidenceStateMemory
from agentguard_zero.defender_state.retriever import retrieve_memory
from agentguard_zero.defender_state.trust_manager import ContextualTrustManager
from agentguard_zero.env.checker import full_check
from agentguard_zero.env.cyber_env_v2 import CyberDefenseEnvV2
from agentguard_zero.env.oracle_v2 import _probe_metrics, score_trajectory_v2
from agentguard_zero.evaluation.rq3_memory import (
    counterfactual_memory_impact,
    memory_lifecycle_metrics,
)
from agentguard_zero.governance.v5c import score_v5c_candidate, select_v5c
from agentguard_zero.protocol import TASK_FAMILY_MAP, TMCD_RELEASE_REVISION
from agentguard_zero.schemas.action_schema_v4 import DEFAULT_ACTION_PACKET_V4, parse_action_json_v4
from agentguard_zero.schemas.scenario_schema_v2 import (
    OOD_FAMILIES,
    minimal_example_v2,
    paired_counterpart_v2,
    paired_minimal_examples_v2,
    public_prefix_hash,
    validate_pair_v2,
    validate_scenario_v2,
)
from agentguard_zero.training.dca_dataset import (
    DCA_PROMPT_VERSION,
    TASK_FOCI,
    build_dca_messages,
    task_example_v2,
)
from agentguard_zero.training.coevolution import (
    LineageError,
    load_checkpoint_manifest,
    scenario_fingerprint,
    sha256_source_tree,
    sha256_tree,
    write_base_manifest,
    write_frozen_manifest,
)
from agentguard_zero.training.vda_dataset import (
    build_vda_prompt,
    public_instance_id,
    vda_system_prompt_v4,
)
from agentguard_zero.variants import (
    ACTIVE_PROBE_TOOLS,
    PASSIVE_VERIFICATION_TOOLS,
    TRAINING_VARIANTS,
    experiment_variant,
)
from agentguard_zero.world.public_projector import assert_public, forbidden_public_paths, project_public
from scripts.build_vda_round_pool import _split_stratified, _split_task_quotas, _task_id
from scripts.build_vda_partial_gate_dataset import collect_balanced_records
from scripts.eval_tmcd_systems import (
    SYSTEMS,
    SYSTEM_ALIASES,
    adapt_packet_v2,
    default_candidate_count,
    oracle_action,
    select_runtime_candidate,
)
from scripts.level1_rollout_server import Level1RolloutStore
from scripts.generate_dca_scenarios import (
    DCA_CANDIDATE_NORMALIZATION_VERSION,
    _candidate_record,
    _canonicalize_candidate_identity,
)
from scripts.merge_dca_candidate_shards import merge_candidate_shards
from scripts.prune_gate_recovery_checkpoint import prune_gate_recovery
from scripts.validate_vda_training_log import parse_training_metrics
from scripts.vda_feedback_server import _generation_messages
from curriculum.reward_function import dca_online_reward
from curriculum.reward_function.vda_reward import score_vda_prediction_v2_fallback


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
    def test_forbidden_nested_action_field_is_invalid_not_server_fatal(self) -> None:
        scenario = minimal_example_v2()
        action = _action()
        action["tool_call"]["metadata"] = {
            "trigger": "continuation_delta",
            "cost_validated": True,
        }

        packet, ok, reason = parse_action_json_v4(json.dumps(action))
        self.assertFalse(ok)
        self.assertEqual(packet, DEFAULT_ACTION_PACKET_V4)
        self.assertIn("forbidden_action_field:$.tool_call.metadata", reason)

        store = Level1RolloutStore()
        result = store.handle(
            {
                "trajectory_ids": ["forbidden-nested-field"],
                "actions": [json.dumps(action)],
                "finish": [True],
                "is_last_step": [False],
                "extra_fields": [{"scenario": scenario}],
            }
        )
        self.assertEqual(result["valids"], [0])
        self.assertIn("forbidden_action_field", result["observations"][0]["invalid_reason"])

    def test_compact_wire_action_normalizes_to_internal_lists(self) -> None:
        wire = {
            "schema_version": 4,
            "belief": copy.deepcopy(DEFAULT_ACTION_PACKET_V4["belief"]),
            "assessment": None,
            "trust_operation": {
                "op": "hold",
                "source_id": "sensor-A",
                "event_id": "event-0",
                "evidence_refs": [],
            },
            "memory_operation": None,
            "memory_use": None,
            "uncertainty": 1.0,
            "tool_call": {"name": "None", "args": {}},
            "safety_check": {
                "business_risk": 0.0,
                "overresponse_risk": 0.0,
                "justification": "observe",
            },
            "response": {"tier": "L0", "action": "Observe", "target": "none"},
        }
        packet, ok, message = parse_action_json_v4(json.dumps(wire))
        self.assertEqual((ok, message), (True, "ok"))
        self.assertEqual(packet["trust_operations"], [wire["trust_operation"]])
        self.assertEqual(packet["memory_operations"], [])
        self.assertNotIn("trust_operation", packet)

    def test_nested_tool_action_must_be_an_object(self) -> None:
        action = _action(
            tool_call={
                "name": "ShadowActionProbe",
                "args": {"action": "LimitSession"},
            }
        )
        packet, ok, message = parse_action_json_v4(json.dumps(action))
        self.assertFalse(ok)
        self.assertEqual(message, "tool_action_not_dict")
        self.assertEqual(packet, DEFAULT_ACTION_PACKET_V4)

    def test_unexpected_belief_field_falls_back_without_crashing_rollout(self) -> None:
        action = _action(
            belief=copy.deepcopy(DEFAULT_ACTION_PACKET_V4["belief"])
            | {"top_objective": "credential_theft"}
        )
        packet, ok, message = parse_action_json_v4(json.dumps(action))
        self.assertFalse(ok)
        self.assertEqual(message, "unexpected_belief_keys")
        self.assertEqual(packet, DEFAULT_ACTION_PACKET_V4)

        scenario = minimal_example_v2()
        result = Level1RolloutStore().handle(
            {
                "trajectory_ids": ["malformed-belief-regression"],
                "actions": [json.dumps(action)],
                "finish": [False],
                "is_last_step": [False],
                "extra_fields": [{"scenario": scenario, "max_env_steps": 2}],
            }
        )
        self.assertEqual(set(result), {"observations", "dones", "valids"})
        self.assertEqual(result["valids"], [0])

    def test_environment_ignores_non_protocol_belief_fields(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=2)
        action = _action(
            belief=copy.deepcopy(DEFAULT_ACTION_PACKET_V4["belief"])
            | {"top_objective": "credential_theft"}
        )
        _observation, tool_result, _done = env.step(action)
        self.assertEqual(tool_result, {"tool": "None"})

    def test_environment_rejects_malformed_nested_tool_action(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=2)
        action = _action(
            tool_call={
                "name": "ShadowActionProbe",
                "args": {"action": "LimitSession"},
            }
        )
        _obs, tool_result, _done = env.step(action)
        self.assertEqual(tool_result["error"], "invalid_action_argument")

    def test_public_projection_is_recursive(self) -> None:
        internal = {
            "outer": [{"truth_value": False, "nested": {"is_fake": True, "spoofability": 0.9}}],
            "oracle": {"true_objective": "exfiltration"},
        }
        public = project_public(internal)
        self.assertEqual(forbidden_public_paths(public), [])
        self.assertEqual(public, {"outer": [{"nested": {}}]})

    def test_vda_prompt_uses_opaque_identity_without_proxy_labels(self) -> None:
        scenario = minimal_example_v2(trajectory_type="legitimate_change")
        prompt = build_vda_prompt(scenario)
        context = json.loads(prompt.split("Current decision instance:", 1)[1])
        self.assertEqual(context["instance_id"], public_instance_id(scenario))
        self.assertNotIn("legitimate_change", context["instance_id"])
        for key in (
            "scenario_id",
            "pair_id",
            "trajectory_type",
            "source_assurance_level",
            "spoofability",
            "is_fake",
        ):
            self.assertFalse(_has_key(context, key), key)

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

    def test_public_evidence_snapshot_is_compact_and_referencable(self) -> None:
        store = EvidenceStore()
        event = {
            "event_id": "e0",
            "source_id": "A",
            "type": "alert",
            "claim": "suspicious transfer",
            "claim_semantics": {
                "entity_id": "db",
                "predicate": "data_transfer",
                "object": "external",
                "scope": "network",
            },
        }
        evidence_id = store.add_event(event, time=0)
        row = store.public_snapshot(time=0)[0]
        self.assertEqual(row["evidence_id"], evidence_id)
        self.assertEqual(row["content"]["claim"], event["claim"])
        self.assertEqual(row["content"]["claim_semantics"], event["claim_semantics"])
        self.assertNotIn("root_source_ids", row)
        self.assertNotIn("claim_keys", row)
        self.assertNotIn("integrity_status", row)
        self.assertNotIn("available_at", row)

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

    def test_noop_does_not_create_tool_evidence(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=2)
        env.observe()
        next_obs, tool_result, _done = env.step(_action())
        self.assertTrue(
            all(row["evidence_type"] != "none_result" for row in next_obs["available_evidence"])
        )
        self.assertEqual(tool_result, {"tool": "None"})

    def test_unknown_event_probe_does_not_create_evidence(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=2)
        env.observe()
        action = _action(
            tool_call={"name": "SourceChallenge", "args": {"event_id": "missing-event"}}
        )
        next_obs, tool_result, _done = env.step(action)
        self.assertEqual(tool_result["error"], "event_not_available_in_current_snapshot")
        self.assertTrue(
            all(row["evidence_type"] != "sourcechallenge_result" for row in next_obs["available_evidence"])
        )

    def test_rollout_continuation_sends_only_new_evidence(self) -> None:
        scenario = minimal_example_v2()
        initial = CyberDefenseEnvV2(scenario, max_steps=3).observe()
        initial_ids = {row["evidence_id"] for row in initial["available_evidence"]}
        result = Level1RolloutStore().handle(
            {
                "trajectory_ids": ["delta-test"],
                "actions": [json.dumps(DEFAULT_ACTION_PACKET_V4)],
                "finish": [False],
                "is_last_step": [False],
                "extra_fields": [{"scenario": scenario, "max_env_steps": 3}],
            }
        )
        payload = json.loads(result["observations"][0]["obs"])
        observation = payload["observation"]
        self.assertEqual(observation["observation_mode"], "continuation_delta")
        self.assertIn("memory", observation["retained_defender_sections"])
        delta_ids = {row["evidence_id"] for row in observation["available_evidence"]}
        self.assertTrue(delta_ids.isdisjoint(initial_ids))

    def test_rollout_emits_periodic_defender_state_checkpoint(self) -> None:
        scenario = minimal_example_v2()
        store = Level1RolloutStore(defender_checkpoint_interval=8)
        state = store._get_or_create_state(
            "checkpoint-test",
            {"scenario": scenario, "max_env_steps": 16},
        )
        next_obs = state.env.observe()
        state.steps = 4
        delta = store._continue_observation(
            state,
            next_obs,
            {"tool": "None"},
            True,
            None,
            False,
        )
        delta_payload = json.loads(delta["obs"])["observation"]
        self.assertEqual(delta_payload["observation_mode"], "continuation_delta")

        state.steps = 8
        checkpoint = store._continue_observation(
            state,
            next_obs,
            {"tool": "None"},
            True,
            None,
            False,
        )
        checkpoint_payload = json.loads(checkpoint["obs"])["observation"]
        self.assertEqual(checkpoint_payload["observation_mode"], "continuation_checkpoint")
        self.assertEqual(
            set(checkpoint_payload["defender_state"]),
            set(next_obs["defender_state"]),
        )

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
            {"event_id": "claim-1", "source_id": "A", "claim_semantics": claim},
            time=0,
            public_prior=0.70,
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

    def test_public_memory_record_avoids_duplicate_claim_fields(self) -> None:
        _store, _trust, memory, _first, _second, memory_id = self._state_fixture()
        row = memory.public_record(memory_id)
        self.assertIn("claim", row)
        self.assertIn("evidence_refs", row)
        self.assertNotIn("claim_key", row)
        self.assertNotIn("entity_id", row)

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

    def test_memory_extremes_are_not_reward_equivalent_to_selective_governance(self) -> None:
        true_claim = {
            "entity_id": "asset-A",
            "predicate": "risk",
            "object": "high",
            "scope": "exfiltration",
        }
        false_claim = {
            "entity_id": "asset-B",
            "predicate": "risk",
            "object": "low",
            "scope": "exfiltration",
        }
        evidence = {
            "ev-true": {"event_id": "event-true", "parent_evidence_ids": []},
            "ev-false": {"event_id": "event-false", "parent_evidence_ids": []},
        }

        class Store:
            def get(self, evidence_id: str):
                return evidence.get(evidence_id)

        def metrics(true_status: str, false_status: str) -> dict[str, float]:
            def record(status: str, ref: str) -> dict[str, object]:
                return {
                    "status": status,
                    "evidence_refs": [ref],
                    "retrieval_count": 5,
                    "acceptance_count": int(status == "confirmed"),
                    "transition_history": (
                        [{"to": "confirmed", "time": 1}]
                        if status == "confirmed"
                        else []
                    ),
                    "created_at": 0,
                }

            env = SimpleNamespace(
                oracle_ledger=[
                    {
                        "event_id": "event-true",
                        "truth_value": True,
                        "claim_semantics": true_claim,
                    },
                    {
                        "event_id": "event-false",
                        "truth_value": False,
                        "claim_semantics": false_claim,
                    },
                ],
                evidence_store=Store(),
                memory=SimpleNamespace(
                    records={
                        "mem-true": record(true_status, "ev-true"),
                        "mem-false": record(false_status, "ev-false"),
                    }
                ),
            )
            return memory_lifecycle_metrics(env)

        all_quarantined = metrics("quarantined", "quarantined")
        selective = metrics("confirmed", "rejected")
        all_confirmed = metrics("confirmed", "confirmed")
        self.assertEqual(all_quarantined["benign_memory_retention"], 0.0)
        self.assertEqual(selective["benign_memory_retention"], 1.0)
        self.assertEqual(selective["poison_promotion_rate"], 0.0)
        self.assertEqual(all_confirmed["poison_promotion_rate"], 1.0)

    def test_memory_action_impact_requires_paired_counterfactual_replay(self) -> None:
        action = _action(response={"tier": "L1", "action": "Observe", "target": "asset-A"})
        changed = copy.deepcopy(action)
        changed["response"] = {"tier": "L2", "action": "LimitSession", "target": "asset-A"}
        result = counterfactual_memory_impact(
            action_with_memory=changed,
            action_without_memory=action,
            utility_with_memory=0.2,
            utility_without_memory=0.8,
        )
        self.assertEqual(result["counterfactual_memory_activation"], 1.0)
        self.assertAlmostEqual(result["counterfactual_memory_downstream_harm"], 0.6)


class TMCDV2DatasetTests(unittest.TestCase):
    def test_training_ablation_registry_is_complete(self) -> None:
        expected = {
            "full",
            "no_dca_training",
            "no_frontier_filtering",
            "no_active_probing",
            "no_passive_verification",
            "append_only_memory",
            "no_trust_recalibration",
            "no_business_aware_reward",
        }
        self.assertEqual(set(TRAINING_VARIANTS), expected)
        self.assertFalse(experiment_variant("no_dca_training").train_dca)
        self.assertFalse(
            experiment_variant("no_frontier_filtering").frontier_filtering
        )

    def test_action_space_ablation_prompts_remove_disabled_tools(self) -> None:
        no_active = vda_system_prompt_v4("no_active_probing")
        no_passive = vda_system_prompt_v4("no_passive_verification")
        for tool in ACTIVE_PROBE_TOOLS:
            self.assertNotIn(tool, no_active)
        for tool in PASSIVE_VERIFICATION_TOOLS:
            self.assertNotIn(tool, no_passive)
        self.assertIn(
            "trust_operation must be null",
            vda_system_prompt_v4("no_trust_recalibration"),
        )

    def test_environment_enforces_disabled_tool_families(self) -> None:
        for variant, tool, error in (
            (
                "no_active_probing",
                "SourceChallenge",
                "active_probing_disabled_by_ablation",
            ),
            (
                "no_passive_verification",
                "CrossCheck",
                "passive_verification_disabled_by_ablation",
            ),
        ):
            scenario = minimal_example_v2()
            scenario.setdefault("metadata", {})["experiment_variant"] = variant
            env = CyberDefenseEnvV2(scenario, max_steps=2)
            observation = env.observe()
            event_id = observation["observed_events"][0]["event_id"]
            _next, result, _done = env.step(
                _action(tool_call={"name": tool, "args": {"event_id": event_id}})
            )
            self.assertEqual(result["error"], error)

    def test_trust_recalibration_ablation_preserves_source_state(self) -> None:
        scenario = minimal_example_v2()
        scenario.setdefault("metadata", {})[
            "experiment_variant"
        ] = "no_trust_recalibration"
        env = CyberDefenseEnvV2(scenario, max_steps=2)
        observation = env.observe()
        event = observation["observed_events"][0]
        evidence_id = event["evidence_id"]
        source_id = event["source_id"]
        before = copy.deepcopy(env.trust_manager.public_snapshot())
        env.step(
            _action(
                trust_operations=[
                    {
                        "op": "contradict",
                        "source_id": source_id,
                        "event_id": event["event_id"],
                        "evidence_refs": [evidence_id],
                    }
                ]
            )
        )
        after = env.trust_manager.public_snapshot()
        self.assertEqual(before["source_reputation"], after["source_reputation"])
        self.assertEqual(
            before["current_claim_trust"][event["event_id"]],
            after["current_claim_trust"][event["event_id"]],
        )
        self.assertEqual(
            env.history[-1]["trust_events"][0]["reason"],
            "trust_recalibration_disabled_by_ablation",
        )

    def test_business_reward_ablation_keeps_metrics_but_removes_penalty(self) -> None:
        scores = {}
        for variant in ("full", "no_business_aware_reward"):
            scenario = minimal_example_v2()
            scenario.setdefault("metadata", {})["experiment_variant"] = variant
            env = CyberDefenseEnvV2(scenario, max_steps=1)
            env.observe()
            env.step(
                _action(
                    belief={
                        "exfiltration": 0.85,
                        "sabotage": 0.05,
                        "persistence": 0.05,
                        "credential_theft": 0.05,
                    },
                    uncertainty=0.8,
                    response={
                        "tier": "L3",
                        "action": "Isolate",
                        "target": "database",
                    },
                )
            )
            scores[variant] = score_trajectory_v2(env)
        self.assertLess(scores["full"]["business_reward_component"], 0.0)
        self.assertEqual(
            scores["no_business_aware_reward"]["business_reward_component"], 0.0
        )
        for metric in ("business_cost", "overresponse", "unauthorized_high_impact"):
            self.assertEqual(scores["full"][metric], scores["no_business_aware_reward"][metric])
        self.assertGreater(
            scores["no_business_aware_reward"]["reward"], scores["full"]["reward"]
        )

    def test_frozen_dca_manifest_advances_lineage_without_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            base_path = root / "dca" / "round_0" / "manifest.json"
            base_path.parent.mkdir(parents=True)
            write_base_manifest(
                base_path,
                role="dca",
                backbone="qwen3.5-4b",
                model_path=str(model),
                seed=7,
            )
            data_manifest = root / "feedback.json"
            data_manifest.write_text('{"parameter_update":false}\n', encoding="utf-8")
            frozen_path = root / "dca" / "round_1" / "manifest.json"
            frozen_path.parent.mkdir(parents=True)
            write_frozen_manifest(
                frozen_path,
                role="dca",
                backbone="qwen3.5-4b",
                round_index=1,
                model_path=str(model),
                seed=7,
                parent_manifest_path=str(base_path),
                training_data_manifest_path=str(data_manifest),
                training_config={"experiment_variant": "no_dca_training"},
            )
            manifest = load_checkpoint_manifest(
                frozen_path,
                role="dca",
                backbone="qwen3.5-4b",
                round_index=1,
            )
            self.assertEqual(manifest["status"], "frozen")
            self.assertFalse(manifest["training_config"]["parameter_update"])
            self.assertIsNone(manifest["adapter_path"])

    def test_partial_gate_dataset_requires_balanced_hard_valid_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"
            manifest_sha = "a" * 64
            with path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "kind": "meta",
                            "config": {"checkpoint_manifest_sha256": manifest_sha},
                        }
                    )
                    + "\n"
                )
                for focus in TASK_FOCI:
                    scenario = task_example_v2(focus)
                    handle.write(
                        json.dumps(
                            {
                                "kind": "record",
                                "record": {
                                    "task_focus": focus,
                                    "parse_ok": True,
                                    "checks": {"all_ok": True},
                                    "scenario": scenario,
                                    "scenario_fingerprint": scenario_fingerprint(scenario),
                                },
                            }
                        )
                        + "\n"
                    )
            records = collect_balanced_records(
                [path], rows_per_task=1, dca_manifest_sha=manifest_sha
            )
            self.assertEqual(len(records), 4)
            with self.assertRaises(ValueError):
                collect_balanced_records(
                    [path], rows_per_task=2, dca_manifest_sha=manifest_sha
                )

    def test_dca_prompt_uses_task_specific_valid_examples(self) -> None:
        self.assertGreaterEqual(DCA_PROMPT_VERSION, 5)
        for focus in TASK_FOCI:
            example = task_example_v2(focus)
            self.assertTrue(full_check(example)["all_ok"], focus)
            messages = build_dca_messages(focus, nonce=7)
            self.assertIn(example["scenario_family"], messages[1]["content"])
            self.assertIn("honest, deceptive, mixed, legitimate_change, recovered", messages[0]["content"])

    def test_formal_split_quotas_reserve_900_per_task(self) -> None:
        quotas = _split_task_quotas({"train": 2400, "dev": 400, "xplay": 800})
        for task_id in ("T1", "T2", "T3", "T4"):
            self.assertEqual(sum(split[task_id] for split in quotas.values()), 900)

    def test_difficulty_stratified_split_is_exact_and_keeps_t2_pairs(self) -> None:
        items = []
        for task_id in ("T1", "T2", "T3", "T4"):
            if task_id == "T2":
                for index in range(5):
                    for branch in ("betrayal", "legitimate_change"):
                        items.append(
                            {
                                "task_id": task_id,
                                "scenario_fingerprint": f"{task_id}-{index}-{branch}",
                                "scenario": {
                                    "pair_id": f"pair-{index}",
                                    "trajectory_type": branch,
                                },
                                "cfc": {"frontier_score": 1.0 - index / 10.0},
                            }
                        )
            else:
                for index in range(10):
                    items.append(
                        {
                            "task_id": task_id,
                            "scenario_fingerprint": f"{task_id}-{index}",
                            "scenario": {},
                            "cfc": {"frontier_score": 1.0 - index / 10.0},
                        }
                    )
        splits = _split_stratified(
            items,
            {"train": 24, "dev": 8, "xplay": 8},
            17,
        )
        expected = {
            "train": {"T1": 6, "T2": 6, "T3": 6, "T4": 6},
            "dev": {"T1": 2, "T2": 2, "T3": 2, "T4": 2},
            "xplay": {"T1": 2, "T2": 2, "T3": 2, "T4": 2},
        }
        pair_splits = {}
        for split, rows in splits.items():
            counts = {}
            for row in rows:
                counts[row["task_id"]] = counts.get(row["task_id"], 0) + 1
                if row["task_id"] == "T2":
                    pair_splits.setdefault(row["scenario"]["pair_id"], set()).add(split)
            self.assertEqual(counts, expected[split])
        self.assertTrue(all(len(values) == 1 for values in pair_splits.values()))

    def test_split_rejects_non_unique_t2_pair_ids(self) -> None:
        items = []
        for task_id in ("T1", "T3", "T4"):
            for index in range(10):
                items.append(
                    {
                        "task_id": task_id,
                        "scenario_fingerprint": f"{task_id}-{index}",
                        "scenario": {},
                        "cfc": {"frontier_score": 0.5},
                    }
                )
        for index in range(5):
            for branch in ("betrayal", "legitimate_change"):
                items.append(
                    {
                        "task_id": "T2",
                        "scenario_fingerprint": f"T2-{index}-{branch}",
                        "scenario": {
                            "pair_id": "reused" if index < 2 else f"pair-{index}",
                            "trajectory_type": branch,
                        },
                        "cfc": {"frontier_score": 0.5},
                    }
                )
        with self.assertRaises(LineageError):
            _split_stratified(items, {"train": 24, "dev": 8, "xplay": 8}, 17)

    def test_candidate_normalization_assigns_unique_t2_pair_ids(self) -> None:
        manifest = {"backbone": "qwen3.5-4b", "round": 1}
        first = minimal_example_v2()
        second = copy.deepcopy(first)
        _canonicalize_candidate_identity(first, index=4, manifest=manifest)
        _canonicalize_candidate_identity(second, index=8, manifest=manifest)
        self.assertNotEqual(first["pair_id"], second["pair_id"])
        self.assertEqual(first["prefix_hash"], public_prefix_hash(first))
        self.assertEqual(
            DCA_CANDIDATE_NORMALIZATION_VERSION,
            3,
        )

    def test_candidate_merge_preserves_generation_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for shard_index in range(2):
                path = Path(directory) / f"shard-{shard_index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "seed": 11,
                            "backbone": "qwen3.5-4b",
                            "source_dca_round": 1,
                            "source_dca_checkpoint_manifest": "/tmp/dca.json",
                            "source_dca_checkpoint_manifest_sha256": "a" * 64,
                            "num_candidates_requested": 2,
                            "num_shards": 2,
                            "shard_index": shard_index,
                            "experiment_variant": "append_only_memory",
                            "generation_prompt_version": 5,
                            "candidate_normalization_version": 2,
                            "tmcd_release_revision": TMCD_RELEASE_REVISION,
                            "max_attempts": 3,
                            "candidates": [
                                {
                                    "candidate_index": shard_index,
                                    "scenario_fingerprint": str(shard_index),
                                    "parse_ok": True,
                                    "checks": {"all_ok": True},
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            merged = merge_candidate_shards(paths, 2)
            self.assertEqual(merged["experiment_variant"], "append_only_memory")
            self.assertEqual(merged["generation_prompt_version"], 5)
            self.assertEqual(merged["candidate_normalization_version"], 2)
            self.assertEqual(merged["tmcd_release_revision"], TMCD_RELEASE_REVISION)
            self.assertEqual(merged["max_attempts"], 3)
            legacy = json.loads(paths[0].read_text(encoding="utf-8"))
            legacy.pop("tmcd_release_revision")
            paths[0].write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaises(ValueError):
                merge_candidate_shards(paths, 2)

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

    def test_noop_and_blocked_isolate_have_expected_effects(self) -> None:
        no_op = CyberDefenseEnvV2(minimal_example_v2(), max_steps=1)
        no_op.observe()
        done = False
        while not done:
            _observation, _tool_result, done = no_op.step(_action())
        no_op_score = score_trajectory_v2(no_op)
        isolate = CyberDefenseEnvV2(minimal_example_v2(), max_steps=1)
        isolate.observe()
        response = {"tier": "L3", "action": "Isolate", "target": "database"}
        belief = {"exfiltration": 0.85, "sabotage": 0.05, "persistence": 0.05, "credential_theft": 0.05}
        isolate.step(_action(response=response, belief=belief, uncertainty=0.8))
        isolate_score = score_trajectory_v2(isolate)
        self.assertTrue(no_op_score["attack_success"])
        self.assertEqual(isolate_score["business_cost"], 0.0)
        self.assertFalse(isolate_score["attack_mitigated"])
        self.assertGreaterEqual(isolate_score["unauthorized_high_impact"], 1)

    def test_tools_never_default_to_hidden_attack_target(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=2)
        env.observe()
        _obs, tool_result, _done = env.step(
            _action(tool_call={"name": "GraphQuery", "args": {}})
        )
        self.assertEqual(tool_result["error"], "missing_required_arg:node")

    def test_wrong_response_target_does_not_mitigate_hidden_attack(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=1)
        env.observe()
        response = {"tier": "L3", "action": "Isolate", "target": "user_pc"}
        belief = {
            "exfiltration": 0.85,
            "sabotage": 0.05,
            "persistence": 0.05,
            "credential_theft": 0.05,
        }
        env.step(_action(response=response, belief=belief, uncertainty=0.1))
        self.assertFalse(env.attack_mitigated)

    def test_oracle_defender_uses_the_true_asset_target(self) -> None:
        scenario = minimal_example_v2()
        env = CyberDefenseEnvV2(scenario, max_steps=2)
        public_context = env.observe()
        action = oracle_action(public_context, scenario, turn=1)
        self.assertEqual(
            action["response"]["target"],
            scenario["true_attack"]["target"],
        )


class TMCDV2ReleaseTests(unittest.TestCase):
    def test_gate_recovery_pruning_preserves_stable_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            step = root / "checkpoints" / "global_step_1"
            step.mkdir(parents=True)
            (step / "optimizer.pt").write_bytes(b"recovery")
            manifest = root / "checkpoint_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "checkpoint",
                        "role": "vda",
                        "backbone": "qwen3.5-4b",
                        "round": 1,
                        "checkpoint_path": str(step),
                        "adapter_path": str(adapter),
                        "adapter_sha256": sha256_tree(adapter),
                    }
                ),
                encoding="utf-8",
            )
            report = prune_gate_recovery(manifest)
            self.assertFalse(step.exists())
            self.assertTrue(adapter.exists())
            self.assertGreater(report["removed_bytes"], 0)

    def test_vda_gate_requires_trajectory_reward_and_real_optimizer_update(self) -> None:
        line = " - ".join(
            (
                "step:1",
                "reward_extra_info/level1_trajectory_reward_available:1.0",
                "reward_extra_info/single_step_reward_fallback:0.0",
                "actor/grad_norm:0.05",
                "actor/lr:2e-5",
                "env/ratio_of_valid_action:0.9",
                "env/action_length/max:257.0",
                "env/obs_length/max:679.0",
                "perf/max_memory_allocated_gb:17.3",
                "perf/max_memory_reserved_gb:26.0",
                "timing_s/step:468.0",
                "training/global_step:1",
            )
        )
        report = parse_training_metrics(
            line,
            expected_step=1,
            action_budget=320,
            observation_budget=1280,
        )
        self.assertEqual(report["trajectory_reward_available"], 1.0)
        terminal_token_report = parse_training_metrics(
            line.replace("env/action_length/max:257.0", "env/action_length/max:322.0"),
            expected_step=1,
            action_budget=320,
            observation_budget=1280,
        )
        self.assertEqual(terminal_token_report["max_action_tokens"], 322.0)
        with self.assertRaises(ValueError):
            parse_training_metrics(
                line.replace("env/action_length/max:257.0", "env/action_length/max:323.0"),
                expected_step=1,
                action_budget=320,
                observation_budget=1280,
            )
        with self.assertRaises(ValueError):
            parse_training_metrics(
                line.replace("actor/lr:2e-5", "actor/lr:0.0"),
                expected_step=1,
                action_budget=320,
                observation_budget=1280,
            )

    def test_source_tree_hash_ignores_runtime_cache_but_tracks_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "module.py"
            source.write_text("value = 1\n", encoding="utf-8")
            first = sha256_source_tree(root)
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "module.cpython-312.pyc").write_bytes(b"runtime cache")
            self.assertEqual(first, sha256_source_tree(root))
            source.write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(first, sha256_source_tree(root))

    def test_training_feedback_does_not_call_v5c(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "vda_feedback_server.py").read_text(encoding="utf-8")
        self.assertNotIn("select_candidate(", source)
        self.assertIn('"v5c_used": False', source)

    def test_dca_reward_normalizes_untrusted_metadata_before_hard_check(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "curriculum" / "reward_function" / "dca_online_reward.py"
        ).read_text(encoding="utf-8")
        self.assertIn('_as_dict(scenario.get("metadata"))', source)

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
        self.assertIn('"source_trees"', prepare_source)
        self.assertIn("source tree hash mismatch", preflight_source)
        self.assertIn('f"{report_path.stem}.protocol_smoke.json"', preflight_source)

    def test_v24_formal_jobs_are_node_pinned_and_refuse_overwrite(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = {
            "tmcd_v24_4b_full_node175.dsub.sh": ("cyclone001-agent-175", "full"),
            "tmcd_v24_9b_full_node217.dsub.sh": ("cyclone001-agent-217", "full"),
            "tmcd_v24_4b_append_only_node208.dsub.sh": (
                "cyclone001-agent-208",
                "append_only_memory",
            ),
        }
        for name, (node, variant) in expected.items():
            source = (root / "scripts" / "jobs" / name).read_text(encoding="utf-8")
            self.assertIn(f"#DSUB -pn {node}", source)
            self.assertIn('artifact-scope tmcd_v24', source)
            self.assertIn(f"--experiment-variant {variant}", source)
            self.assertIn("Refusing to overwrite formal TMCD v2.4 outputs", source)
            self.assertIn('AGZ_FORMAL_RESUME:-0', source)
            self.assertIn("--dca-feedback-candidates 4000", source)
            self.assertIn("--vda-candidates 10000", source)
            self.assertIn("--vda-train-size 2400", source)
            self.assertIn("--vda-dev-size 400", source)
            self.assertIn("--vda-xplay-size 800", source)
            self.assertIn("AGZ_REQUIRE_TRAJECTORY_REWARD=1", source)
            self.assertNotIn("select_only", source)

    def test_round_runner_invalidates_downstream_pool_and_can_stop_before_vda(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "run_dca_first_round.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--stop-after-stage"', source)
        self.assertIn('args.stop_after_stage == stage', source)
        self.assertIn(
            '"build_isolated_vda_pool",\n                    "stale"',
            source,
        )
        self.assertIn(
            "candidate generation signature changed after VDA recovery checkpoints",
            source,
        )
        dca_env = source.index('"AGZ_REQUIRE_TRAJECTORY_REWARD": "0"')
        dca_launch = source.index('train_dca_qwen35_lora.sh')
        vda_env = source.index('"AGZ_REQUIRE_TRAJECTORY_REWARD": "1"')
        vda_launch = source.index('train_vda_qwen35_lora.sh')
        self.assertLess(dca_env, dca_launch)
        self.assertLess(dca_launch, vda_env)
        self.assertLess(vda_env, vda_launch)


class TMCDV2TeacherReviewTests(unittest.TestCase):
    @staticmethod
    def _add_positive_support_evidence(
        env: CyberDefenseEnvV2,
        event: dict,
    ) -> str:
        return env.evidence_store.add_tool_result(
            {
                "tool": "CrossCheck",
                "event_id": event["event_id"],
                "verdict": "supported",
                "consistency": 1.0,
                "claim_semantics": copy.deepcopy(event["claim_semantics"]),
                "root_source_ids": ["independent-verifier"],
            },
            time=-1,
            parent_evidence_ids=[event["evidence_id"]],
        )

    @staticmethod
    def _authorize_with_semantics(
        semantics: dict,
        *,
        oracle_mode: bool = False,
    ) -> tuple[CyberDefenseEnvV2, dict]:
        scenario = minimal_example_v2()
        scenario["event_schedule"][0]["claim_semantics"] = copy.deepcopy(semantics)
        scenario["event_schedule"][0]["entity_id"] = semantics["entity_id"]
        env = CyberDefenseEnvV2(scenario, oracle_mode=oracle_mode)
        observation = env.observe()
        event = observation["observed_events"][0]
        support_evidence = TMCDV2TeacherReviewTests._add_positive_support_evidence(
            env, event
        )
        claim = env.trust_manager.claim_trust[event["event_id"]]
        claim["status"] = "supported"
        claim["score"] = 0.90
        claim["support_evidence_refs"] = [support_evidence]
        env.step(
            _action(
                evidence_assessment=[
                    {"event_id": event["event_id"], "status": "supported"}
                ],
                response={"tier": "L3", "action": "Isolate", "target": "database"},
            )
        )
        return env, env.history[-1]["response_result"]

    def test_supported_low_risk_claim_cannot_authorize_isolate(self) -> None:
        _env, result = self._authorize_with_semantics(
            {
                "entity_id": "database",
                "predicate": "asset_risk_level",
                "object": "low",
                "scope": "cyber_defense",
            }
        )
        self.assertTrue(result["blocked"])
        self.assertEqual(result["executed_action"], "Observe")

    def test_supported_high_risk_claim_can_authorize_isolate(self) -> None:
        _env, result = self._authorize_with_semantics(
            {
                "entity_id": "database",
                "predicate": "asset_risk_level",
                "object": "critical",
                "scope": "cyber_defense",
            }
        )
        self.assertFalse(result["blocked"])
        self.assertEqual(result["executed_action"], "Isolate")

    def test_scenario_metadata_cannot_enable_oracle_override(self) -> None:
        scenario = minimal_example_v2()
        scenario.setdefault("metadata", {})["oracle_defender"] = True
        valid, reason = validate_scenario_v2(scenario)
        self.assertFalse(valid)
        self.assertEqual(reason, "privileged_metadata_forbidden:oracle_defender")
        with self.assertRaisesRegex(ValueError, "privileged_metadata_forbidden"):
            CyberDefenseEnvV2(scenario)

    def test_oracle_override_requires_explicit_runtime_flag(self) -> None:
        response = {"tier": "L3", "action": "Isolate", "target": "database"}
        normal = CyberDefenseEnvV2(minimal_example_v2())
        normal.observe()
        normal.step(_action(response=response))
        self.assertTrue(normal.history[-1]["response_result"]["blocked"])

        oracle = CyberDefenseEnvV2(minimal_example_v2(), oracle_mode=True)
        oracle.observe()
        oracle.step(_action(response=response))
        self.assertFalse(oracle.history[-1]["response_result"]["blocked"])

    def test_generated_candidate_strips_privileged_metadata(self) -> None:
        scenario = task_example_v2(TASK_FOCI[0])
        scenario.setdefault("metadata", {}).update(
            {"oracle_defender": True, "oracle_mode": True, "privileged": True}
        )
        record = _candidate_record(
            raw_output=json.dumps(scenario),
            index=0,
            focus=TASK_FOCI[0],
            nonce=7,
            attempt=1,
            args=SimpleNamespace(
                seed=7,
                shard_index=0,
                num_shards=1,
                experiment_variant="full",
            ),
            manifest={"adapter_path": None, "round": 1, "backbone": "qwen3.5-4b"},
            manifest_path=Path("manifest.json"),
            manifest_sha="a" * 64,
        )
        self.assertTrue(record["checks"]["all_ok"])
        metadata = record["scenario"]["metadata"]
        self.assertFalse(
            {"oracle_defender", "oracle_mode", "privileged"} & set(metadata)
        )

    def test_task_focus_family_mismatch_is_rejected(self) -> None:
        scenario = task_example_v2(TASK_FOCI[2])
        record = _candidate_record(
            raw_output=json.dumps(scenario),
            index=0,
            focus=TASK_FOCI[0],
            nonce=7,
            attempt=1,
            args=SimpleNamespace(
                seed=7,
                shard_index=0,
                num_shards=1,
                experiment_variant="full",
            ),
            manifest={"adapter_path": None, "round": 1, "backbone": "qwen3.5-4b"},
            manifest_path=Path("manifest.json"),
            manifest_sha="a" * 64,
        )
        self.assertFalse(record["checks"]["all_ok"])
        self.assertIn(
            "task_family_mismatch:T1:profile_poisoning",
            record["checks"]["valid"]["message"],
        )

    def test_task_focus_family_mapping_is_exact(self) -> None:
        for focus in TASK_FOCI:
            scenario = task_example_v2(focus)
            record = {"task_focus": focus}
            self.assertEqual(_task_id(record, scenario), focus.split()[0])
            self.assertEqual(
                scenario["scenario_family"], TASK_FAMILY_MAP[focus.split()[0]]
            )
        mismatch = task_example_v2(TASK_FOCI[2])
        self.assertEqual(_task_id({"task_focus": TASK_FOCI[0]}, mismatch), "unknown")

    def test_only_positive_support_refs_enter_authorization_evidence(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2())
        observation = env.observe()
        event = observation["observed_events"][0]
        parent = event["evidence_id"]
        positive = env.evidence_store.add_tool_result(
            {
                "tool": "CrossCheck",
                "event_id": event["event_id"],
                "verdict": "supported",
                "consistency": 1.0,
                "claim_semantics": copy.deepcopy(event["claim_semantics"]),
                "root_source_ids": ["sensor-B"],
            },
            time=-1,
            parent_evidence_ids=[parent],
        )
        negative_probe = env.evidence_store.add_tool_result(
            {
                "tool": "SourceChallenge",
                "event_id": event["event_id"],
                "probe_id": "probe-negative",
                "verdict": "challenge_failed",
                "contradiction_risk": 0.2,
                "claim_semantics": copy.deepcopy(event["claim_semantics"]),
                "root_source_ids": ["sensor-B"],
            },
            time=-1,
            parent_evidence_ids=[parent],
        )
        committed = env.trust_manager.apply(
            [
                {
                    "op": "support",
                    "source_id": event["source_id"],
                    "event_id": event["event_id"],
                    "evidence_refs": [positive, negative_probe],
                }
            ],
            evidence_store=env.evidence_store,
            time=0,
        )[0]
        self.assertTrue(committed["committed"])
        claim = env.trust_manager.claim_for(event["event_id"])
        self.assertEqual(claim["support_evidence_refs"], [positive])
        self.assertNotIn(negative_probe, claim["support_evidence_refs"])
        env.step(
            _action(
                evidence_assessment=[
                    {"event_id": event["event_id"], "status": "supported"}
                ],
                response={"tier": "L3", "action": "Isolate", "target": "database"},
            )
        )
        refs = env.history[-1]["response_result"]["authorization_evidence_ids"]
        self.assertIn(positive, refs)
        self.assertNotIn(negative_probe, refs)

    def test_multi_event_same_time_oracle_ledger_records_every_event(self) -> None:
        scenario = minimal_example_v2()
        second = copy.deepcopy(scenario["event_schedule"][-1])
        second["event_id"] = "event-same-time-b"
        second["time"] = 0
        scenario["event_schedule"].append(second)
        env = CyberDefenseEnvV2(scenario)
        observation = env.observe()
        public_ids = {event["event_id"] for event in observation["observed_events"]}
        ledger_ids = {event["event_id"] for event in env.oracle_ledger.snapshot()}
        self.assertIn("event-prefix-0", public_ids)
        self.assertIn("event-same-time-b", public_ids)
        self.assertTrue({"event-prefix-0", "event-same-time-b"}.issubset(ledger_ids))

    def test_crosscheck_roots_are_derived_from_evidence_not_source_names(self) -> None:
        scenario = minimal_example_v2()
        second = copy.deepcopy(scenario["event_schedule"][-1])
        second["event_id"] = "event-same-claim-b"
        second["time"] = 0
        scenario["event_schedule"].append(second)

        forged = CyberDefenseEnvV2(copy.deepcopy(scenario))
        observation = forged.observe()
        first = next(event for event in observation["observed_events"] if event["event_id"] == "event-prefix-0")
        forged.step(
            _action(
                tool_call={
                    "name": "CrossCheck",
                    "args": {"event_id": first["event_id"], "sources": ["sensor-B"]},
                }
            )
        )
        self.assertEqual(
            forged.last_tool_result["error"],
            "crosscheck_requires_evidence_ids",
        )

        valid_env = CyberDefenseEnvV2(copy.deepcopy(scenario))
        observation = valid_env.observe()
        first = next(event for event in observation["observed_events"] if event["event_id"] == "event-prefix-0")
        second_public = next(
            event for event in observation["observed_events"] if event["event_id"] == "event-same-claim-b"
        )
        valid_env.step(
            _action(
                tool_call={
                    "name": "CrossCheck",
                    "args": {
                        "event_id": first["event_id"],
                        "evidence_ids": [second_public["evidence_id"]],
                        "sources": ["fabricated-source"],
                    },
                }
            )
        )
        tool_evidence = valid_env.evidence_store.get(
            valid_env.last_tool_result["evidence_id"],
            time=1,
        )
        self.assertEqual(set(tool_evidence["root_source_ids"]), {"sensor-A", "sensor-B"})
        self.assertNotIn("fabricated-source", tool_evidence["root_source_ids"])

    def test_train_and_full_share_pre_state_response_authorization(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2())
        observation = env.observe()
        event = observation["observed_events"][0]
        parent = event["evidence_id"]
        support = env.evidence_store.add_tool_result(
            {
                "tool": "CrossCheck",
                "event_id": event["event_id"],
                "verdict": "supported",
                "claim_semantics": copy.deepcopy(event["claim_semantics"]),
                "root_source_ids": ["sensor-B"],
            },
            time=-1,
            parent_evidence_ids=[parent],
        )
        packet = _action(
            evidence_assessment=[
                {"event_id": event["event_id"], "status": "supported", "suspected_poisoning": False}
            ],
            trust_operations=[
                {
                    "op": "support",
                    "source_id": "sensor-A",
                    "event_id": event["event_id"],
                    "evidence_refs": [support],
                }
            ],
            response={"tier": "L3", "action": "Isolate", "target": "database"},
            belief={
                "exfiltration": 0.85,
                "sabotage": 0.05,
                "persistence": 0.05,
                "credential_theft": 0.05,
            },
        )
        scored = score_v5c_candidate({"observation": observation}, json.dumps(packet))
        self.assertFalse(scored.admissible)
        env.step(packet)
        self.assertTrue(env.history[-1]["response_result"]["blocked"])
        self.assertTrue(env.history[-1]["trust_events"][0]["committed"])
        self.assertEqual(env.trust_manager.claim_for(event["event_id"])["status"], "supported")

    def test_safe_probe_fallback_observes_only_when_budget_exhausted(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2())
        context = {"observation": env.observe()}
        selected, diagnostics = select_v5c(context, ["{}"])
        self.assertEqual(selected["tool_call"]["name"], "SourceChallenge")
        self.assertEqual(diagnostics["fallback_type"], "SourceChallenge")

        exhausted = copy.deepcopy(context)
        exhausted["observation"]["defense_context"]["remaining_verification_budget"] = 0.0
        selected, diagnostics = select_v5c(exhausted, ["{}"])
        self.assertEqual(selected["tool_call"]["name"], "None")
        self.assertEqual(diagnostics["fallback_type"], "Observe")

    def test_global_trust_decay_is_time_advanced_and_idempotent(self) -> None:
        manager = ContextualTrustManager(decay=0.8)
        source = manager.ensure_source("sensor-A", public_prior=0.5, time=0)
        source["alpha"] = 8.0
        source["beta"] = 2.0
        manager._refresh_source(source)
        before = source["mean"]
        manager.advance_time(5)
        after = source["mean"]
        manager.advance_time(5)
        self.assertLess(abs(after - 0.5), abs(before - 0.5))
        self.assertEqual(source["mean"], after)

    def test_duplicate_ingest_noops_and_new_evidence_merges(self) -> None:
        store = EvidenceStore()
        claim = {
            "entity_id": "database",
            "predicate": "attack_objective",
            "object": "exfiltration",
            "scope": "cyber_defense",
        }
        first = store.add_event(
            {
                "event_id": "event-a",
                "source_id": "sensor-A",
                "claim_semantics": claim,
            },
            time=0,
        )
        second = store.add_event(
            {
                "event_id": "event-b",
                "source_id": "sensor-B",
                "claim_semantics": claim,
            },
            time=0,
        )
        memory = EvidenceStateMemory()
        trust = ContextualTrustManager()

        def ingest(ref: str, source: str) -> dict:
            return memory.apply(
                [
                    {
                        "op": "ingest",
                        "claim": claim,
                        "source_ids": [source],
                        "evidence_refs": [ref],
                        "target_status": "quarantined",
                    }
                ],
                evidence_store=store,
                trust_manager=trust,
                time=0,
            )[0]

        created = ingest(first, "sensor-A")
        duplicate = ingest(first, "sensor-A")
        merged = ingest(second, "sensor-B")
        record = memory.records[created["memory_id"]]
        self.assertFalse(duplicate["committed"])
        self.assertEqual(duplicate["reason"], "duplicate_ingest")
        self.assertTrue(merged["committed"])
        self.assertEqual(merged["reason"], "evidence_merged")
        self.assertEqual(record["version"], 2)
        self.assertEqual(set(record["source_ids"]), {"sensor-A", "sensor-B"})

    def test_train_k_selector_controls_are_registered_and_reproducible(self) -> None:
        controls = {
            "agentguard_zero_train_random_k",
            "agentguard_zero_train_mitigation_best_of_k",
            "agentguard_zero_train_soft_v5c",
        }
        self.assertTrue(controls.issubset(set(SYSTEMS)))
        self.assertTrue(all(default_candidate_count(system, 4) == 4 for system in controls))

        env = CyberDefenseEnvV2(minimal_example_v2())
        context = {"observation": env.observe()}
        candidates = [json.dumps(_action()), json.dumps(_action(rationale="second"))]
        first = select_runtime_candidate(
            "agentguard_zero_train_random_k",
            context,
            candidates,
            "agentguard_zero_select",
            selector_mode="v5_c_evidence_governor",
            seed=17,
        )
        second = select_runtime_candidate(
            "agentguard_zero_train_random_k",
            context,
            candidates,
            "agentguard_zero_select",
            selector_mode="v5_c_evidence_governor",
            seed=17,
        )
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.diagnostics["selector"], "random_k")

    def test_v5c_selected_packet_is_the_packet_executed_by_environment(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2())
        observation = env.observe()
        event_id = observation["observed_events"][0]["event_id"]
        blocked = _action(
            evidence_assessment=[{"event_id": event_id, "status": "unverified"}],
            response={"tier": "L3", "action": "Isolate", "target": "database"},
        )
        probe = _action(
            tool_call={"name": "SourceChallenge", "args": {"event_id": event_id}},
            response={"tier": "L0", "action": "Observe", "target": "none"},
        )
        selected = select_runtime_candidate(
            "agentguard_zero_select",
            {"observation": observation},
            [json.dumps(blocked), json.dumps(probe)],
            "agentguard_zero_select",
            selector_mode="v5_c_evidence_governor",
        )
        self.assertEqual(selected.packet, probe)
        env.step(selected.packet)
        self.assertEqual(env.history[-1]["action_packet"], selected.packet)

    def test_authorization_precedes_hidden_world_effect(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2())
        env.observe()
        before = env.hidden_world.clone_snapshot()
        env.step(
            _action(
                response={"tier": "L3", "action": "Isolate", "target": "database"},
                belief={
                    "exfiltration": 0.85,
                    "sabotage": 0.05,
                    "persistence": 0.05,
                    "credential_theft": 0.05,
                },
            )
        )
        result = env.history[-1]["response_result"]
        self.assertTrue(result["blocked"])
        self.assertEqual(result["executed_action"], "Observe")
        self.assertEqual(env.business_cost, 0.0)
        self.assertEqual(env.high_impact_count, 0)
        self.assertEqual(env.hidden_world.attack_pressure, before["attack_pressure"])

    def test_supported_high_impact_action_can_execute(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2())
        observation = env.observe()
        event = observation["observed_events"][0]
        event_id = event["event_id"]
        support_evidence = self._add_positive_support_evidence(env, event)
        env.trust_manager.claim_trust[event_id]["status"] = "supported"
        env.trust_manager.claim_trust[event_id]["score"] = 0.90
        env.trust_manager.claim_trust[event_id]["support_evidence_refs"] = [
            support_evidence
        ]
        env.step(
            _action(
                evidence_assessment=[{"event_id": event_id, "status": "supported"}],
                belief={
                    "exfiltration": 0.85,
                    "sabotage": 0.05,
                    "persistence": 0.05,
                    "credential_theft": 0.05,
                },
                response={"tier": "L3", "action": "Isolate", "target": "database"},
            )
        )
        result = env.history[-1]["response_result"]
        self.assertFalse(result["blocked"])
        self.assertEqual(result["executed_action"], "Isolate")
        self.assertGreater(env.business_cost, 0.0)
        self.assertLess(env.hidden_world.attack_pressure, 1.0)

    def test_verification_budget_is_enforced_before_tool_execution(self) -> None:
        scenario = minimal_example_v2()
        scenario["defense_constraints"]["verification_budget"] = 1.0
        env = CyberDefenseEnvV2(scenario)
        observation = env.observe()
        event_id = observation["observed_events"][0]["event_id"]
        _next, result, _done = env.step(
            _action(tool_call={"name": "CanaryProbe", "args": {"event_id": event_id}})
        )
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["executed"])
        self.assertEqual(env.verification_cost, 0.0)

    def test_decoy_probe_changes_the_next_public_observation(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2())
        env.observe()
        next_observation, result, _done = env.step(
            _action(tool_call={"name": "DecoyProbe", "args": {"zone": "server"}})
        )
        self.assertTrue(result["deployed"])
        self.assertTrue(
            any(event.get("type") == "decoy_probe_result" for event in next_observation["observed_events"])
        )
        probe_state = next_observation["defender_state"]["probe_state"]
        self.assertEqual(probe_state[0]["status"], "resolved")
        self.assertIn(probe_state[0]["result"], {"interaction_observed", "no_interaction_observed"})

    def test_active_probe_is_qualitative_and_not_a_perfect_truth_oracle(self) -> None:
        failures = 0
        for index in range(48):
            scenario = minimal_example_v2()
            scenario["scenario_id"] = f"noise-check-{index}"
            env = CyberDefenseEnvV2(scenario)
            observation = env.observe()
            event_id = observation["observed_events"][0]["event_id"]
            _next, result, _done = env.step(
                _action(tool_call={"name": "SourceChallenge", "args": {"event_id": event_id}})
            )
            failures += int(result["verdict"] == "challenge_failed")
            for forbidden in ("truth_value", "spoofability", "contradiction_risk", "challenge_consistency"):
                self.assertNotIn(forbidden, result)
        self.assertGreater(failures, 0)
        self.assertLess(failures, 48)

    def test_trust_evidence_is_idempotent(self) -> None:
        event = {
            "event_id": "event-a",
            "source_id": "sensor-a",
            "type": "alert",
            "claim_semantics": {
                "entity_id": "database",
                "predicate": "objective",
                "object": "exfiltration",
                "scope": "cyber_defense",
            },
        }
        store = EvidenceStore()
        parent = store.add_event(event, time=0)
        evidence = store.add_tool_result(
            {"tool": "SourceChallenge", "event_id": "event-a", "verdict": "challenge_passed"},
            time=0,
            parent_evidence_ids=[parent],
        )
        manager = ContextualTrustManager()
        manager.register_claim(event, time=0)
        operation = {
            "op": "support",
            "source_id": "sensor-a",
            "event_id": "event-a",
            "evidence_refs": [evidence],
        }
        first = manager.apply([operation], evidence_store=store, time=1)[0]
        alpha = manager.source_reputation["sensor-a"]["alpha"]
        second = manager.apply([operation], evidence_store=store, time=1)[0]
        self.assertTrue(first["committed"])
        self.assertFalse(second["committed"])
        self.assertEqual(second["reason"], "duplicate_trust_evidence")
        self.assertEqual(manager.source_reputation["sensor-a"]["alpha"], alpha)

    def test_independence_uses_maximum_disjoint_root_set(self) -> None:
        store = EvidenceStore()
        first = store.add_event({"event_id": "a", "source_id": "a", "type": "alert"}, time=0)
        second = store.add_event({"event_id": "b", "source_id": "b", "type": "alert"}, time=0)
        combined = store.add_tool_result(
            {"tool": "CrossCheck", "root_source_ids": ["a", "b"]},
            time=0,
        )
        self.assertEqual(store.independent_count([combined, first, second], time=1), 2)
        self.assertEqual(store.independent_count([second, combined, first], time=1), 2)

    def test_memory_acceptance_counts_only_support_for_decisions(self) -> None:
        memory = EvidenceStateMemory()
        memory.records["mem-a"] = {
            "memory_id": "mem-a",
            "status": "quarantined",
            "claim": {},
            "source_ids": [],
            "acceptance_count": 0,
        }
        memory.record_usage(
            [
                {"memory_id": "mem-a", "usage": "contradict", "used_for": "belief"},
                {"memory_id": "mem-a", "usage": "background", "used_for": "response"},
                {"memory_id": "mem-a", "usage": "support", "used_for": "tool"},
                {"memory_id": "mem-a", "usage": "support", "used_for": "response"},
            ],
            retrieved_ids={"mem-a"},
            time=1,
        )
        record = memory.records["mem-a"]
        self.assertEqual(record["acceptance_count"], 1)
        self.assertEqual(record["usage_counts"]["contradict"], 1)
        self.assertEqual(record["usage_counts"]["background"], 1)
        self.assertEqual(record["usage_counts"]["support"], 2)

    def test_memory_sources_are_derived_from_evidence_lineage(self) -> None:
        event = {
            "event_id": "event-a",
            "source_id": "sensor-a",
            "type": "alert",
            "claim_semantics": {
                "entity_id": "database",
                "predicate": "objective",
                "object": "exfiltration",
                "scope": "cyber_defense",
            },
        }
        store = EvidenceStore()
        evidence = store.add_event(event, time=0)
        memory = EvidenceStateMemory()
        result = memory.apply(
            [
                {
                    "op": "ingest",
                    "claim": event["claim_semantics"],
                    "source_ids": ["forged-source"],
                    "evidence_refs": [evidence],
                }
            ],
            evidence_store=store,
            trust_manager=ContextualTrustManager(),
            time=0,
        )[0]
        self.assertFalse(result["committed"])
        self.assertEqual(result["reason"], "source_lineage_mismatch")
        self.assertEqual(result["derived_source_ids"], ["sensor-a"])

    def test_irrelevant_memory_is_not_retrieved_by_recency_alone(self) -> None:
        memory = EvidenceStateMemory()
        memory.records["mem-unrelated"] = {
            "memory_id": "mem-unrelated",
            "status": "confirmed",
            "claim": {
                "entity_id": "printer",
                "predicate": "maintenance",
                "object": "complete",
                "scope": "facilities",
            },
            "source_ids": ["facilities"],
            "updated_at": 10,
            "retrieval_count": 0,
        }
        result = retrieve_memory(
            memory,
            [
                {
                    "entity_id": "database",
                    "source_id": "sensor-a",
                    "type": "alert",
                    "objective_hint": "exfiltration",
                    "claim": "database collection activity",
                }
            ],
            time=10,
        )
        self.assertEqual(result["retrieved_confirmed"], [])
        self.assertEqual(result["retrieved_memory_ids"], [])

    def test_belief_is_normalized_and_operation_lists_are_bounded(self) -> None:
        packet = _action(
            belief={
                "exfiltration": 2.0,
                "sabotage": 1.0,
                "persistence": 1.0,
                "credential_theft": 0.0,
            }
        )
        normalized, ok, _message = parse_action_json_v4(json.dumps(packet))
        self.assertTrue(ok)
        self.assertAlmostEqual(sum(normalized["belief"].values()), 1.0)
        packet["trust_operations"] = [
            {"op": "hold", "source_id": f"s-{index}", "event_id": "", "evidence_refs": []}
            for index in range(5)
        ]
        _parsed, ok, message = parse_action_json_v4(json.dumps(packet))
        self.assertFalse(ok)
        self.assertEqual(message, "trust_operations_not_list")

    def test_v5c_hard_gate_rejects_unknown_evidence_and_ignores_self_reported_risk(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2())
        context = {"observation": env.observe()}
        low_report = _action(
            uncertainty=0.0,
            safety_check={"business_risk": 0.0, "overresponse_risk": 0.0, "justification": "safe"},
        )
        high_report = _action(
            uncertainty=1.0,
            safety_check={"business_risk": 1.0, "overresponse_risk": 1.0, "justification": "risky"},
        )
        first = score_v5c_candidate(context, json.dumps(low_report))
        second = score_v5c_candidate(context, json.dumps(high_report))
        self.assertEqual(first.score, second.score)
        invalid = _action(
            trust_operations=[
                {
                    "op": "support",
                    "source_id": "sensor-A",
                    "event_id": context["observation"]["observed_events"][0]["event_id"],
                    "evidence_refs": ["missing-evidence"],
                }
            ]
        )
        selected, diagnostics = select_v5c(context, [json.dumps(invalid)])
        self.assertEqual(selected["tool_call"]["name"], "SourceChallenge")
        self.assertEqual(diagnostics["fallback"], "safe_active_probe")

    def test_decision_horizon_covers_late_events(self) -> None:
        scenario = minimal_example_v2()
        scenario["event_schedule"][-1]["time"] = 12
        env = CyberDefenseEnvV2(scenario, max_steps=4)
        self.assertGreaterEqual(env.max_steps, 14)

    def test_multi_event_adapter_targets_the_referenced_event(self) -> None:
        scenario = minimal_example_v2()
        scenario["event_schedule"][1]["time"] = 0
        observation = CyberDefenseEnvV2(scenario).observe()
        second = observation["observed_events"][1]
        legacy = {
            "belief": copy.deepcopy(DEFAULT_ACTION_PACKET_V4["belief"]),
            "evidence_assessment": [{"status": "unverified"}],
            "memory_update": {"profile_quarantine": [], "confirmed_profile": [], "rejected_profile": []},
            "tool_call": {"name": "SourceChallenge", "args": {"event_id": second["event_id"]}},
            "uncertainty": 0.5,
            "safety_check": {"business_risk": 0.0, "overresponse_risk": 0.0, "justification": ""},
            "response": {"tier": "L0", "action": "Observe", "target": "none"},
        }
        adapted = adapt_packet_v2(legacy, {"observation": observation})
        self.assertEqual(adapted["trust_operations"][0]["event_id"], second["event_id"])
        self.assertEqual(
            {item["event_id"] for item in adapted["evidence_assessment"]},
            {item["event_id"] for item in observation["observed_events"]},
        )

    def test_full_has_one_canonical_system_id(self) -> None:
        self.assertIn("agentguard_zero_full", SYSTEMS)
        self.assertNotIn("agentguard_zero_train_select", SYSTEMS)
        self.assertEqual(
            SYSTEM_ALIASES["agentguard_zero_train_select"],
            "agentguard_zero_full",
        )


class TMCDV2PerformanceReviewTests(unittest.TestCase):
    def test_high_impact_requires_target_relevant_claim(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2())
        observation = env.observe()
        event_id = observation["observed_events"][0]["event_id"]
        env.trust_manager.claim_trust[event_id]["status"] = "supported"
        env.trust_manager.claim_trust[event_id]["score"] = 0.90
        env._observation_cache.clear()
        public_context = {"observation": env.observe()}
        packet = _action(
            evidence_assessment=[{"event_id": event_id, "status": "supported"}],
            response={"tier": "L3", "action": "Isolate", "target": "file_server"},
        )
        scored = score_v5c_candidate(public_context, json.dumps(packet))
        self.assertFalse(scored.admissible)
        self.assertIn(
            "high_impact_requires_target_relevant_supported_public_evidence",
            scored.hard_violations,
        )
        env.step(packet)
        result = env.history[-1]["response_result"]
        self.assertTrue(result["blocked"])
        self.assertEqual(result["authorization_reason"], scored.hard_violations[0])

    def test_same_root_support_does_not_raise_source_reputation(self) -> None:
        store = EvidenceStore()
        claim = {
            "entity_id": "database",
            "predicate": "attack_objective",
            "object": "exfiltration",
            "scope": "cyber_defense",
        }
        evidence_id = store.add_event(
            {
                "event_id": "event-a",
                "source_id": "sensor-A",
                "verdict": "supported",
                "claim_semantics": claim,
            },
            time=0,
        )
        manager = ContextualTrustManager()
        manager.register_claim(
            {"event_id": "event-a", "source_id": "sensor-A", "claim_semantics": claim},
            time=0,
            public_prior=0.55,
        )
        before = manager.source_reputation["sensor-A"]["mean"]
        event = manager.apply(
            [
                {
                    "op": "support",
                    "source_id": "sensor-A",
                    "event_id": "event-a",
                    "evidence_refs": [evidence_id],
                }
            ],
            evidence_store=store,
            time=0,
        )[0]
        self.assertTrue(event["committed"])
        self.assertFalse(event["source_reputation_updated"])
        self.assertEqual(manager.source_reputation["sensor-A"]["mean"], before)
        self.assertGreater(manager.claim_for("event-a")["score"], before)

    def test_probe_followed_by_unrelated_action_is_not_useful(self) -> None:
        store = EvidenceStore()
        probe_evidence = store.add_tool_result(
            {"tool": "SourceChallenge", "active_probe": True, "probe_id": "probe-a"},
            time=-1,
        )
        env = SimpleNamespace(
            t=2,
            evidence_store=store,
            history=[
                {
                    "tool_result": {
                        "active_probe": True,
                        "probe_id": "probe-a",
                        "evidence_id": probe_evidence,
                    }
                },
                {
                    "trust_events": [],
                    "memory_events": [],
                    "response_result": {
                        "executed_action": "LimitSession",
                        "authorization_evidence_ids": [],
                    },
                },
            ],
        )
        count, useful, probe_yield, availability, state_updates, action_revisions = (
            _probe_metrics(env)
        )
        self.assertEqual((count, useful, state_updates, action_revisions), (1, 0, 0, 0))
        self.assertEqual(probe_yield, 0.0)
        self.assertEqual(availability, 1.0)

    def test_first_actionable_time_ignores_honest_prefix(self) -> None:
        env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=6)
        env.observe()
        score = score_trajectory_v2(env)
        self.assertEqual(score["first_actionable_time"], 2)
        self.assertEqual(score["decision_delay_steps"], 4)

    def test_formal_mode_rejects_missing_trajectory_reward(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"AGZ_REQUIRE_TRAJECTORY_REWARD": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "terminal trajectory reward"):
                score_vda_prediction_v2_fallback(json.dumps(DEFAULT_ACTION_PACKET_V4))

    def test_feedback_prompt_keeps_instruction_at_turn_16(self) -> None:
        state = {
            "initial_messages": [
                {"role": "system", "content": "SYSTEM ROLE; strict JSON schema required"}
            ],
            "instruction_messages": [
                {"role": "system", "content": "SYSTEM ROLE; strict JSON schema required"}
            ],
            "continuation_prompt_mode": "snapshot",
            "history": [{"t": index, "decision": {"action": "Observe"}} for index in range(16)],
            "history_window": 6,
            "public_context": {"observation": {"time": 16}},
        }
        messages = _generation_messages(state)
        self.assertIn("SYSTEM ROLE", messages[0]["content"])
        self.assertIn("strict JSON schema", messages[0]["content"])
        self.assertIn('"history_steps_total":16', messages[-1]["content"])

    def test_rollout_store_parallelizes_distinct_trajectory_ids(self) -> None:
        store = Level1RolloutStore(max_parallel_trajectories=2)
        original = store._step_state
        counter_lock = threading.Lock()
        active = 0
        peak = 0

        def measured(*args, **kwargs):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            try:
                return original(*args, **kwargs)
            finally:
                with counter_lock:
                    active -= 1

        scenario = minimal_example_v2()
        with mock.patch.object(store, "_step_state", side_effect=measured):
            result = store.handle(
                {
                    "trajectory_ids": ["parallel-a", "parallel-b"],
                    "actions": [json.dumps(DEFAULT_ACTION_PACKET_V4)] * 2,
                    "finish": [False, False],
                    "is_last_step": [False, False],
                    "extra_fields": [
                        {"scenario": copy.deepcopy(scenario), "max_env_steps": 4},
                        {"scenario": copy.deepcopy(scenario), "max_env_steps": 4},
                    ],
                }
            )
        self.assertEqual(result["valids"], [1, 1])
        self.assertEqual(peak, 2)

    def test_rollout_store_serializes_same_trajectory_id(self) -> None:
        store = Level1RolloutStore(max_parallel_trajectories=4)
        original = store._step_state
        counter_lock = threading.Lock()
        active = 0
        peak = 0

        def measured(*args, **kwargs):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            try:
                return original(*args, **kwargs)
            finally:
                with counter_lock:
                    active -= 1

        scenario = minimal_example_v2()
        with mock.patch.object(store, "_step_state", side_effect=measured):
            result = store.handle(
                {
                    "trajectory_ids": ["serial-a", "serial-a"],
                    "actions": [json.dumps(DEFAULT_ACTION_PACKET_V4)] * 2,
                    "finish": [False, False],
                    "is_last_step": [False, False],
                    "extra_fields": [
                        {"scenario": scenario, "max_env_steps": 4},
                        {"scenario": scenario, "max_env_steps": 4},
                    ],
                }
            )
        self.assertEqual(result["valids"], [1, 1])
        self.assertEqual(peak, 1)

    def test_feedback_history_window_is_explicit_and_bounded(self) -> None:
        state = {
            "initial_messages": [{"role": "system", "content": "system"}],
            "instruction_messages": [{"role": "system", "content": "system"}],
            "continuation_prompt_mode": "snapshot",
            "history": [{"t": index} for index in range(10)],
            "history_window": 3,
            "public_context": {"observation": {"time": 10}},
        }
        messages = _generation_messages(state)
        content = messages[-1]["content"]
        payload = json.loads(
            content.split("Compact trajectory state (history is chronological):", 1)[1]
            .split("\nReturn the next compact strict VDA JSON action only.", 1)[0]
        )
        self.assertEqual([item["t"] for item in payload["history"]], [7, 8, 9])
        self.assertEqual(payload["history_steps_total"], 10)
        self.assertTrue(payload["history_truncated"])

        state["history_window"] = 0
        full_content = _generation_messages(state)[-1]["content"]
        full_payload = json.loads(
            full_content.split("Compact trajectory state (history is chronological):", 1)[1]
            .split("\nReturn the next compact strict VDA JSON action only.", 1)[0]
        )
        self.assertEqual(len(full_payload["history"]), 10)
        self.assertNotIn("history_truncated", full_payload)
        self.assertNotIn("history_steps_total", full_payload)

    def test_reward_fingerprint_cache_detects_external_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            with mock.patch.dict(
                os.environ,
                {"AGZ_DCA_REWARD_FSYNC_EVERY_BATCHES": "8"},
                clear=False,
            ):
                dca_online_reward._append_rows(
                    path, [{"scenario_fingerprint": "fingerprint-a"}]
                )
            self.assertEqual(
                dca_online_reward._existing_fingerprints(path),
                {"fingerprint-a"},
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"scenario_fingerprint": "fingerprint-b"}) + "\n")
            self.assertEqual(
                dca_online_reward._existing_fingerprints(path),
                {"fingerprint-a", "fingerprint-b"},
            )

    def test_formal_jobs_lock_low_risk_performance_controls(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "scripts/jobs/tmcd_v2_4b_full_node175.dsub.sh",
            "scripts/jobs/tmcd_v2_4b_append_only_node208.dsub.sh",
            "scripts/jobs/tmcd_v2_9b_full_node217.dsub.sh",
        ):
            source = (root / relative_path).read_text(encoding="utf-8")
            self.assertLess(
                source.index('source "${ROOT}/scripts/env.sh"'),
                source.index("export AGZ_DCA_PPO_MICRO_BATCH_SIZE_PER_GPU=1"),
            )
            self.assertIn(
                "export AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES=16",
                source,
            )
            self.assertIn("export AGZ_DCA_REWARD_FSYNC_EVERY_BATCHES=8", source)
            self.assertIn("export AGZ_REQUIRE_TRAJECTORY_REWARD=1", source)
            self.assertIn("export AGZ_VDA_FEEDBACK_HISTORY_WINDOW=6", source)
            self.assertIn("env | grep '^AGZ_'", source)
            self.assertIn("=<redacted>", source)

if __name__ == "__main__":
    unittest.main()
