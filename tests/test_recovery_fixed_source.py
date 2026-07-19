from __future__ import annotations

import copy
import unittest

from scripts.build_recovery_sequence_balanced_data import select_with_target_cap
from scripts.eval_recovery_fixed_source import balanced_scenario_prefix

from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.canonical_scenarios import canonical_recovery_group
from agentguard_zero.recovery.public_teacher import public_state_digest
from agentguard_zero.recovery.model_policy import RecoveryModelPolicy
from agentguard_zero.recovery.action_serialization import action_first_wire_json
from agentguard_zero.recovery.action_intent import (
    INTENT_FORMAT,
    action_intent,
    action_intent_wire_json,
    compact_intent_prompt,
    parse_action_intent,
)
from agentguard_zero.schemas.action_schema_v4 import parse_action_json_v4
from agentguard_zero.recovery.source_counterfactuals import (
    audit_counterfactual,
    counterfactual_groups,
)


class FixedSourceCounterfactualTest(unittest.TestCase):
    def test_counterfactual_is_public_equivalent_and_distinct(self) -> None:
        for task_id in ("T1", "T2", "T3", "T4"):
            source = canonical_recovery_group(task_id, 7)[0]
            paired = audit_counterfactual(source)
            self.assertNotEqual(source["scenario_id"], paired["scenario_id"])
            self.assertEqual(
                public_state_digest(instantiate_scenario(source).observe()),
                public_state_digest(instantiate_scenario(paired).observe()),
            )

    def test_source_is_not_mutated(self) -> None:
        source = canonical_recovery_group("T3", 8)[0]
        snapshot = copy.deepcopy(source)
        groups = counterfactual_groups([source])
        self.assertEqual(source, snapshot)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)


class _RecordingTokenizer:
    chat_template = "qwen-test-template"

    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "rendered"


class RecoveryInferenceContractTest(unittest.TestCase):
    def test_qwen_thinking_is_explicitly_disabled(self) -> None:
        policy = object.__new__(RecoveryModelPolicy)
        policy.tokenizer = _RecordingTokenizer()
        policy.output_format = "full_v4"
        rendered = policy.render_prompt("public state")
        self.assertEqual(rendered, "rendered")
        self.assertEqual(policy.tokenizer.kwargs["enable_thinking"], False)
        self.assertEqual(policy.tokenizer.kwargs["add_generation_prompt"], True)

    def test_exact_target_frequency_cap_is_deterministic(self) -> None:
        rows = [
            {"record_id": "c", "target": "same"},
            {"record_id": "a", "target": "same"},
            {"record_id": "b", "target": "same"},
            {"record_id": "d", "target": "other"},
        ]
        selected = select_with_target_cap(rows, target_cap=2)
        self.assertEqual([row["record_id"] for row in selected], ["a", "b", "d"])

    def test_quick_evaluation_is_task_balanced(self) -> None:
        scenarios = [
            {"scenario_id": f"{task}-{index}", "metadata": {"task_id": task}}
            for task in ("T1", "T2", "T3", "T4")
            for index in range(5)
        ]
        selected, method = balanced_scenario_prefix(scenarios, 8)
        self.assertEqual(method, "task_balanced_prefix")
        counts = {
            task: sum(row["metadata"]["task_id"] == task for row in selected)
            for task in ("T1", "T2", "T3", "T4")
        }
        self.assertEqual(counts, {"T1": 2, "T2": 2, "T3": 2, "T4": 2})

    def test_action_first_serialization_is_schema_preserving(self) -> None:
        packet = {
            "schema_version": 4,
            "belief": {
                "exfiltration": 0.25,
                "sabotage": 0.25,
                "persistence": 0.25,
                "credential_theft": 0.25,
            },
            "assessment": None,
            "trust_operation": None,
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
        serialized = action_first_wire_json(packet)
        self.assertTrue(serialized.startswith('{"response":{"action":"Observe"'))
        _, valid, reason = parse_action_json_v4(serialized)
        self.assertTrue(valid, reason)

        packet["tool_call"] = {
            "name": "SourceChallenge",
            "args": {"event_id": "evt-1", "evidence_ids": ["ev-1"]},
        }
        probe = action_first_wire_json(packet)
        self.assertTrue(probe.startswith('{"tool_call":{"name":"SourceChallenge"'))

    def test_compact_intent_roundtrip_preserves_primary_action(self) -> None:
        packet = {
            "schema_version": 4,
            "belief": {
                "exfiltration": 0.25,
                "sabotage": 0.25,
                "persistence": 0.25,
                "credential_theft": 0.25,
            },
            "assessment": None,
            "trust_operation": None,
            "memory_operation": None,
            "memory_use": None,
            "uncertainty": 1.0,
            "tool_call": {
                "name": "LogQuery",
                "args": {"entity_id": "host-1"},
            },
            "safety_check": {
                "business_risk": 0.0,
                "overresponse_risk": 0.0,
                "justification": "query",
            },
            "response": {"tier": "L1", "action": "Observe", "target": "none"},
        }
        intent = action_intent(packet)
        self.assertEqual(intent["kind"], "tool")
        wire = action_intent_wire_json(packet)
        compiled, valid, reason = parse_action_intent(wire)
        self.assertTrue(valid, reason)
        self.assertEqual(compiled["tool_call"]["name"], "LogQuery")

        packet["tool_call"] = {"name": "None", "args": {}}
        packet["response"] = {
            "tier": "L2",
            "action": "Isolate",
            "target": "host-1",
        }
        wire = action_intent_wire_json(packet)
        compiled, valid, reason = parse_action_intent(wire)
        self.assertTrue(valid, reason)
        self.assertEqual(compiled["response"]["action"], "Isolate")

    def test_compact_intent_prompt_removes_full_schema_scaffold(self) -> None:
        full = (
            "Return this full schema and key order.\n"
            'Current decision instance:{"observation":{"time":1}}'
        )
        compact = compact_intent_prompt(full)
        self.assertIn(INTENT_FORMAT.split("_")[0], "compact")
        self.assertNotIn("Return this full schema", compact)
        self.assertIn('"observation":{"time":1}', compact)


if __name__ == "__main__":
    unittest.main()
