from __future__ import annotations

import copy
import json
import unittest

from agentguard_zero.schemas.scenario_schema_v2 import minimal_example_v2
from agentguard_zero.training.vda_dataset import build_vda_prompt
from scripts.level1_rollout_server import Level1RolloutStore


class ProgressiveVariantRuntimeTests(unittest.TestCase):
    def assert_no_persistent_state(self, defender_state) -> None:
        defender_state = defender_state or {}
        self.assertEqual(defender_state.get("trust", {}), {})
        self.assertEqual(defender_state.get("memory", {}), {})
        self.assertEqual(defender_state.get("probe_state", []), [])

    def _state(self, variant: str):
        scenario = minimal_example_v2()
        scenario.setdefault("metadata", {})["experiment_variant"] = "full"
        original = copy.deepcopy(scenario)
        extra = {
            "scenario": json.dumps(scenario),
            "experiment_variant": variant,
            "max_env_steps": 2,
        }
        store = Level1RolloutStore(max_states=4)
        state = store._get_or_create_state(f"runtime-{variant}", extra)
        self.assertEqual(scenario, original)
        return scenario, state

    def test_static_train_disables_state_and_all_verification_tools(self) -> None:
        scenario, state = self._state("static_train")
        env = state.env
        self.assertEqual(env.experiment_variant, "static_train")
        self.assertFalse(env.state_layer_enabled)
        self.assertFalse(env.variant.passive_verification)
        self.assertFalse(env.variant.active_probing)
        self.assert_no_persistent_state(env.observe().get("defender_state"))
        snapshot = env._internal_events()
        passive = env._execute_tool({"name": "LogQuery", "args": {}}, snapshot)
        active = env._execute_tool(
            {"name": "SourceChallenge", "args": {"event_id": "evt-0"}},
            snapshot,
        )
        self.assertEqual(
            passive.get("error"), "passive_verification_disabled_by_ablation"
        )
        self.assertEqual(active.get("error"), "active_probing_disabled_by_ablation")
        prompt = build_vda_prompt(scenario, experiment_variant="static_train")
        public = json.loads(prompt.split("Current decision instance:", 1)[1])
        self.assert_no_persistent_state(public["observation"].get("defender_state"))
        self.assertIn("Passive verification is unavailable", prompt)

    def test_verification_tools_enables_only_passive_checks(self) -> None:
        scenario, state = self._state("verification_tools")
        env = state.env
        self.assertEqual(env.experiment_variant, "verification_tools")
        self.assertFalse(env.state_layer_enabled)
        self.assertTrue(env.variant.passive_verification)
        self.assertFalse(env.variant.active_probing)
        snapshot = env._internal_events()
        passive = env._execute_tool({"name": "LogQuery", "args": {}}, snapshot)
        active = env._execute_tool(
            {"name": "SourceChallenge", "args": {"event_id": "evt-0"}},
            snapshot,
        )
        self.assertNotIn("error", passive)
        self.assertEqual(active.get("error"), "active_probing_disabled_by_ablation")
        prompt = build_vda_prompt(scenario, experiment_variant="verification_tools")
        self.assertIn("LogQuery|CrossCheck|ProvenanceCheck|GraphQuery", prompt)
        self.assertNotIn("SourceChallenge|CanaryProbe", prompt)
        public = json.loads(prompt.split("Current decision instance:", 1)[1])
        self.assert_no_persistent_state(public["observation"].get("defender_state"))


if __name__ == "__main__":
    unittest.main()
