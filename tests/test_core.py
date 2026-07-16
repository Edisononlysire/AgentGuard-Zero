from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentguard_zero.env.checker import full_check
from agentguard_zero.memory.profile_memory import init_memory, update_memory
from agentguard_zero.rewards.dca_reward import compute_dca_reward
from agentguard_zero.rewards.vda_reward import compute_vda_reward
from agentguard_zero.schemas.action_schema import DEFAULT_ACTION_PACKET, validate_action_packet
from agentguard_zero.schemas.scenario_schema import minimal_example
from agentguard_zero.training.coevolution import RoundLayout, scenario_fingerprint
from eval_level1_select import active_probe_candidate
from eval_tmcd_systems import default_candidate_count, model_policy


class CoreMethodTests(unittest.TestCase):
    def test_minimal_scenario_passes_all_hard_checks(self) -> None:
        checks = full_check(minimal_example())
        self.assertTrue(checks["all_ok"], checks)

    def test_action_schema_rejects_non_string_memory_items(self) -> None:
        packet = copy.deepcopy(DEFAULT_ACTION_PACKET)
        packet["memory_update"]["confirmed_profile"] = [{"not": "a string"}]
        ok, message = validate_action_packet(packet)
        self.assertFalse(ok)
        self.assertEqual(message, "memory_confirmed_profile_item_not_string")

    def test_profile_memory_keeps_three_independent_partitions(self) -> None:
        memory = update_memory(
            init_memory(),
            {
                "confirmed_profile": ["verified-source"],
                "profile_quarantine": ["unverified-claim"],
                "rejected_profile": ["known-spoof"],
            },
        )
        self.assertEqual(memory["confirmed_profile"], ["verified-source"])
        self.assertEqual(memory["quarantined_profile"], ["unverified-claim"])
        self.assertEqual(memory["rejected_profile"], ["known-spoof"])

    def test_fingerprint_ignores_provenance_but_not_semantics(self) -> None:
        first = minimal_example()
        second = copy.deepcopy(first)
        second["scenario_id"] = "different-id"
        second["metadata"] = {"source_round": 99}
        self.assertEqual(scenario_fingerprint(first), scenario_fingerprint(second))
        second["true_attack"]["objective"] = "persistence"
        self.assertNotEqual(scenario_fingerprint(first), scenario_fingerprint(second))

    def test_dca_reward_hard_gates_invalid_scenarios(self) -> None:
        scenario = minimal_example()
        checks = full_check(scenario)
        valid = compute_dca_reward(
            scenario,
            {
                "checks": checks,
                "oracle_solvable": True,
                "current_vda_safe_success": False,
                "current_vda_safe_utility": 0.1,
                "difficulty": 0.7,
                "verification_value": 0.8,
            },
            task_focus="T1",
        )
        invalid_checks = copy.deepcopy(checks)
        invalid_checks["safe"]["ok"] = False
        invalid = compute_dca_reward(
            scenario,
            {"checks": invalid_checks, "oracle_solvable": True},
            task_focus="T1",
        )
        self.assertGreater(valid["overall"], invalid["overall"])
        self.assertEqual(invalid["overall"], -1.0)

    def test_vda_reward_exposes_safety_components(self) -> None:
        reward = compute_vda_reward(
            {
                "reward": 0.6,
                "attack_mitigated": True,
                "poison_defense": 1.0,
                "probe_yield": 0.5,
                "business_cost": 0.2,
                "overresponse": False,
            }
        )
        self.assertEqual(reward["overall"], 0.6)
        self.assertEqual(reward["mitigation"], 1.0)
        self.assertEqual(reward["business_cost"], -0.2)

    def test_v5c_adds_a_low_risk_active_probe(self) -> None:
        public_context = {
            "t": 0,
            "events": [
                {
                    "event_id": "event-1",
                    "source": "sensor-a",
                    "source_reliability": 0.2,
                    "spoofability": 0.9,
                }
            ],
            "verification_cost": 0,
        }
        candidate = active_probe_candidate(public_context, "v5_c_frontier_minimax")
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.packet["tool_call"]["name"], "SourceChallenge")
        self.assertEqual(candidate.packet["memory_update"]["confirmed_profile"], [])
        self.assertEqual(candidate.packet["memory_update"]["profile_quarantine"], ["event-1"])


class ReleaseIntegrityTests(unittest.TestCase):
    def test_round_layout_separates_roles_and_rounds(self) -> None:
        layout = RoundLayout(ROOT, "qwen3.5-4b", source_round=1)
        self.assertEqual(layout.target_round, 2)
        self.assertNotEqual(layout.checkpoint_dir("dca"), layout.checkpoint_dir("vda"))
        self.assertIn("round_2", str(layout.data_dir))

    def test_round_layout_keeps_v24_formal_outputs_isolated(self) -> None:
        layout = RoundLayout(
            ROOT,
            "qwen3.5-4b",
            source_round=0,
            artifact_scope="tmcd_v24",
            experiment_variant="append_only_memory",
        )
        self.assertTrue(
            str(layout.data_dir).endswith(
                "data/tmcd_v24/ablations/append_only_memory/qwen3.5-4b/round_1"
            )
        )
        self.assertTrue(
            str(layout.checkpoint_dir("vda")).endswith(
                "checkpoints/tmcd_v24/ablations/append_only_memory/"
                "qwen3.5-4b/vda/round_1"
            )
        )

    def test_orchestrator_orders_dca_update_before_fresh_pool_and_vda(self) -> None:
        source = (SCRIPTS / "run_dca_first_round.py").read_text(encoding="utf-8")
        self.assertLess(source.index('stage = "update_dca"'), source.index('stage = "generate_fresh_vda_candidates"'))
        self.assertLess(source.index('stage = "generate_fresh_vda_candidates"'), source.index('stage = "update_vda"'))
        self.assertIn("train_vda_qwen35_lora.sh", source)

    def test_training_environment_is_repository_relative(self) -> None:
        source = (SCRIPTS / "env.sh").read_text(encoding="utf-8")
        self.assertIn("BASH_SOURCE", source)
        self.assertIn("third_party/verl", source)
        self.assertNotIn("ROOT=${AGZ_ROOT:-/", source)
        self.assertNotIn("source /", source)

    def test_train_plus_v5c_uses_adapter_candidates_and_selector(self) -> None:
        system = "agentguard_zero_train_select"
        self.assertEqual(model_policy(system), "agentguard_zero_select")
        self.assertEqual(default_candidate_count(system, 4), 4)


if __name__ == "__main__":
    unittest.main()
