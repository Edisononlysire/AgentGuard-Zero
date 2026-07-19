from __future__ import annotations

import copy
import unittest

from agentguard_zero.candidate.compiler import CandidateCompiler, InvalidCandidate
from agentguard_zero.candidate.generator import CandidateGenerator, public_belief_variants
from agentguard_zero.candidate.metrics import action_flags, summarize_candidate_traces
from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.canonical_scenarios import canonical_recovery_group
from agentguard_zero.schemas.action_schema_v4 import validate_action_packet_v4


class CandidatePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        scenario = canonical_recovery_group("T3", 99000)[0]
        self.observation = instantiate_scenario(scenario).observe()
        self.generator = CandidateGenerator(min_candidates=8, max_candidates=24)

    def test_generation_is_deterministic_and_bounded(self) -> None:
        first = self.generator.generate(self.observation)
        second = self.generator.generate(copy.deepcopy(self.observation))
        self.assertGreaterEqual(len(first), 8)
        self.assertLessEqual(len(first), 24)
        self.assertEqual(
            [item.candidate_id for item in first],
            [item.candidate_id for item in second],
        )
        self.assertEqual(len(first), len({item.candidate_id for item in first}))

    def test_candidate_keys_change_without_changing_semantics(self) -> None:
        first = self.generator.generate(self.observation, permutation_seed=11)
        second = self.generator.generate(self.observation, permutation_seed=12)
        self.assertEqual(
            {item.semantic_id for item in first},
            {item.semantic_id for item in second},
        )
        first_keys = {item.semantic_id: item.candidate_key for item in first}
        second_keys = {item.semantic_id: item.candidate_key for item in second}
        self.assertTrue(
            all(first_keys[key] != second_keys[key] for key in first_keys)
        )

    def test_compiler_validates_every_admitted_packet(self) -> None:
        candidates = self.generator.generate(self.observation)
        compiler = CandidateCompiler()
        for candidate in candidates:
            packet = compiler.compile(candidate.candidate_id, candidates)
            self.assertTrue(validate_action_packet_v4(packet)[0])
        with self.assertRaises(InvalidCandidate):
            compiler.compile("missing", candidates)

    def test_belief_variants_are_normalized(self) -> None:
        variants = public_belief_variants(self.observation)
        self.assertEqual(
            set(variants),
            {"uniform", "public_posterior", "top1_moderate", "top2_ambiguous"},
        )
        for belief in variants.values():
            self.assertAlmostEqual(sum(belief.values()), 1.0)

    def test_multilabel_metrics_do_not_collapse_memory_and_mitigation(self) -> None:
        packet = copy.deepcopy(self.generator.generate(self.observation)[0].compiled_packet)
        packet["memory_usage"] = [
            {"memory_id": "memory-1", "usage": "support", "used_for": "response"}
        ]
        packet["response"] = {
            "tier": "L2",
            "action": "ShadowBlock",
            "target": "database",
        }
        flags = action_flags(packet)
        self.assertTrue(flags.memory_use)
        self.assertTrue(flags.mitigation)
        metrics = summarize_candidate_traces(
            [
                {
                    "task_id": "T3",
                    "invalid_noop": False,
                    "action_flags": flags.to_dict(),
                    "candidate_regret": 0.1,
                }
            ]
        )
        self.assertEqual(metrics["memory_use_rate"], 1.0)
        self.assertEqual(metrics["mitigation_rate"], 1.0)

    def test_invalid_noop_is_not_observe(self) -> None:
        metrics = summarize_candidate_traces(
            [
                {
                    "task_id": "T1",
                    "invalid_noop": True,
                    "action_flags": {"observe_only": True},
                }
            ]
        )
        self.assertEqual(metrics["invalid_noop_rate"], 1.0)
        self.assertEqual(metrics["observe_only_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
