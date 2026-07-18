from __future__ import annotations

import copy
import json
import pickle
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard_zero.env.scenario_instantiator import instantiate_scenario
from agentguard_zero.recovery.bootstrap_data import (
    audit_bootstrap_records,
    build_bootstrap_records,
)
from agentguard_zero.recovery.canonical_scenarios import canonical_recovery_suite
from agentguard_zero.recovery.dynamic_sampling import (
    evaluate_rollout_group,
    summarize_update_batch,
)
from agentguard_zero.recovery.fixed_policies import run_fixed_policy_episode
from agentguard_zero.recovery.gates import (
    choose_gate_a_arm,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_stage0_gate,
)
from agentguard_zero.recovery.public_teacher import (
    PublicStateRobustTeacher,
    TeacherDecision,
    compact_wire_json,
    enumerate_public_candidates,
    public_state_digest,
)
from agentguard_zero.recovery.protocol import RecoveryConfig
from agentguard_zero.rewards.recovery_vda_reward import compute_recovery_reward
from agentguard_zero.schemas.action_schema_v4 import (
    DEFAULT_ACTION_PACKET_V4,
    parse_action_json_v4,
)
from agentguard_zero.schemas.scenario_schema_v2 import paired_minimal_examples_v2
from agentguard_zero.training.vda_dataset import build_vda_prompt
from agentguard_zero.world.public_projector import assert_public


def _gate_a_metrics(**overrides):
    metrics = {
        "scenario_count": 200,
        "original_gate_scenarios": 80,
        "new_canonical_scenarios": 120,
        "candidate_count": 1,
        "decoding": "greedy",
        "action_validity": 0.995,
        "actionable_observe_rate": 0.60,
        "active_probe_rate": 0.12,
        "attack_mitigation": 0.20,
        "safe_success": 0.15,
        "probe_yield": 0.10,
        "trust_memory_operation_rate": 0.08,
        "invalid_operation_rate": 0.01,
        "safe_utility": 0.25,
    }
    metrics.update(overrides)
    return metrics


class RecoveryProtocolTests(unittest.TestCase):
    def test_frozen_recovery_config_matches_python_protocol(self):
        payload = json.loads(
            (ROOT / "configs/recovery/action_support_bootstrap_v1.json").read_text(
                encoding="utf-8"
            )
        )
        config = RecoveryConfig()
        self.assertEqual(payload["protocol_version"], config.protocol_version)
        self.assertEqual(payload["zero_definition"], config.zero_definition)
        self.assertEqual(payload["lineage"], config.lineage_disposition)

    def test_bootstrap_replicates_public_label_without_branch_id_leak(self):
        class ObserveTeacher:
            def decide(self, worlds, **_kwargs):
                observation = worlds[0].observe()
                digest = public_state_digest(observation)
                return TeacherDecision(
                    public_state_digest=digest,
                    selected_candidate_id="observe",
                    selected_category="observe",
                    selected_packet=copy.deepcopy(DEFAULT_ACTION_PACKET_V4),
                    robust_value=0.0,
                    observe_value=0.0,
                    advantage_over_observe=0.0,
                    public_candidate_count=1,
                    admitted_candidate_count=1,
                    world_count=len(worlds),
                    search_horizon=1,
                    q_audit={"observe": 0.0},
                )

        group = list(paired_minimal_examples_v2())
        result = build_bootstrap_records(
            [group],
            teacher=ObserveTeacher(),
            max_records=2,
        )
        self.assertEqual(len(result.train_records), 2)
        self.assertNotEqual(
            result.train_records[0]["record_id"],
            result.train_records[1]["record_id"],
        )
        self.assertEqual(
            result.train_records[0]["prompt"],
            result.train_records[1]["prompt"],
        )
        for scenario in group:
            self.assertNotIn(
                str(scenario["scenario_id"]),
                result.train_records[0]["prompt"],
            )

    def test_canonical_recovery_suite_has_balanced_public_world_pairs(self):
        groups = canonical_recovery_suite(scenario_count=8)
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(len(group) == 2 for group in groups))
        for group in groups:
            observations = [instantiate_scenario(item).observe() for item in group]
            self.assertEqual(
                public_state_digest(observations[0]),
                public_state_digest(observations[1]),
            )

    def test_noop_uniform_belief_is_not_credited_as_correct_intent(self):
        first, _ = paired_minimal_examples_v2()
        _, score = run_fixed_policy_episode(first, "no_op")
        self.assertIs(score["correct_intent"], False)

    def test_public_teacher_candidates_are_public_and_cover_action_support(self):
        first, _ = paired_minimal_examples_v2()
        observation = instantiate_scenario(first).observe()
        candidates = enumerate_public_candidates(observation)
        categories = {item.category for item in candidates}
        self.assertTrue(
            {
                "observe",
                "passive_verification",
                "active_probe",
                "trust",
                "memory",
                "mitigation",
            }.issubset(categories)
        )
        for candidate in candidates:
            assert_public(candidate.packet)

    def test_public_teacher_uses_one_action_without_mutating_worlds(self):
        first, second = paired_minimal_examples_v2()
        envs = [instantiate_scenario(first), instantiate_scenario(second)]
        self.assertEqual(
            public_state_digest(envs[0].observe()),
            public_state_digest(envs[1].observe()),
        )
        before = [pickle.dumps(env, protocol=5) for env in envs]
        teacher = PublicStateRobustTeacher(beam_width=3, max_candidates=40)
        decision = teacher.decide(envs, horizon=1)
        self.assertEqual(decision.world_count, 2)
        self.assertFalse(decision.hidden_state_in_target)
        assert_public(decision.selected_packet)
        target = compact_wire_json(decision.selected_packet)
        _, valid, reason = parse_action_json_v4(target)
        self.assertTrue(valid, reason)
        self.assertEqual(
            [pickle.dumps(env, protocol=5) for env in envs],
            before,
        )

    def test_public_teacher_belief_updates_only_from_public_tool_evidence(self):
        first, _ = paired_minimal_examples_v2()
        env = instantiate_scenario(first)
        challenge = next(
            item
            for item in enumerate_public_candidates(env.observe())
            if item.category == "active_probe"
            and item.label.startswith("SourceChallenge:")
        )
        env.step(copy.deepcopy(challenge.packet))
        observation = env.observe()
        assert_public(observation)
        observe = next(
            item
            for item in enumerate_public_candidates(observation)
            if item.category == "observe"
        )
        belief = observe.packet["belief"]
        self.assertGreater(max(belief.values()), min(belief.values()))
        self.assertAlmostEqual(sum(belief.values()), 1.0)

    def test_public_teacher_rejects_single_hidden_world_at_gate_entry(self):
        first, _ = paired_minimal_examples_v2()
        with self.assertRaisesRegex(ValueError, "minimum"):
            PublicStateRobustTeacher().decide(
                [instantiate_scenario(first)],
                horizon=1,
            )

    def test_dynamic_sampling_requests_two_more_then_filters_or_uses(self):
        initial = [
            {
                "scenario_id": "x",
                "reward": -1.0,
                "action_class": "observe",
                "safe_success": False,
            },
            {
                "scenario_id": "x",
                "reward": -1.0,
                "action_class": "observe",
                "safe_success": False,
            },
        ]
        first = evaluate_rollout_group(initial)
        self.assertEqual(first.action, "resample")
        self.assertEqual(first.additional_rollouts, 2)
        useful = evaluate_rollout_group(
            [
                {
                    "scenario_id": "x",
                    "reward": value,
                    "action_class": action,
                    "safe_success": success,
                }
                for value, action, success in (
                    (-1.0, "observe", False),
                    (-0.3, "active_probe", False),
                    (0.2, "trust", False),
                    (1.0, "mitigation", True),
                )
            ]
        )
        self.assertTrue(useful.usable_for_policy_gradient)
        self.assertTrue(summarize_update_batch([useful])["policy_update_allowed"])

    def test_recovery_reward_gives_no_positive_format_reward(self):
        reward = compute_recovery_reward(
            {
                "parse_ok": True,
                "schema_ok": True,
                "action": "Observe",
                "teacher_advantage": 0.20,
                "counterfactual_advantage": -0.2,
            }
        )
        self.assertEqual(reward["format_component"], 0.0)
        self.assertLess(reward["noop_component"], 0.0)
        invalid = compute_recovery_reward({"parse_ok": False, "schema_ok": False})
        self.assertLess(invalid["overall"], 0.0)
        self.assertFalse(invalid["normal_environment_action_allowed"])

    def test_stage0_gate_is_strictly_ordered_and_fail_closed(self):
        metrics = {
            "oracle": {
                "scenario_count": 200,
                "safe_utility": 0.80,
                "attack_mitigation": 0.95,
            },
            "public_state_teacher": {
                "scenario_count": 200,
                "safe_utility": 0.55,
                "attack_mitigation": 0.50,
            },
            "random_legal": {
                "scenario_count": 200,
                "safe_utility": 0.20,
                "attack_mitigation": 0.10,
            },
            "no_op": {
                "scenario_count": 200,
                "safe_utility": 0.10,
                "attack_mitigation": 0.0,
            },
            "overreact": {
                "scenario_count": 200,
                "safe_utility": -0.10,
                "attack_mitigation": 0.10,
            },
        }
        self.assertTrue(evaluate_stage0_gate(metrics).accepted)
        broken = copy.deepcopy(metrics)
        broken["public_state_teacher"]["safe_utility"] = 0.15
        self.assertFalse(evaluate_stage0_gate(broken).accepted)

    def test_gate_a_dual_arm_selection_and_hard_no_go(self):
        base = evaluate_gate_a(
            _gate_a_metrics(safe_utility=0.30),
            arm="qwen3.5_base",
        )
        vda1 = evaluate_gate_a(
            _gate_a_metrics(safe_success=0.20, safe_utility=0.28),
            arm="vda_1",
        )
        selected = choose_gate_a_arm([base, vda1])
        self.assertTrue(selected["accepted"])
        self.assertEqual(selected["selected_arm"], "qwen3.5_base")
        collapsed = evaluate_gate_a(
            _gate_a_metrics(
                attack_mitigation=0.0,
                probe_yield=0.0,
                trust_memory_operation_rate=0.0,
                actionable_observe_rate=0.90,
            ),
            arm="vda_1",
        )
        self.assertFalse(collapsed.accepted)
        self.assertTrue(
            any(item.startswith("hard_no_go:") for item in collapsed.failures)
        )

    def test_gate_b_requires_improvement_kl_replay_and_contract(self):
        baseline = {"safe_utility": 0.20, "attack_mitigation": 0.20}
        metrics = {
            "scenario_count": 240,
            "rl_steps": 10,
            "initial_rollouts": 2,
            "adaptive_rollouts": 4,
            "bootstrap_replay_ratio": 0.20,
            "use_kl_loss": True,
            "kl_coef": 0.02,
            "action_validity": 0.99,
            "actionable_observe_rate": 0.70,
            "probe_yield": 0.12,
            "attack_mitigation": 0.22,
            "safe_utility": 0.24,
        }
        self.assertTrue(evaluate_gate_b(metrics, baseline).accepted)
        broken = dict(metrics)
        broken["safe_utility"] = 0.19
        self.assertFalse(evaluate_gate_b(broken, baseline).accepted)

    def test_bootstrap_audit_separates_training_rows_from_q_audit(self):
        first, _ = paired_minimal_examples_v2()
        env = instantiate_scenario(first)
        prompt = build_vda_prompt(first, env.observe())
        target = compact_wire_json(
            enumerate_public_candidates(env.observe(), max_candidates=1)[0].packet
        )
        categories = (
            "observe",
            "passive_verification",
            "active_probe",
            "trust",
            "memory",
            "mitigation",
        )
        train = []
        audits = []
        for index, category in enumerate(categories):
            record_id = f"r-{index}"
            train.append(
                {
                    "record_id": record_id,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": target},
                    ],
                    "prompt": prompt,
                    "target": target,
                    "public_state_digest": public_state_digest(env.observe()),
                    "action_category": category,
                    "source_policy": "public_state_robust_teacher",
                }
            )
            audits.append(
                {
                    "record_id": record_id,
                    "hidden_state_in_target": False,
                    "model_input_hidden_state": False,
                    "model_target_hidden_state": False,
                    "hidden_state_usage": "offline_robust_utility_only",
                    "world_count": 2,
                }
            )
        result = audit_bootstrap_records(train, audits)
        self.assertTrue(result["accepted"], result)
        leaked = copy.deepcopy(train)
        leaked[0]["scenario"] = json.dumps(first)
        self.assertFalse(audit_bootstrap_records(leaked, audits)["accepted"])


if __name__ == "__main__":
    unittest.main()
