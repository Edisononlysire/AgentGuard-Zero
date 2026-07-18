from __future__ import annotations

import copy
import json
import pickle
import sys
import tempfile
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
    ACTION_HORIZONS,
    SHORTLIST_QUOTAS,
    PublicStateRobustTeacher,
    TeacherDecision,
    compact_wire_json,
    enumerate_public_candidates,
    public_state_digest,
)
from agentguard_zero.recovery.protocol import RecoveryConfig
from agentguard_zero.rewards.recovery_vda_reward import (
    REQUIRED_RECOVERY_SIGNALS,
    compute_recovery_reward,
    validate_recovery_signal_batch,
)
from agentguard_zero.schemas.action_schema_v4 import (
    DEFAULT_ACTION_PACKET_V4,
    parse_action_json_v4,
)
from agentguard_zero.schemas.scenario_schema_v2 import paired_minimal_examples_v2
from agentguard_zero.training.vda_dataset import build_vda_prompt
from agentguard_zero.world.public_projector import assert_public
from scripts.build_recovery_bootstrap import load_accepted_stage0


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
    def test_bootstrap_parent_stage0_gate_is_fail_closed(self):
        payload = {
            "kind": "recovery_stage0_audit",
            "protocol_version": RecoveryConfig().protocol_version,
            "accepted": True,
            "status": "accepted",
            "next_stage": "bootstrap_data_build_and_audit",
            "model_calls": 0,
            "parameter_updates": 0,
            "verdict": {
                "gate": "stage0_fixed_policy",
                "accepted": True,
                "next_stage": "bootstrap_data_build_and_audit",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage0.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(load_accepted_stage0(path)["accepted"])
            payload["parameter_updates"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "model-free"):
                load_accepted_stage0(path)

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
        self.assertEqual(
            payload["bootstrap_sft"]["lora_target_modules"],
            list(config.bootstrap_sft.lora_target_modules),
        )
        self.assertEqual(
            payload["execution_scope"]["allowed_now"],
            [
                "stage0_fixed_policy_gate",
                "public_teacher_behavior_audit",
                "bootstrap_data_build_after_accepted_stage0",
            ],
        )
        locked = set(payload["execution_scope"]["locked_until_teacher_review"])
        self.assertTrue(
            {
                "gate_a_dual_arm_bootstrap_sft",
                "single_dagger_correction",
                "gate_b_10_step_rl",
                "dca_vda_coevolution",
            }.issubset(locked)
        )

    def test_bootstrap_emits_one_record_per_public_decision(self):
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
                    core_q_audit={"observe": 0.0},
                )

        group = list(paired_minimal_examples_v2())
        result = build_bootstrap_records(
            [group],
            teacher=ObserveTeacher(),
            max_records=1,
        )
        self.assertEqual(len(result.train_records), 1)
        self.assertEqual(len(result.audit_records), 1)
        self.assertEqual(result.audit_records[0]["world_count"], 2)
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
        teacher = PublicStateRobustTeacher(beam_width=20, max_candidates=40)
        decision = teacher.decide(envs, horizon=3)
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

    def test_public_teacher_uses_common_horizon_and_diverse_shortlist_quota(self):
        self.assertEqual(set(ACTION_HORIZONS.values()), {3})
        self.assertGreaterEqual(sum(SHORTLIST_QUOTAS.values()), 16)
        self.assertGreaterEqual(SHORTLIST_QUOTAS["active_probe"], 4)
        self.assertEqual(SHORTLIST_QUOTAS["mitigation"], 6)
        with self.assertRaisesRegex(ValueError, "beam_width"):
            PublicStateRobustTeacher(beam_width=4)
        with self.assertRaisesRegex(ValueError, "root horizon"):
            first, second = paired_minimal_examples_v2()
            PublicStateRobustTeacher().decide(
                [instantiate_scenario(first), instantiate_scenario(second)],
                horizon=2,
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
                horizon=3,
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
        partial_progress = evaluate_rollout_group(
            [
                {
                    "scenario_id": "partial",
                    "reward": value,
                    "action_class": action,
                    "safe_success": False,
                }
                for value, action in (
                    (-1.0, "observe"),
                    (-0.7, "active_probe"),
                    (-0.2, "trust"),
                    (0.1, "mitigation"),
                )
            ]
        )
        self.assertTrue(partial_progress.all_failed)
        self.assertTrue(partial_progress.usable_for_policy_gradient)

    def test_recovery_reward_gives_no_positive_format_reward(self):
        signals = {
            "parse_ok": True,
            "schema_ok": True,
            "safe_success": False,
            "attack_success": False,
            "action": "Observe",
            "teacher_advantage": 0.20,
            "counterfactual_advantage": -0.2,
            "new_evidence": False,
            "uncertainty_reduced": False,
            "probe_grounded_state_update": False,
            "probe_grounded_authorization": False,
            "probe_counterfactual_improvement": False,
            "correct_state_transitions": 0,
            "false_state_transitions": 0,
            "business_cost": 0.0,
            "verification_cost": 0.0,
            "delay": 0.0,
            "core_utility": -0.1,
        }
        self.assertEqual(set(signals), set(REQUIRED_RECOVERY_SIGNALS))
        reward = compute_recovery_reward(signals)
        self.assertEqual(reward["format_component"], 0.0)
        self.assertLess(reward["noop_component"], 0.0)
        invalid_signals = dict(signals, parse_ok=False, schema_ok=False)
        invalid = compute_recovery_reward(invalid_signals)
        self.assertLess(invalid["overall"], 0.0)
        self.assertFalse(invalid["normal_environment_action_allowed"])
        missing = dict(signals)
        missing.pop("teacher_advantage")
        with self.assertRaisesRegex(RuntimeError, "teacher_advantage"):
            compute_recovery_reward(missing)

        smoke_rows = []
        for index in range(32):
            row = dict(signals)
            row.update(
                {
                    "action": "SourceChallenge" if index % 3 == 0 else "Observe",
                    "teacher_advantage": index / 31.0,
                    "counterfactual_advantage": (index - 16) / 16.0,
                    "core_utility": 0.6 if index % 2 == 0 else -0.6,
                    "new_evidence": index % 3 == 0,
                    "correct_state_transitions": int(index % 4 == 0),
                }
            )
            smoke_rows.append(row)
        self.assertTrue(validate_recovery_signal_batch(smoke_rows)["accepted"])
        broken_smoke = copy.deepcopy(smoke_rows)
        broken_smoke[0].pop("teacher_advantage")
        broken_verdict = validate_recovery_signal_batch(broken_smoke)
        self.assertFalse(broken_verdict["accepted"])
        self.assertTrue(
            any("teacher_advantage" in item for item in broken_verdict["failures"])
        )

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
        self.assertEqual(
            evaluate_stage0_gate(metrics).next_stage,
            "bootstrap_data_build_and_audit",
        )
        diagnostic_only = copy.deepcopy(metrics)
        diagnostic_only["random_legal"]["safe_utility"] = 0.05
        self.assertTrue(evaluate_stage0_gate(diagnostic_only).accepted)
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
            prompt_variant = prompt + (" " * index)
            train.append(
                {
                    "record_id": record_id,
                    "messages": [
                        {"role": "user", "content": prompt_variant},
                        {"role": "assistant", "content": target},
                    ],
                    "prompt": prompt_variant,
                    "target": target,
                    "public_state_digest": public_state_digest(env.observe()),
                    "action_category": category,
                    "source_policy": (
                        "finite_counterfactual_public_state_robust_teacher"
                    ),
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
                    "q_audit": {"a": 0.0, "b": 1.0},
                    "core_q_audit": {"a": 0.0, "b": 1.0},
                }
            )
        result = audit_bootstrap_records(train, audits)
        self.assertTrue(result["accepted"], result)
        leaked = copy.deepcopy(train)
        leaked[0]["scenario"] = json.dumps(first)
        self.assertFalse(audit_bootstrap_records(leaked, audits)["accepted"])


if __name__ == "__main__":
    unittest.main()
