"""Regression tests for the revised simulator, not model performance tests."""

import copy

import pytest

from agentguard_zero.env.cyber_env_v2 import CyberDefenseEnvV2
from agentguard_zero.env.oracle_v2 import _probe_metrics, score_trajectory_v2
from agentguard_zero.evaluation.rq4_trust import trust_metrics
from agentguard_zero.schemas.action_schema_v4 import DEFAULT_ACTION_PACKET_V4
from agentguard_zero.schemas.scenario_schema_v2 import minimal_example_v2, validate_scenario_v2
from agentguard_zero.world.hidden_world import HiddenWorld
from agentguard_zero.world.public_projector import assert_public
from agentguard_zero.world.canary import schedule_canary, canary_measurement


def packet(**updates):
    action = copy.deepcopy(DEFAULT_ACTION_PACKET_V4)
    action.update(updates)
    return action


def run_to_end(env):
    while env.t < env.max_steps:
        _, _, done = env.step(packet())
        if done:
            break
    return score_trajectory_v2(env)


def advance_to_divergence(env):
    env.observe()
    while env.t < env.scenario['divergence_time']:
        env.step(packet())
    return next(row for row in env.observe()['observed_events'] if row['source_id'] == 'sensor-A')


def evidence_for(env, event, *, verdict='contradicted', available_at=None):
    return env.evidence_store.add_tool_result(
        {'tool': 'CrossCheck', 'event_id': event['event_id'],
         'verdict': verdict, 'claim_semantics': event['claim_semantics']},
        time=env.t - 1 if available_at is None else available_at - 1,
        parent_evidence_ids=[event['evidence_id']],
    )


def test_physical_response_is_independent_of_belief_label():
    worlds = [HiddenWorld(minimal_example_v2(), horizon=8) for _ in range(2)]
    for world, belief in zip(worlds, (True, False)):
        world.apply_response(action='LimitSession', belief_matches=belief,
                             target_matches=True, time=0)
    assert worlds[0].attack_pressure == worlds[1].attack_pressure
    assert worlds[0].response_history[0]['effective_strength'] > 0


def test_wrong_target_still_has_no_physical_effect():
    world = HiddenWorld(minimal_example_v2(), horizon=8)
    before = world.attack_pressure
    world.apply_response(action='Isolate', belief_matches=True, target_matches=False, time=0)
    assert world.attack_pressure == before


def test_no_future_events_are_silently_skipped_after_mitigation():
    from scripts.eval_candidate_policy import _terminal
    scenario = minimal_example_v2()
    scenario['true_attack']['initial_pressure'] = 0.1
    env = CyberDefenseEnvV2(scenario, max_steps=6)
    _, _, done = env.step(packet(response={'tier': 'L1', 'action': 'LimitSession', 'target': 'database'},
                                belief={'exfiltration': 1.0}))
    assert env.attack_mitigated
    assert not done
    assert not _terminal(env)
    assert not score_trajectory_v2(env)['safe_success']
    score = run_to_end(env)
    assert env.t > max(row['time'] for row in scenario['event_schedule'])
    assert score['intent_applicable']


def test_benign_control_completes_without_fabricated_attack_failure():
    scenario = minimal_example_v2(trajectory_type='legitimate_change')
    scenario['true_attack']['present'] = False
    env = CyberDefenseEnvV2(scenario, max_steps=6)
    before = score_trajectory_v2(env)
    assert not before['task_success']
    score = run_to_end(env)
    assert not score['attack_success']
    assert not score['attack_mitigated']
    assert score['task_success'] and score['safe_success']
    assert not score['intent_applicable']
    assert not score['delay_applicable']
    assert score['delay'] == 0


def test_benign_control_does_not_reward_unnecessary_containment():
    scenario = minimal_example_v2(trajectory_type='legitimate_change')
    scenario['true_attack']['present'] = False
    env = CyberDefenseEnvV2(scenario)
    env.step(packet(response={'tier': 'L1', 'action': 'LimitSession', 'target': 'database'}))
    score = run_to_end(env)
    assert score['false_containment_count'] == 1
    assert not score['safe_success']


def test_benign_truth_flag_requires_a_boolean():
    scenario = minimal_example_v2()
    scenario['true_attack']['present'] = 'false'
    assert validate_scenario_v2(scenario)[0] is False


def test_challenge_all_is_suspicion_not_confirmed_betrayal():
    env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=6)
    event = advance_to_divergence(env)
    env.step(packet(trust_operations=[{'op': 'challenge', 'source_id': 'sensor-A',
                                      'event_id': event['event_id'], 'evidence_refs': [event['evidence_id']]}]))
    metrics = trust_metrics(env)
    assert metrics['betrayal_detection_rate'] == 0
    assert metrics['betrayal_suspicion_rate'] == 1


def test_betrayal_population_is_fixed_before_policy_execution():
    env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=6)
    env.observe()
    metrics = trust_metrics(env)
    assert metrics['betrayal_target_count'] == 1
    assert metrics['betrayal_detection_count'] == 0
    assert metrics['betrayal_detection_rate'] == 0.0


def test_challenging_legitimate_change_is_recorded_as_false_suspicion():
    env = CyberDefenseEnvV2(minimal_example_v2(trajectory_type='legitimate_change'))
    event = advance_to_divergence(env)
    env.step(packet(trust_operations=[{'op': 'challenge', 'source_id': 'sensor-A',
                                      'event_id': event['event_id'], 'evidence_refs': [event['evidence_id']]}]))
    metrics = trust_metrics(env)
    assert metrics['betrayal_target_count'] == 0
    assert metrics['legitimate_change_false_suspicion_rate'] > 0


def test_evidence_backed_correct_source_contradiction_detects_betrayal():
    env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=6)
    event = advance_to_divergence(env)
    evidence_id = evidence_for(env, event)
    env.step(packet(trust_operations=[{'op': 'contradict', 'source_id': 'sensor-A',
                                      'event_id': event['event_id'], 'evidence_refs': [evidence_id]}]))
    metrics = trust_metrics(env)
    assert metrics['betrayal_detection_rate'] == 1
    assert metrics['betrayal_detection_count'] == 1
    assert metrics['betrayal_detection_delay_defined']


@pytest.mark.parametrize('failure', ['wrong_source', 'future_evidence', 'unrelated_evidence'])
def test_invalid_contradiction_does_not_detect_betrayal(failure):
    env = CyberDefenseEnvV2(minimal_example_v2(), max_steps=6)
    event = advance_to_divergence(env)
    ref_event = dict(event)
    if failure == 'unrelated_evidence':
        ref_event = next(row for row in env.scenario['event_schedule'] if row['time'] == 0)
        ref_event = dict(ref_event, evidence_id=env.evidence_store.evidence_for_event(ref_event['event_id']))
    ref = evidence_for(env, ref_event, available_at=env.t + 2 if failure == 'future_evidence' else None)
    env.step(packet(trust_operations=[{'op': 'contradict',
                                      'source_id': 'sensor-B' if failure == 'wrong_source' else 'sensor-A',
                                      'event_id': event['event_id'], 'evidence_refs': [ref]}]))
    assert trust_metrics(env)['betrayal_detection_rate'] == 0


def test_failed_tool_attempt_is_charged_once_and_separately_reported():
    env = CyberDefenseEnvV2(minimal_example_v2())
    env.step(packet(tool_call={'name': 'SourceChallenge', 'args': {'event_id': 'missing'}}))
    score = score_trajectory_v2(env)
    assert score['invalid_tool_cost'] > 0
    assert score['invalid_tool_cost'] == score['verification_cost']
    assert score['valid_verification_cost'] == 0
    assert env.history[-1]['tool_result']['cost'] == score['verification_cost']
    assert score['active_probe_count'] == 1
    assert score['probe_result_availability_rate'] == 0.0
    assert not env.history[-1]['tool_result'].get('active_probe', False)


def test_pending_probe_receipt_is_not_measurement_evidence():
    env = CyberDefenseEnvV2(minimal_example_v2())
    env._resolve_due_probes = lambda: []
    _, result, _ = env.step(packet(tool_call={'name': 'DecoyProbe', 'args': {'zone': 'server'}}))
    assert result['status'] == 'pending'
    assert 'evidence_id' not in result
    assert _probe_metrics(env)[3] == 0


def test_probe_reference_is_not_given_unconditional_reward():
    env = CyberDefenseEnvV2(minimal_example_v2())
    event = env.observe()['observed_events'][0]
    env.step(packet(tool_call={'name': 'SourceChallenge', 'args': {'event_id': event['event_id']}}))
    score = score_trajectory_v2(env)
    assert score['active_probe_reward_component'] == 0
    assert score['probe_yield_is_causal'] is False
    assert score['metric_revision'].startswith('aep-')


def test_resolved_probe_is_public_and_cannot_authorize_same_step_response():
    env = CyberDefenseEnvV2(minimal_example_v2())
    event = env.observe()['observed_events'][0]
    _, result, _ = env.step(packet(
        tool_call={'name': 'SourceChallenge', 'args': {'event_id': event['event_id']}},
        response={'tier': 'L3', 'action': 'Isolate', 'target': 'database'},
        evidence_assessment=[{'event_id': event['event_id'], 'status': 'supported', 'score': 1.0}],
    ))
    assert env.history[-1]['response_result']['blocked']
    assert result['available_at'] == 1
    assert env.evidence_store.get(result['evidence_id'], time=0) is None
    assert_public(env.observe())


def test_canary_does_not_read_claim_truth_or_objective():
    scenario = minimal_example_v2()
    event = scenario['event_schedule'][0]
    changed = dict(event, truth_value=False, is_fake=True, objective_hint='sabotage')
    kwargs = dict(profile=scenario['source_profiles'][0], model={}, time=0,
                  probe_id='probe-0-0', draw=lambda *_: 0.5)
    first = schedule_canary(event=event, **kwargs)
    second = schedule_canary(event=changed, **kwargs)
    assert first == second
    assert canary_measurement(first, time=1) == canary_measurement(second, time=1)


def test_canary_normal_packet_loss_is_not_a_malicious_verdict():
    scenario = minimal_example_v2()
    model = {'canary': {'drop_probability': 1.0, 'timeout': 3}}
    pending = schedule_canary(profile=scenario['source_profiles'][0], model=model,
                              event=scenario['event_schedule'][0], time=0,
                              probe_id='p0', draw=lambda *_: 0.5)
    assert canary_measurement(pending, time=1) is None
    measured = canary_measurement(pending, time=3)
    assert measured['status'] == 'expired'
    assert not measured['marker_received']
    assert 'verdict' not in measured
    assert 'truth_value' not in measured


def test_canary_injection_delivery_and_evidence_commit_are_separate():
    scenario = minimal_example_v2()
    scenario['probe_model'] = {'canary': {'drop_probability': 0.0, 'alter_probability': 0.0, 'latency': 2}}
    env = CyberDefenseEnvV2(scenario)
    event = env.observe()['observed_events'][0]
    observation, receipt, _ = env.step(packet(tool_call={'name': 'CanaryProbe', 'args': {'event_id': event['event_id']}}))
    assert receipt['status'] == 'pending'
    assert 'evidence_id' not in receipt
    assert _probe_metrics(env)[3] == 0
    observation, _, _ = env.step(packet())
    result = next(row for row in observation['observed_events'] if row.get('type') == 'canary_probe_result')
    assert result['marker_received'] and result['content_intact']
    assert env.evidence_store.get(result['evidence_id'], time=1) is None
    assert env.evidence_store.get(result['evidence_id'], time=2) is not None
    assert _probe_metrics(env)[3] == 1
    assert_public(observation)


@pytest.mark.parametrize('value', [-0.1, 1.2, float('nan'), True])
def test_canary_rejects_invalid_noise_parameters(value):
    scenario = minimal_example_v2()
    scenario['probe_model'] = {'canary': {'drop_probability': value}}
    assert not validate_scenario_v2(scenario)[0]


def test_probe_use_diagnostic_does_not_change_reward(monkeypatch):
    env = CyberDefenseEnvV2(minimal_example_v2())
    env.observe()
    monkeypatch.setattr('agentguard_zero.env.oracle_v2._probe_metrics', lambda _: (1, 0, 0.0, 1.0, 0, 0))
    before = score_trajectory_v2(env)['reward']
    monkeypatch.setattr('agentguard_zero.env.oracle_v2._probe_metrics', lambda _: (1, 1, 1.0, 1.0, 1, 1))
    after = score_trajectory_v2(env)['reward']
    assert before == after


def test_outcome_aggregation_excludes_inapplicable_populations():
    from agentguard_zero.evaluation.outcomes import aggregate_outcomes
    attack = run_to_end(CyberDefenseEnvV2(minimal_example_v2()))
    scenario = minimal_example_v2(trajectory_type='legitimate_change')
    scenario['true_attack']['present'] = False
    benign = run_to_end(CyberDefenseEnvV2(scenario))
    summary = aggregate_outcomes([attack, benign])
    assert summary['scenario_count'] == 2
    assert summary['attack_scenario_count'] == 1
    assert summary['safe_success_rate'] == 0.5
    benign_only = aggregate_outcomes([benign])
    assert benign_only['intent_accuracy'] is None
    assert benign_only['attack_mitigation_rate'] is None
    assert benign_only['betrayal_detection_rate'] is None
    with pytest.raises(ValueError, match='mix legacy'):
        aggregate_outcomes([attack, dict(benign, metric_revision='old')])


def test_completed_episode_cannot_be_reopened_to_repair_its_score():
    env = CyberDefenseEnvV2(minimal_example_v2())
    before = run_to_end(env)
    with pytest.raises(RuntimeError, match='episode_already_complete'):
        env.step(packet(response={'tier': 'L1', 'action': 'LimitSession', 'target': 'database'}))
    assert score_trajectory_v2(env) == before


def test_budget_denial_does_not_charge_again():
    scenario = minimal_example_v2()
    scenario['defense_constraints']['verification_budget'] = 1
    env = CyberDefenseEnvV2(scenario)
    failed = packet(tool_call={'name': 'SourceChallenge', 'args': {'event_id': 'missing'}})
    env.step(failed)
    charged = env.verification_cost
    _, result, _ = env.step(failed)
    assert result['status'] == 'budget_exhausted'
    assert result['cost'] == 0
    assert env.verification_cost == charged
    assert env.invalid_tool_cost == charged


def test_teacher_has_no_legacy_probe_bonus_under_revised_metrics():
    from agentguard_zero.recovery.public_teacher import teacher_rollout_shaping
    env = CyberDefenseEnvV2(minimal_example_v2())
    env.observe()
    assert teacher_rollout_shaping(env) == 0.0


def test_shard_merge_uses_applicable_denominators(tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.eval_candidate_policy import merge
    from agentguard_zero.protocol import AEP_ENV_REVISION, AEP_METRIC_REVISION
    paths = []
    for index in range(2):
        applicable = 1 if index == 0 else 0
        metrics = {
            'scenario_count': 1, 'safe_success': 1.0, 'task_success': 1.0,
            'attack_scenario_count': applicable, 'betrayal_target_count': applicable,
            'attack_mitigation': 1.0 if applicable else None,
            'betrayal_detection': 1.0 if applicable else None,
            'metric_revision': AEP_METRIC_REVISION, 'environment_revision': AEP_ENV_REVISION,
        }
        path = tmp_path / f'shard{index}.json'
        path.write_text(json.dumps({'shard_index': index, 'shard_count': 2, 'traces': [],
                                    'metrics': {**metrics, 'by_task_terminal': {'T2': dict(metrics)}}}))
        paths.append(path)
    output = tmp_path / 'merged.json'
    merge(SimpleNamespace(inputs=paths, output=output))
    metrics = json.loads(output.read_text())['metrics']
    assert metrics['attack_mitigation'] == 1.0
    assert metrics['betrayal_detection'] == 1.0
    assert metrics['task_success'] == 1.0
    assert metrics['scenario_count'] == 2
    assert metrics['attack_scenario_count'] == 1
    assert metrics['by_task_terminal']['T2']['betrayal_detection'] == 1.0
