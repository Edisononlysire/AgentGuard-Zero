from __future__ import annotations

from typing import Any


from agentguard_zero.defender_state.evidence_signals import evidence_signal


def _grounded_contradiction(env: Any, operation: dict, claim: dict, *, start: int) -> bool:
    """Audit the committed decision at its own time, not a decayed final score."""
    if operation.get('op') != 'contradict' or not operation.get('source_reputation_updated'):
        return False
    time = int(operation.get('time', -1))
    refs = list(operation.get('evidence_refs', []))
    if not refs or time < max(start, int(claim['time'])):
        return False
    store = env.evidence_store
    if not store.validate_refs(refs, time=time)[0]:
        return False
    if not store.refs_support_claim(refs, claim.get('claim_semantics', {}), time=time)[0]:
        return False
    records = [store.get(ref, time=time) for ref in refs]
    records = [row for row in records if row and int(row.get('available_at', -1)) >= start]
    signals = [evidence_signal(row) for row in records]
    return sum(neg for _, neg in signals) > sum(pos for pos, _ in signals)


def trust_metrics(env: Any) -> dict[str, Any]:
    ledger = list(getattr(env, 'oracle_ledger', []))
    by_id = {str(row['event_id']): row for row in ledger}
    claim_state = env.trust_manager.claim_trust
    operations = [
        row for row in env.trust_manager.events
        if row.get('committed') and str(row.get('event_id', '')) in by_id
        and str(row.get('source_id', '')) == str(by_id[str(row['event_id'])]['source_id'])
        and int(row.get('time', -1)) >= int(by_id[str(row['event_id'])]['time'])
    ]
    true_ids = {key for key, row in by_id.items() if row.get('truth_value', True)}
    false_ids = set(by_id) - true_ids
    challenged = {str(row['event_id']) for row in operations if row.get('op') in {'challenge', 'contradict'}}
    contradicted = {str(row['event_id']) for row in operations if row.get('op') == 'contradict'}
    supported = {str(row['event_id']) for row in operations if row.get('op') in {'support', 'recover'}}
    brier = [
        (float(claim_state.get(key, {}).get('score', 0.5)) - float(key in true_ids)) ** 2
        for key in by_id
    ]
    divergence = int(env.scenario.get('divergence_time', 0))
    change_ids = {
        key for key in true_ids if int(by_id[key]['time']) >= divergence
    } if env.scenario.get('trajectory_type') == 'legitimate_change' else set()

    targets = []
    # Fix the evaluation population before policy execution; an early failure
    # must not remove an undetected scheduled betrayal from the denominator.
    scheduled = {
        str(row['event_id']): row for row in env.scenario.get('event_schedule', [])
        if 0 <= int(row['time']) < env.max_steps
    }
    if env.scenario.get('scenario_family') == 'trust_betrayal' and env.scenario.get('trajectory_type') == 'betrayal':
        for profile in env.scenario.get('source_profiles', []):
            source_id = str(profile['source_id'])
            for segment in profile.get('behavior_schedule', []):
                if segment.get('mode') not in {'deceptive', 'mixed'}:
                    continue
                start = max(divergence, int(segment['start_time']))
                end = int(segment['end_time'])
                false_events = {
                    key for key, row in scheduled.items() if not row.get('truth_value', True)
                    and str(row['source_id']) == source_id
                    and start <= int(row['time']) <= end
                }
                if false_events:
                    targets.append((source_id, start, env.max_steps - 1, false_events))
    delays = []
    suspected = 0
    for source, start, end, ids in targets:
        relevant = [
            row for row in operations if str(row['source_id']) == source
            and str(row['event_id']) in ids and start <= int(row['time']) <= end
        ]
        suspected += int(any(row.get('op') in {'challenge', 'contradict'} for row in relevant))
        detected_at = [
            int(row['time']) for row in relevant
            if _grounded_contradiction(env, row, by_id[str(row['event_id'])], start=start)
        ]
        if detected_at:
            delays.append(float(min(detected_at) - start))
    false_alarms = true_ids & contradicted
    return {
        'claim_trust_brier': sum(brier) / max(1, len(brier)),
        'high_impact_false_claim_acceptance': len(false_ids & supported) / max(1, len(false_ids)),
        'false_distrust_rate': len(false_alarms) / max(1, len(true_ids)),
        'false_suspicion_rate': len(true_ids & challenged) / max(1, len(true_ids)),
        'legitimate_change_false_suspicion_rate': len(change_ids & challenged) / max(1, len(change_ids)),
        'legitimate_change_false_rejection_rate': len(change_ids & contradicted) / max(1, len(change_ids)),
        'legitimate_change_claim_count': len(change_ids),
        'betrayal_target_count': len(targets),
        'betrayal_detection_applicable': bool(targets),
        'betrayal_detection_count': len(delays),
        'betrayal_detection_rate': len(delays) / max(1, len(targets)),
        'betrayal_suspicion_rate': suspected / max(1, len(targets)),
        'betrayal_detection_delay': sum(delays) / max(1, len(delays)),
        'betrayal_detection_delay_defined': bool(delays),
        'true_change_acceptance': len(change_ids & supported) / max(1, len(change_ids)),
    }
