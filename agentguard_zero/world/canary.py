"""Bounded synthetic telemetry intervention, with no access to claim truth."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable


DEFAULT_CHANNEL = {'drop_probability': 0.10, 'alter_probability': 0.02, 'latency': 1, 'timeout': 3}
MODES = {'honest', 'deceptive', 'mixed', 'legitimate_change', 'recovered'}


def validate_probe_model(model: Any) -> None:
    if not isinstance(model, dict) or set(model) - {'canary'}:
        raise ValueError('invalid_probe_model')
    canary = model.get('canary', {})
    if not isinstance(canary, dict) or set(canary) - {*DEFAULT_CHANNEL, 'mode_overrides'}:
        raise ValueError('invalid_canary_model')
    overrides = canary.get('mode_overrides', {})
    if not isinstance(overrides, dict) or set(overrides) - MODES:
        raise ValueError('invalid_canary_mode_overrides')
    base = {**DEFAULT_CHANNEL, **{key: value for key, value in canary.items() if key != 'mode_overrides'}}
    for delta in [{}, *overrides.values()]:
        if not isinstance(delta, dict) or set(delta) - set(DEFAULT_CHANNEL):
            raise ValueError('invalid_canary_channel')
        if 'timeout' in delta:
            raise ValueError('canary_deadline_must_be_common_to_all_modes')
        channel = {**base, **delta}
        for key in ('drop_probability', 'alter_probability'):
            value = channel[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'invalid_canary_{key}')
        for key in ('latency', 'timeout'):
            if isinstance(channel[key], bool) or not isinstance(channel[key], int) or channel[key] < 1:
                raise ValueError(f'invalid_canary_{key}')
        if channel['latency'] > channel['timeout']:
            raise ValueError('canary_latency_exceeds_timeout')


def schedule_canary(*, profile: dict, model: dict, event: dict, time: int,
                    probe_id: str, draw: Callable[[str, Any], float]) -> dict:
    validate_probe_model(model)
    mode = next((str(row['mode']) for row in profile.get('behavior_schedule', [])
                 if int(row['start_time']) <= time <= int(row['end_time'])), 'honest')
    settings = model.get('canary', {})
    channel = {**DEFAULT_CHANNEL, **{key: value for key, value in settings.items() if key != 'mode_overrides'},
               **settings.get('mode_overrides', {}).get(mode, {})}
    source = str(profile['source_id'])
    nonce = hashlib.sha256(f'{source}:{time}:{probe_id}'.encode()).hexdigest()[:20]
    noise_key = {'source': source, 'probe_id': probe_id}
    return {
        'probe_id': probe_id, 'source': source, 'issued_at': time, 'nonce': nonce,
        'arrives_at': time + channel['latency'], 'expires_at': time + channel['timeout'],
        'dropped': draw('canary-drop', noise_key) < channel['drop_probability'],
        'altered': draw('canary-alter', noise_key) < channel['alter_probability'],
        'event_id': str(event['event_id']),
    }


def canary_measurement(pending: dict, *, time: int) -> dict | None:
    if pending['dropped'] and time < pending['expires_at']:
        return None
    if not pending['dropped'] and time < pending['arrives_at']:
        return None
    received = not pending['dropped']
    intact = received and not pending['altered']
    state = 'received' if received else 'timeout'
    return {
        'event_id': f"{pending['probe_id']}-result", 'time': time,
        'type': 'canary_probe_result', 'source_id': 'canary-controller',
        'entity_id': pending['source'], 'probe_id': pending['probe_id'],
        'claim': f"Telemetry marker {state}",
        'claim_semantics': {'entity_id': pending['source'], 'predicate': 'canary_delivery',
                            'object': state, 'scope': 'telemetry_measurement'},
        'challenged_event_id': pending['event_id'], 'sensor_id': pending['source'],
        'issued_at': pending['issued_at'], 'status': 'resolved' if received else 'expired',
        'marker_received': received, 'content_intact': intact if received else None,
        'observed_marker': pending['nonce'] if intact else ('altered-marker' if received else None),
        'elapsed_steps': time - pending['issued_at'], 'probe_generated': True,
    }
