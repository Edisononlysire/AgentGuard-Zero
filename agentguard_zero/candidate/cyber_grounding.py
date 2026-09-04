"""Cyber-grounded variants of the canonical T1--T4 recovery scenarios.

The mappings in this module are public defensive abstractions. They contain no
payloads, exploit commands, real addresses, or private organization data.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from agentguard_zero.env.checker import full_check
from agentguard_zero.recovery.canonical_scenarios import canonical_recovery_group
from agentguard_zero.schemas.scenario_schema_v2 import public_prefix_hash


ATTACK_VERSION = "19.0"
CAGE2_COMMIT = "26ce1c1253fa9e2e73f25e6a7f2da32860c11257"


CYBER_ARCHETYPES: dict[str, tuple[dict[str, Any], ...]] = {
    "T1": (
        {
            "id": "endpoint-defense-impairment",
            "attack_ids": ("T1562.001", "T1059.001"),
            "tactics": ("Defense Impairment", "Execution"),
            "telemetry": ("edr_tamper_alert", "windows_powershell_4104", "zeek_conn"),
            "assets": ("analyst_workstation", "identity_provider", "application_server", "customer_database"),
            "replay": "CAGE2-Meander-discovery-to-impact-sanitized",
            "cage_actions": ("Monitor", "Analyse", "Remove"),
        },
        {
            "id": "ambiguous-staged-exfiltration",
            "attack_ids": ("T1074.001", "T1041"),
            "tactics": ("Collection", "Exfiltration"),
            "telemetry": ("file_audit_4663", "proxy_egress_flow", "dns_query"),
            "assets": ("finance_workstation", "file_server", "proxy_gateway", "finance_database"),
            "replay": "ATTACK-Eval-data-staging-sanitized",
            "cage_actions": ("Monitor", "Analyse", "Remove"),
        },
    ),
    "T2": (
        {
            "id": "valid-account-trusted-relationship",
            "attack_ids": ("T1078", "T1199"),
            "tactics": ("Initial Access", "Persistence"),
            "telemetry": ("idp_signin", "vpn_auth", "cloud_audit"),
            "assets": ("partner_laptop", "vpn_gateway", "identity_provider", "payment_api"),
            "replay": "CAGE2-BLine-valid-session-sanitized",
            "cage_actions": ("Monitor", "Analyse", "Remove", "Restore"),
        },
        {
            "id": "remote-services-lateral-movement",
            "attack_ids": ("T1021.001", "T1078.002"),
            "tactics": ("Lateral Movement", "Defense Evasion"),
            "telemetry": ("windows_logon_4624", "rdp_gateway", "edr_network_connection"),
            "assets": ("contractor_laptop", "jump_host", "admin_console", "directory_server"),
            "replay": "CAGE2-BLine-lateral-movement-sanitized",
            "cage_actions": ("Monitor", "Analyse", "Remove", "Restore"),
        },
    ),
    "T3": (
        {
            "id": "security-data-manipulation",
            "attack_ids": ("T1565.001", "T1112"),
            "tactics": ("Impact", "Defense Evasion"),
            "telemetry": ("siem_audit_log", "registry_change", "edr_process"),
            "assets": ("analyst_host", "siem_collector", "case_store", "security_database"),
            "replay": "ATTACK-Eval-security-data-manipulation-sanitized",
            "cage_actions": ("Monitor", "Analyse", "Remove", "Restore"),
        },
        {
            "id": "account-policy-manipulation",
            "attack_ids": ("T1098", "T1484.001"),
            "tactics": ("Persistence", "Privilege Escalation"),
            "telemetry": ("directory_audit_5136", "idp_admin_audit", "change_ticket"),
            "assets": ("admin_workstation", "identity_provider", "policy_server", "directory_database"),
            "replay": "ATTACK-Eval-account-policy-change-sanitized",
            "cage_actions": ("Monitor", "Analyse", "Remove", "Restore"),
        },
    ),
    "T4": (
        {
            "id": "inhibit-system-recovery",
            "attack_ids": ("T1490", "T1489"),
            "tactics": ("Impact",),
            "telemetry": ("backup_admin_audit", "service_control_7036", "edr_process"),
            "assets": ("operator_console", "backup_server", "production_api", "order_database"),
            "replay": "CAGE2-impact-response-cost-sanitized",
            "cage_actions": ("Monitor", "Analyse", "Remove", "Restore"),
        },
        {
            "id": "service-disruption-false-positive",
            "attack_ids": ("T1499.004", "T1562.001"),
            "tactics": ("Impact", "Defense Impairment"),
            "telemetry": ("load_balancer_health", "application_error", "edr_tamper_alert"),
            "assets": ("support_portal", "load_balancer", "application_server", "customer_database"),
            "replay": "CAGE2-benign-activity-overresponse-sanitized",
            "cage_actions": ("Monitor", "Analyse", "Remove", "Restore"),
        },
    ),
}


def _archetype(task_id: str, group_index: int) -> dict[str, Any]:
    choices = CYBER_ARCHETYPES[task_id]
    return copy.deepcopy(choices[group_index % len(choices)])


def cyber_grounding_for(task_id: str, group_index: int) -> dict[str, Any]:
    """Return the frozen non-privileged provenance for one public scenario."""

    profile = _archetype(task_id, group_index)
    wire = f"v9:{task_id}:{group_index}:{profile['id']}".encode("utf-8")
    return {
        "schema_version": 1,
        "profile_id": profile["id"],
        "mitre_attack_version": ATTACK_VERSION,
        "mitre_technique_ids": list(profile["attack_ids"]),
        "mitre_tactics": list(profile["tactics"]),
        "telemetry_sources": list(profile["telemetry"]),
        "cage_challenge2_commit": CAGE2_COMMIT,
        "cage_blue_action_mapping": list(profile["cage_actions"]),
        "sanitized_replay_id": profile["replay"],
        "contains_exploit_payload": False,
        "contains_real_address": False,
        "contains_private_organization": False,
        "mapping_digest": hashlib.sha256(wire).hexdigest(),
    }


def _apply_profile(scenario: dict[str, Any], task_id: str, group_index: int) -> None:
    profile = _archetype(task_id, group_index)
    old_assets = list((scenario.get("network_context") or {}).get("assets") or [])
    names = list(profile["assets"])
    if len(old_assets) > len(names):
        raise ValueError("cyber archetype has fewer assets than the canonical scenario")
    mapping = {
        str(asset["id"]): names[index] for index, asset in enumerate(old_assets)
    }
    new_assets = []
    for index, asset in enumerate(old_assets):
        updated = copy.deepcopy(asset)
        updated["id"] = names[index]
        updated["zone"] = (
            "user" if index == 0 else "security" if index == 1 else "production"
        )
        new_assets.append(updated)
    network = scenario["network_context"]
    network["assets"] = new_assets
    network["reachable_edges"] = [
        [mapping.get(str(left), str(left)), mapping.get(str(right), str(right))]
        for left, right in network.get("reachable_edges", [])
    ]
    attack = scenario["true_attack"]
    attack["entry"] = mapping.get(str(attack.get("entry")), names[0])
    attack["target"] = mapping.get(str(attack.get("target")), names[-1])
    for index, event in enumerate(scenario.get("event_schedule") or []):
        old_entity = str(event.get("entity_id", ""))
        entity = mapping.get(old_entity, attack["target"])
        event["entity_id"] = entity
        event["type"] = profile["telemetry"][index % len(profile["telemetry"])]
        semantics = dict(event.get("claim_semantics") or {})
        semantics["entity_id"] = entity
        event["claim_semantics"] = semantics
        event["claim"] = (
            f"{event['type']} reports {semantics.get('predicate', 'activity')} "
            f"for {entity} consistent with {semantics.get('object', 'unknown')}"
        )
    oracle = scenario.get("oracle") or {}
    oracle["overresponse_condition"] = (
        f"high_impact_action_on_{attack['target']}_without_verified_risk"
    )
    metadata = dict(scenario.get("metadata") or {})
    metadata["generator"] = "agentguard_zero_v9_cyber_grounded_v1"
    metadata["generator_group_index"] = int(group_index)
    metadata["cyber_grounding"] = cyber_grounding_for(task_id, group_index)
    scenario["metadata"] = metadata


def cyber_grounded_recovery_group(
    task_id: str, group_index: int
) -> list[dict[str, Any]]:
    """Create a schema-valid hidden-world group with cyber-grounded public cues."""

    group = canonical_recovery_group(task_id, group_index)
    for scenario in group:
        _apply_profile(scenario, task_id, group_index)
    if task_id == "T2":
        for scenario in group:
            scenario["prefix_hash"] = public_prefix_hash(scenario)
    for scenario in group:
        checks = full_check(scenario)
        if not checks.get("all_ok", False):
            raise ValueError(
                f"invalid cyber-grounded scenario {scenario['scenario_id']}: {checks}"
            )
    return group


def cyber_grounded_recovery_group_v2(
    task_id: str, group_index: int
) -> list[dict[str, Any]]:
    """Create the corrected lifecycle scenarios used by V10-v2 data.

    V1's T3 sequence never supplied a second independent public record for the
    item written to long-term memory.  The memory FSM therefore had no legal
    route from quarantine to confirmed, so every generated response used the
    record as a contradiction and ``accepted_memory_ids`` stayed empty.  V2
    keeps the poisoning setup, but makes the later source-A record an
    independent corroboration of the source-B risk claim.  This permits the
    intended public-only ingest -> retrieve -> verify -> promote -> use chain.

    The original constructor is intentionally left unchanged so frozen V1
    datasets and suites remain reproducible.
    """

    group = cyber_grounded_recovery_group(task_id, group_index)
    if task_id == "T2":
        for scenario in group:
            scenario.setdefault("defense_constraints", {})[
                "require_passive_check_before_active_probe"
            ] = True
            metadata = dict(scenario.get("metadata") or {})
            grounding = dict(metadata.get("cyber_grounding") or {})
            grounding["low_cost_passive_anchor"] = True
            metadata["cyber_grounding"] = grounding
            metadata["generator"] = "agentguard_zero_v10_cyber_grounded_v2"
            scenario["metadata"] = metadata
            scenario["prefix_hash"] = public_prefix_hash(scenario)
    if task_id == "T3":
        for scenario in group:
            events = list(scenario.get("event_schedule") or [])
            risk = next(
                (
                    event
                    for event in events
                    if str(event.get("event_id", "")).endswith("profile-high")
                ),
                None,
            )
            corroboration = next(
                (
                    event
                    for event in events
                    if str(event.get("event_id", "")).endswith(
                        "history-dependent"
                    )
                ),
                None,
            )
            if risk is None or corroboration is None:
                raise ValueError(
                    "T3-v2 requires profile-high and history-dependent events"
                )
            independent_source = next(
                (
                    str(event.get("source_id", ""))
                    for event in events
                    if str(event.get("source_id", ""))
                    and str(event.get("source_id", ""))
                    != str(risk.get("source_id", ""))
                ),
                "",
            )
            if not independent_source:
                raise ValueError("T3-v2 requires an independent corroborating source")
            safe = next(
                (
                    event
                    for event in events
                    if str(event.get("event_id", "")).endswith("profile-safe")
                ),
                None,
            )
            if safe is None:
                raise ValueError("T3-v2 requires the profile-safe event")
            # Corroborate before the final-decision/delayed-recall window.  This
            # leaves one step each for verification, trust grounding, promotion,
            # and a later confirmed retrieval that can be accepted by response.
            safe["time"], corroboration["time"] = (
                corroboration["time"],
                safe["time"],
            )
            corroboration["source_id"] = independent_source
            corroboration["source"] = independent_source
            corroboration["claim_semantics"] = copy.deepcopy(
                risk.get("claim_semantics") or {}
            )
            corroboration["entity_id"] = str(risk.get("entity_id", ""))
            corroboration["objective_hint"] = str(
                risk.get("objective_hint", "exfiltration")
            )
            corroboration["claim"] = (
                f"{corroboration.get('type', 'telemetry')} independently corroborates "
                f"{corroboration['claim_semantics'].get('predicate', 'risk')}="
                f"{corroboration['claim_semantics'].get('object', 'unknown')} for "
                f"{corroboration['entity_id']}"
            )
            recovery_contract = group_index % 2 == 1
            if recovery_contract:
                # Content-counterfactual branch: two independent sources first
                # repeat the same false content, which makes the legal lifecycle
                # ingest -> cross-check -> contradict -> reject.  Two later,
                # independent and truthful records carry the corrected content,
                # permitting support -> reopen -> accepted use.  Public states
                # never expose the truth labels; they only expose the changing
                # evidence content and the resulting verification records.
                risk["truth_value"] = False
                risk["is_fake"] = True
                risk["spoofability"] = 0.0
                corroboration["truth_value"] = False
                corroboration["is_fake"] = True
                corroboration["spoofability"] = 0.0
                recovery_events = [
                    event
                    for event in events
                    if str(event.get("event_id", "")).endswith(
                        ("final-decision", "delayed-recall")
                    )
                ]
                if len(recovery_events) != 2:
                    raise ValueError("T3-v2 recovery requires two recovery events")
                for index, event in enumerate(recovery_events):
                    event["source_id"] = (
                        str(risk.get("source_id", ""))
                        if index == 0
                        else independent_source
                    )
                    event["source"] = event["source_id"]
                    event["claim_semantics"] = copy.deepcopy(
                        risk.get("claim_semantics") or {}
                    )
                    event["entity_id"] = str(risk.get("entity_id", ""))
                    event["objective_hint"] = str(
                        risk.get("objective_hint", "exfiltration")
                    )
                    event["claim"] = (
                        f"{event.get('type', 'telemetry')} supplies corrected "
                        f"independent evidence for "
                        f"{event['claim_semantics'].get('predicate', 'risk')}="
                        f"{event['claim_semantics'].get('object', 'unknown')} for "
                        f"{event['entity_id']}"
                    )
                    event["truth_value"] = True
                    event["is_fake"] = False
                    event["spoofability"] = 0.0
                # Keep the corrected claim in the live retrieval context while
                # the tool result is grounded, the rejected record is reopened,
                # and the confirmed record is finally used.  Retrieval is keyed
                # to the current public event rather than to hidden episode
                # state, so each lifecycle step receives an explicit public cue.
                for time in (6, 7, 8, 9, 10, 11):
                    event = copy.deepcopy(recovery_events[time % 2])
                    event["event_id"] = (
                        f"event-recovery-t3-{group_index:04d}-recovery-refresh-{time}"
                    )
                    event["time"] = time
                    event["claim"] = (
                        f"{event.get('type', 'telemetry')} repeats corrected "
                        f"evidence for "
                        f"{event['claim_semantics'].get('predicate', 'risk')}="
                        f"{event['claim_semantics'].get('object', 'unknown')} for "
                        f"{event['entity_id']}"
                    )
                    events.append(event)
                events.sort(
                    key=lambda event: (
                        int(event.get("time", -1)),
                        str(event.get("event_id", "")),
                    )
                )
                scenario["event_schedule"] = events
                lifecycle_contract = "t3_content_counterfactual_recovery_v2"
            else:
                corroboration["truth_value"] = True
                corroboration["is_fake"] = False
                corroboration["spoofability"] = 0.20
                lifecycle_contract = "t3_independent_corroboration_v2"
            scenario.setdefault("defense_constraints", {})["horizon"] = 12
            scenario["defense_constraints"]["verification_budget"] = max(
                8,
                int(
                    scenario["defense_constraints"].get(
                        "verification_budget", 8
                    )
                ),
            )
            metadata = dict(scenario.get("metadata") or {})
            metadata["lifecycle_contract"] = lifecycle_contract
            metadata["generator"] = "agentguard_zero_v10_cyber_grounded_v2"
            grounding = dict(metadata.get("cyber_grounding") or {})
            grounding["lifecycle_contract"] = lifecycle_contract
            grounding["content_counterfactual"] = recovery_contract
            grounding["recovery_trajectory"] = recovery_contract
            metadata["cyber_grounding"] = grounding
            scenario["metadata"] = metadata
    for scenario in group:
        checks = full_check(scenario)
        if not checks.get("all_ok", False):
            raise ValueError(
                f"invalid cyber-grounded v2 scenario {scenario['scenario_id']}: {checks}"
            )
    return group


def known_attack_ids() -> set[str]:
    return {
        technique
        for profiles in CYBER_ARCHETYPES.values()
        for profile in profiles
        for technique in profile["attack_ids"]
    }
