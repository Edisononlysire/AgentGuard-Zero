from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

OBJECTIVES = ["exfiltration", "sabotage", "persistence", "credential_theft"]
PHASES = ["initial_access", "discovery", "lateral_movement", "collection", "exfiltration", "impact", "persistence"]
EVIDENCE_TYPES = ["fake_log", "fake_ttp", "decoy_touch", "suppressed_signal"]


@dataclass
class Asset:
    id: str
    zone: str
    criticality: float = 0.5


@dataclass
class NetworkContext:
    assets: List[Asset]
    reachable_edges: List[List[str]]


@dataclass
class TrueAttack:
    objective: str
    entry: str
    target: str
    phase_schedule: List[str]


@dataclass
class FakeEvidence:
    time: int
    type: str
    source: str
    claim: str
    intended_misbelief: str
    spoofability: float = 0.5


@dataclass
class PoisoningPlan:
    false_objective: str
    fake_evidence: List[FakeEvidence] = field(default_factory=list)
    decoy_behavior: List[Dict[str, Any]] = field(default_factory=list)
    strategy_switch: Optional[Dict[str, Any]] = None


@dataclass
class DefenseConstraints:
    business_budget: float = 5.0
    verification_budget: int = 4
    max_high_impact_actions: int = 1


@dataclass
class OracleSpec:
    true_objective: str
    success_condition: str = "prevent_true_objective"
    overresponse_condition: str = "high_impact_action_without_verified_risk"


@dataclass
class ScenarioSpec:
    scenario_id: str
    network_context: NetworkContext
    true_attack: TrueAttack
    poisoning_plan: PoisoningPlan
    defense_constraints: DefenseConstraints
    oracle: OracleSpec


def scenario_to_dict(s: ScenarioSpec) -> Dict[str, Any]:
    return asdict(s)


def minimal_example() -> Dict[str, Any]:
    return {
        "scenario_id": "S-000001",
        "network_context": {
            "assets": [
                {"id": "user_pc", "zone": "office", "criticality": 0.2},
                {"id": "file_server", "zone": "server", "criticality": 0.7},
                {"id": "database", "zone": "server", "criticality": 0.95},
            ],
            "reachable_edges": [["user_pc", "file_server"], ["file_server", "database"]],
        },
        "true_attack": {
            "objective": "exfiltration",
            "entry": "user_pc",
            "target": "database",
            "phase_schedule": ["initial_access", "discovery", "lateral_movement", "collection", "exfiltration"],
        },
        "poisoning_plan": {
            "false_objective": "sabotage",
            "fake_evidence": [
                {
                    "time": 2,
                    "type": "fake_log",
                    "source": "host_log",
                    "claim": "service crash pattern",
                    "intended_misbelief": "sabotage",
                    "spoofability": 0.8,
                }
            ],
            "decoy_behavior": [{"time": 3, "decoy_type": "backup_server_decoy", "intended_misbelief": "sabotage"}],
            "strategy_switch": {"time": 4, "visible_profile_before": "sabotage", "true_behavior_after": "exfiltration"},
        },
        "defense_constraints": {"business_budget": 5.0, "verification_budget": 4, "max_high_impact_actions": 1},
        "oracle": {"true_objective": "exfiltration", "success_condition": "prevent_exfiltration", "overresponse_condition": "isolate_database_without_verified_risk"},
    }
