from __future__ import annotations

import copy
import math
from typing import Any


MITIGATION_STRENGTH = {
    "Observe": 0.0,
    "CrossCheck": 0.0,
    "QueryProvenance": 0.0,
    "SourceChallenge": 0.0,
    "CanaryProbe": 0.0,
    "DecoyProbe": 0.0,
    "ShadowActionProbe": 0.0,
    "DeployDecoy": 0.25,
    "LimitSession": 0.45,
    "ShadowBlock": 0.55,
    "Isolate": 1.0,
    "Restore": 0.35,
    "Remove": 0.80,
}


class HiddenWorld:
    """Oracle-only attack state. No instance is exposed through observations."""

    def __init__(self, scenario: dict[str, Any], *, horizon: int) -> None:
        attack = scenario.get("true_attack", {})
        self.true_objective = str(attack.get("objective", "unknown"))
        self.entry = str(attack.get("entry", "unknown"))
        self.target = str(attack.get("target", "unknown"))
        self.phase_schedule = [str(item) for item in attack.get("phase_schedule", [])]
        self.phase_index = 0
        self.ticks = 0
        self.phase_duration = max(1, math.ceil(max(1, int(horizon)) / max(1, len(self.phase_schedule))))
        self.attack_pressure = 1.0
        self.mitigated = False
        self.succeeded = False
        self.response_history: list[dict[str, Any]] = []

    @property
    def current_phase(self) -> str:
        if not self.phase_schedule:
            return "unknown"
        index = min(self.phase_index, len(self.phase_schedule) - 1)
        return self.phase_schedule[index]

    def snapshot(self) -> dict[str, Any]:
        return {
            "true_objective": self.true_objective,
            "entry": self.entry,
            "target": self.target,
            "phase_index": self.phase_index,
            "phase_duration": self.phase_duration,
            "current_phase": self.current_phase,
            "attack_pressure": self.attack_pressure,
            "mitigated": self.mitigated,
            "succeeded": self.succeeded,
        }

    def apply_response(self, *, action: str, belief_matches: bool, time: int) -> None:
        strength = float(MITIGATION_STRENGTH.get(str(action), 0.0))
        effective = strength if belief_matches else 0.0
        self.attack_pressure = max(0.0, self.attack_pressure - effective)
        self.mitigated = self.attack_pressure <= 1e-6
        self.response_history.append(
            {
                "time": int(time),
                "action": str(action),
                "belief_matches": bool(belief_matches),
                "mitigation_strength": strength,
                "effective_strength": effective,
            }
        )

    def advance_attack(self) -> None:
        if self.mitigated or self.succeeded:
            return
        self.ticks += 1
        if self.ticks % self.phase_duration:
            return
        self.phase_index += 1
        if self.phase_schedule and self.phase_index >= len(self.phase_schedule):
            self.succeeded = True

    def clone_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.snapshot())
