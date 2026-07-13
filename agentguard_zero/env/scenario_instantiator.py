from __future__ import annotations

from typing import Any, Dict

from agentguard_zero.env.cyber_env import CyberProfilePoisoningEnv


def instantiate_scenario(scenario: Dict[str, Any], max_steps: int | None = None) -> CyberProfilePoisoningEnv:
    return CyberProfilePoisoningEnv(scenario, max_steps=max_steps)
