from __future__ import annotations

from typing import Any, Dict

from agentguard_zero.env.cyber_env import CyberProfilePoisoningEnv
from agentguard_zero.env.cyber_env_v2 import CyberDefenseEnvV2


def instantiate_scenario(
    scenario: Dict[str, Any],
    max_steps: int | None = None,
    *,
    oracle_mode: bool = False,
) -> CyberProfilePoisoningEnv | CyberDefenseEnvV2:
    if scenario.get("protocol_version") == "tmcd-v2":
        return CyberDefenseEnvV2(
            scenario,
            max_steps=max_steps,
            oracle_mode=oracle_mode,
        )
    return CyberProfilePoisoningEnv(scenario, max_steps=max_steps)
