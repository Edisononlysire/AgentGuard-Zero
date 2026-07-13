"""Reward-manager registry used by the AgentGuard-Zero training runtime.

AgentGuard-Zero uses VerL's `batch` and `naive` managers with project-specific
reward functions. Unrelated math, search, SQL, and code-execution managers are
intentionally not imported in this release.
"""

from verl.workers.reward_manager import get_reward_manager_cls, register

__all__ = ["get_reward_manager_cls", "register"]
