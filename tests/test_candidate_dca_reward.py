from __future__ import annotations

import unittest

from agentguard_zero.rewards.candidate_dca_reward import (
    compute_candidate_dca_reward,
    frontier_score,
)


class CandidateDCARewardTest(unittest.TestCase):
    def test_frontier_prefers_mixed_outcomes(self) -> None:
        self.assertEqual(frontier_score([0, 0, 0, 0]), 0.0)
        self.assertEqual(frontier_score([1, 1, 1, 1]), 0.0)
        self.assertEqual(frontier_score([1, 1, 0, 0]), 1.0)

    def test_infrastructure_failure_cannot_earn_positive_reward(self) -> None:
        result = compute_candidate_dca_reward(
            {
                "teacher_solvable": True,
                "safe_success_samples": [1, 0, 1, 0],
                "compiler_failure": True,
                "novelty": 1.0,
                "skill_gap": 1.0,
                "vda_regret": 1.0,
            }
        )
        self.assertLess(result["reward"], 0.0)
        self.assertTrue(result["infrastructure_failure"])


if __name__ == "__main__":
    unittest.main()
