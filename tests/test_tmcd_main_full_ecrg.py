from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_tmcd_systems as ev
from agentguard_zero.inference_contract import (
    FORMAL_VDA_MAX_NEW_TOKENS,
    TRAINED_VDA_PROMPT_CONTRACT,
    require_candidate_quality,
    summarize_candidate_quality,
)


def frozen_config() -> dict:
    candidate_quality = summarize_candidate_quality(
        [[{"parse_ok": True, "base_admissible": True}] * 6]
    )
    return {
        "status": "frozen",
        "paper_name": "ECRG",
        "code_name": "V5-C",
        "candidate_count": 6,
        "feature_access": "public_evidence_trust_memory_business_only",
        "hidden_state_access": False,
        "parameter_training": False,
        "vda_parameter_update": False,
        "dca_parameter_update": False,
        "prompt_contract": TRAINED_VDA_PROMPT_CONTRACT,
        "max_new_tokens": FORMAL_VDA_MAX_NEW_TOKENS,
        "candidate_quality": candidate_quality,
        "parameters": {
            "hard_admission_thresholds": {
                "min_supported_score": 0.85,
                "max_high_impact_risk": 1.2,
                "max_business_fraction": 0.75,
                "max_high_impact_uncertainty": 0.6,
            },
            "ranking_weights": {
                "legal": 8.0,
                "evidence": 1.0,
                "verification": 3.5,
                "active_probe": 2.0,
                "quarantine": 2.0,
                "promotion_risk": 3.0,
                "mitigation": 1.0,
                "business": 3.5,
                "overresponse": 4.0,
                "delay": 0.35,
                "betrayal": 2.5,
                "probe_robust": 1.5,
                "poisoning": 2.5,
                "quarantine_robust": 1.0,
                "benign_high_impact": 2.5,
                "reversible": 1.0,
            },
            "safe_probe_fallback_trigger": {"score_floor": 2.0, "risk_floor": 0.8},
        },
    }


def synthetic_result(task: str, value: float) -> dict:
    metrics = {
        "safe_utility": value,
        "attack_mitigation": 0.5,
        "betrayal_detection": value if task.startswith("T2 ") else math.nan,
        "poison_success": value,
        "business_cost": value,
        "verification_cost": 0.0,
        "probe_yield": value,
        "probe_result_availability": value,
        "decision_delay": 0.0,
        "normalized_episode_length": 1.0,
        "false_containment": 0.0,
        "excess_severity": 0.0,
        "memory_recovery": 0.0,
        "trust_brier": 0.0,
        "overresponse": value,
    }
    return {
        "task": task,
        "tmcd_metrics": metrics,
        "score": {},
        "selected_actions": [],
        "selected_json_parse_failures": 0,
        "selected_invalid_tool_calls": 0,
        "selected_invalid_response_actions": 0,
        "steps": 1,
    }


class TMCDMainFullECRGTests(unittest.TestCase):
    def test_frozen_ecrg_config_loader_enforces_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ecrg.json"
            path.write_text(json.dumps(frozen_config()), encoding="utf-8")
            loaded = ev.load_frozen_ecrg_config(path)
            self.assertEqual(loaded["candidate_count"], 6)
            self.assertIs(loaded["hidden_state_access"], False)
            self.assertEqual(loaded["_sha256"], ev.sha256_file(path))

            invalid = frozen_config()
            invalid["candidate_count"] = 4
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate_count"):
                ev.load_frozen_ecrg_config(path)

    def test_summary_uses_task_specific_metric_scopes(self) -> None:
        results = [
            synthetic_result("T1 Active Probing Defense", 0.1),
            synthetic_result("T2 Trust-Building Betrayal", 0.2),
            synthetic_result("T3 Profile / Memory Poisoning", 0.3),
            synthetic_result("T4 Business-Constrained Overreaction", 0.4),
        ]
        args = SimpleNamespace(
            run_name="unit",
            system="agentguard_zero_full",
            model_backend="mock",
            model_path="",
            adapter_path="",
            api_model="",
            candidate_count=6,
            selector_mode="v5_c_evidence_governor",
            offset=0,
            ecrg_config="/frozen/ecrg.json",
            ecrg_config_sha256="abc",
        )
        summary = ev.summarize(results, args)
        self.assertAlmostEqual(summary["probe_yield"], 0.1)
        self.assertAlmostEqual(summary["betrayal_detection"], 0.2)
        self.assertAlmostEqual(summary["poison_success"], 0.3)
        self.assertAlmostEqual(summary["overresponse_rate"], 0.4)
        self.assertAlmostEqual(summary["safe_utility"], 0.25)
        self.assertAlmostEqual(summary["business_cost"], 0.25)

    def test_full_selector_requires_frozen_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen ECRG"):
            ev.select_runtime_candidate(
                "agentguard_zero_full",
                {"observation": {"protocol_version": "tmcd-v2"}},
                ["{}"] * 6,
                "agentguard_zero_select",
                selector_mode="v5_c_evidence_governor",
                ecrg_config=None,
            )

    def test_trained_vda_preserves_exact_training_row_prompt(self) -> None:
        messages = [{"role": "user", "content": "exact training row"}]
        self.assertEqual(
            ev.apply_inference_prompt_contract(
                messages,
                "agentguard_zero_train",
            ),
            messages,
        )
        self.assertEqual(
            ev.apply_inference_prompt_contract(
                messages,
                "agentguard_zero_full",
            ),
            messages,
        )

        baseline = ev.apply_inference_prompt_contract(
            messages,
            "qwen_zero_shot_vda",
        )
        self.assertEqual(baseline[0]["role"], "system")
        self.assertEqual(baseline[1:], messages)

    def test_candidate_quality_gate_rejects_schema_collapse(self) -> None:
        accepted = summarize_candidate_quality(
            [[{"parse_ok": True, "base_admissible": True}] * 6 for _ in range(20)]
        )
        require_candidate_quality(accepted, context="unit")

        collapsed = summarize_candidate_quality(
            [
                [{"parse_ok": False, "base_admissible": False}] * 6
                for _ in range(20)
            ]
        )
        self.assertIs(collapsed["accepted"], False)
        self.assertIn("candidate_parse_ok_rate", collapsed["failures"])
        with self.assertRaisesRegex(ValueError, "candidate-quality gate failed"):
            require_candidate_quality(collapsed, context="unit")


if __name__ == "__main__":
    unittest.main()
