from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agentguard_zero.schemas.scenario_schema import minimal_example
from curriculum_train.examples.reward_function.vda_reward import compute_score


def main() -> None:
    scenario = minimal_example()
    good = json.dumps({
        "belief": {"exfiltration": 0.75, "sabotage": 0.1, "persistence": 0.1, "credential_theft": 0.05},
        "evidence_assessment": [],
        "trust_update": {"source_reliability": {}, "rationale": "initial environment evidence is not contradicted"},
        "uncertainty": 0.35,
        "tool_call": {"name": "None", "args": {}},
        "memory_update": {"profile_quarantine": [], "confirmed_profile": ["true-0"], "rejected_profile": []},
        "safety_check": {"business_risk": 0.2, "overresponse_risk": 0.1, "justification": "low-impact containment"},
        "response": {"tier": "L2", "action": "LimitSession", "target": "user_pc"},
    })
    bad = json.dumps({
        "belief": {"exfiltration": 0.05, "sabotage": 0.8, "persistence": 0.1, "credential_theft": 0.05},
        "evidence_assessment": [],
        "trust_update": {"source_reliability": {}, "rationale": "blind trust"},
        "uncertainty": 0.05,
        "tool_call": {"name": "MadeUpTool", "args": {}},
        "memory_update": {"profile_quarantine": [], "confirmed_profile": ["fake-0"], "rejected_profile": []},
        "safety_check": {"business_risk": 1.0, "overresponse_risk": 1.0, "justification": "panic"},
        "response": {"tier": "L3", "action": "Isolate", "target": "database"},
    })
    scores = compute_score([good, bad, "not json"], [json.dumps(scenario), json.dumps(scenario), json.dumps(scenario)])
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
