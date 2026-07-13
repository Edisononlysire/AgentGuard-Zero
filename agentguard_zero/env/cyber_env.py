from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from agentguard_zero.memory.profile_memory import init_memory, update_memory
from agentguard_zero.schemas.observation_schema import make_observation
from agentguard_zero.tools.business_impact import estimate_business_impact
from agentguard_zero.tools.cross_check import cross_check
from agentguard_zero.tools.graph_query import graph_query
from agentguard_zero.tools.log_query import log_query
from agentguard_zero.tools.provenance import provenance_check


class CyberProfilePoisoningEnv:
    """A safe abstract cyber environment.

    It simulates symbolic attack phases, symbolic poisoned observations, verification tools,
    profile memory, and business cost. It never executes payloads or real exploit logic.
    """

    def __init__(self, scenario: Dict[str, Any], max_steps: int | None = None):
        self.scenario = copy.deepcopy(scenario)
        self.true_objective = scenario["oracle"]["true_objective"]
        self.false_objective = scenario["poisoning_plan"]["false_objective"]
        self.phase_schedule = scenario["true_attack"].get("phase_schedule", [])
        self.max_steps = max_steps or max(5, len(self.phase_schedule) + 2)
        self.t = 0
        self.memory = init_memory()
        self.history: List[Dict[str, Any]] = []
        self.business_cost = 0.0
        self.verification_cost = 0.0
        self.high_impact_count = 0
        self.attack_mitigated = False
        self.attack_success = False
        self.last_tool_result: Dict[str, Any] | None = None
        self.active_decoys: List[Dict[str, Any]] = []

    def _target_criticality(self, target: str) -> float:
        for a in self.scenario.get("network_context", {}).get("assets", []):
            if a.get("id") == target:
                return float(a.get("criticality", 0.5))
        return 0.5

    def _true_event(self) -> Dict[str, Any]:
        phase = self.phase_schedule[min(self.t, max(len(self.phase_schedule) - 1, 0))] if self.phase_schedule else "discovery"
        return {
            "event_id": f"true-{self.t}",
            "time": self.t,
            "type": phase,
            "source": "environment",
            "claim": f"activity consistent with {self.true_objective}:{phase}",
            "objective_hint": self.true_objective,
            "is_fake": False,
            "spoofability": 0.1,
        }

    def _fake_events(self) -> List[Dict[str, Any]]:
        events = []
        for idx, ev in enumerate(self.scenario.get("poisoning_plan", {}).get("fake_evidence", [])):
            if int(ev.get("time", -1)) == self.t:
                e = dict(ev)
                e["event_id"] = e.get("event_id", f"fake-{self.t}-{idx}")
                e["objective_hint"] = e.get("intended_misbelief", self.false_objective)
                e["is_fake"] = True
                events.append(e)
        for idx, ev in enumerate(self.scenario.get("poisoning_plan", {}).get("decoy_behavior", [])):
            if int(ev.get("time", -1)) == self.t:
                events.append({
                    "event_id": f"decoy-{self.t}-{idx}",
                    "time": self.t,
                    "type": "decoy_touch",
                    "source": "decoy_sensor",
                    "claim": f"decoy touched: {ev.get('decoy_type', 'generic_decoy')}",
                    "objective_hint": ev.get("intended_misbelief", self.false_objective),
                    "is_fake": True,
                    "spoofability": 0.7,
                })
        for idx, ev in enumerate(self.active_decoys):
            if int(ev.get("deploy_time", -99)) + 1 == self.t:
                events.append({
                    "event_id": f"probe-decoy-{self.t}-{idx}",
                    "time": self.t,
                    "type": "decoy_probe_result",
                    "source": "decoy_sensor",
                    "claim": f"symbolic decoy probe touched: {ev.get('decoy_type', 'generic_decoy')}",
                    "objective_hint": self.true_objective,
                    "is_fake": False,
                    "spoofability": 0.2,
                    "probe_origin": "DecoyProbe",
                })
        return events

    def current_events(self) -> List[Dict[str, Any]]:
        return [self._true_event()] + self._fake_events()

    def observe(self) -> Dict[str, Any]:
        budget = float(self.scenario.get("defense_constraints", {}).get("business_budget", 5.0)) - self.business_cost
        obs = make_observation(self.t, self.current_events(), budget, self.memory)
        obs["last_tool_result"] = self.last_tool_result
        return obs

    def _find_event(self, event_id: str) -> Dict[str, Any]:
        for item in self.current_events():
            if item.get("event_id") == event_id:
                return item
        return {"event_id": event_id, "source": "unknown", "spoofability": 0.5, "is_fake": False}

    def _safe_action(self, action: Any) -> Dict[str, Any]:
        return action if isinstance(action, dict) else {"action": "Observe", "target": "none"}

    def _source_challenge(self, args: Dict[str, Any]) -> Dict[str, Any]:
        event = self._find_event(args.get("event_id", "unknown"))
        spoof = float(event.get("spoofability", 0.5))
        reliability = max(0.0, min(1.0, 1.0 - spoof))
        contradiction = max(0.0, min(1.0, spoof + (0.2 if event.get("is_fake") else -0.1)))
        return {
            "tool": "SourceChallenge",
            "event_id": event.get("event_id", "unknown"),
            "source": args.get("source", event.get("source", "unknown")),
            "challenge_consistency": reliability,
            "contradiction_risk": contradiction,
            "verdict": "challenge_failed" if contradiction >= 0.6 else "challenge_passed",
            "active_probe": True,
        }

    def _canary_probe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        event = self._find_event(args.get("event_id", "unknown"))
        spoof = float(event.get("spoofability", 0.5))
        triggered = bool(event.get("is_fake") or spoof >= 0.6)
        return {
            "tool": "CanaryProbe",
            "event_id": event.get("event_id", "unknown"),
            "source": args.get("source", event.get("source", "unknown")),
            "canary_type": args.get("canary_type", "symbolic_consistency_canary"),
            "canary_triggered": triggered,
            "leakage_risk": max(0.0, min(1.0, spoof)),
            "verdict": "source_reacted_to_canary" if triggered else "no_canary_reaction",
            "active_probe": True,
        }

    def _decoy_probe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        decoy = {
            "deploy_time": self.t,
            "zone": args.get("zone", "server"),
            "decoy_type": args.get("decoy_type", "generic_decoy"),
        }
        self.active_decoys.append(decoy)
        return {
            "tool": "DecoyProbe",
            "deployed": True,
            "zone": decoy["zone"],
            "decoy_type": decoy["decoy_type"],
            "expected_observation": "decoy_probe_result_next_turn",
            "active_probe": True,
        }

    def _shadow_action_probe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = self._safe_action(args.get("action", {"action": "Observe"}))
        target = action.get("target", self.scenario["true_attack"].get("target", "unknown"))
        impact = estimate_business_impact(action, self._target_criticality(target))
        cost = float(impact.get("estimated_cost", 0.0))
        return impact | {
            "tool": "ShadowActionProbe",
            "shadow": True,
            "active_probe": True,
            "overresponse_risk": min(1.0, cost / 2.0),
            "verdict": "high_cost_shadow_action" if cost >= 1.0 else "bounded_shadow_action",
        }

    def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        tool_call = tool_call if isinstance(tool_call, dict) else {"name": "None", "args": {}}
        name = tool_call.get("name", "None")
        args = tool_call.get("args", {}) or {}
        args = args if isinstance(args, dict) else {}
        self.verification_cost += 1.0 if name not in {"None", None} else 0.0
        if name == "LogQuery":
            return log_query(self.current_events(), source=args.get("source"), time=args.get("time"))
        if name == "CrossCheck":
            return cross_check(self._find_event(args.get("event_id", "unknown")), args.get("sources", []))
        if name == "ProvenanceCheck":
            return provenance_check(self._find_event(args.get("event_id", "unknown")))
        if name == "GraphQuery":
            return graph_query(self.scenario, args.get("node", self.scenario["true_attack"].get("target", "unknown")))
        if name == "BusinessImpactEstimator":
            action = self._safe_action(args.get("action", {"action": "Observe"}))
            target = action.get("target", self.scenario["true_attack"].get("target", "unknown"))
            return estimate_business_impact(action, self._target_criticality(target))
        if name == "SourceChallenge":
            return self._source_challenge(args)
        if name == "CanaryProbe":
            return self._canary_probe(args)
        if name == "DecoyProbe":
            return self._decoy_probe(args)
        if name == "ShadowActionProbe":
            return self._shadow_action_probe(args)
        if name == "DeployDecoy":
            result = self._decoy_probe(args)
            result["tool"] = "DeployDecoy"
            return result
        if name == "ShadowResponse":
            result = self._shadow_action_probe(args)
            result["tool"] = "ShadowResponse"
            return result
        return {"tool": "None", "message": "no tool executed"}

    def step(self, action_packet: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        action_packet = action_packet if isinstance(action_packet, dict) else {}
        tool_result = self.execute_tool(action_packet.get("tool_call", {"name": "None", "args": {}}))
        self.last_tool_result = tool_result
        self.memory = update_memory(self.memory, action_packet.get("memory_update", {}))
        response = self._safe_action(action_packet.get("response", {"action": "Observe", "target": "none"}))
        impact = estimate_business_impact(response, self._target_criticality(response.get("target", self.scenario["true_attack"].get("target", "unknown"))))
        cost = float(impact.get("estimated_cost", 0.0))
        self.business_cost += cost
        if response.get("action") in {"Isolate", "Restore", "Remove"}:
            self.high_impact_count += 1
        belief = action_packet.get("belief", {})
        if belief and max(belief, key=lambda k: float(belief.get(k, 0.0))) == self.true_objective and response.get("action") in {"LimitSession", "ShadowBlock", "DeployDecoy", "Isolate"}:
            self.attack_mitigated = True
        self.history.append({
            "t": self.t,
            "observation": self.observe(),
            "action_packet": action_packet,
            "tool_result": tool_result,
            "business_cost_step": cost,
            "memory": copy.deepcopy(self.memory),
        })
        self.t += 1
        done = self.t >= self.max_steps or self.attack_mitigated
        if done and not self.attack_mitigated:
            self.attack_success = True
        return self.observe() if not done else {}, tool_result, done
