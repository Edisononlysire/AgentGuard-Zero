from __future__ import annotations

import copy
from typing import Any

from agentguard_zero.defender_state.evidence_store import EvidenceStore
from agentguard_zero.defender_state.append_only_memory import AppendOnlyProfileMemory
from agentguard_zero.defender_state.memory_fsm import EvidenceStateMemory
from agentguard_zero.defender_state.retriever import retrieve_memory
from agentguard_zero.defender_state.trust_manager import ContextualTrustManager
from agentguard_zero.schemas.observation_schema_v4 import make_observation_v4
from agentguard_zero.schemas.scenario_schema_v2 import validate_scenario_v2
from agentguard_zero.tools.business_impact import estimate_business_impact
from agentguard_zero.tools.cross_check import cross_check
from agentguard_zero.tools.graph_query import graph_query
from agentguard_zero.tools.log_query import log_query
from agentguard_zero.tools.provenance import provenance_check
from agentguard_zero.world.public_projector import assert_public, project_event, project_public


HIGH_IMPACT_ACTIONS = {"Isolate", "Restore", "Remove"}
MITIGATING_ACTIONS = {"LimitSession", "ShadowBlock", "DeployDecoy", "Isolate"}


class CyberDefenseEnvV2:
    """TMCD-v2 symbolic environment with strict hidden/public state isolation."""

    protocol_version = "tmcd-v2"

    def __init__(self, scenario: dict[str, Any], max_steps: int | None = None) -> None:
        valid, reason = validate_scenario_v2(scenario)
        if not valid:
            raise ValueError(f"invalid TMCD-v2 scenario: {reason}")
        self.scenario = copy.deepcopy(scenario)
        self.true_objective = str(scenario["oracle"]["true_objective"])
        self.max_steps = int(max_steps or max(5, len(scenario.get("true_attack", {}).get("phase_schedule", [])) + 2))
        self.t = 0
        self.evidence_store = EvidenceStore()
        self.trust_manager = ContextualTrustManager()
        self.experiment_variant = str(
            scenario.get("metadata", {}).get("experiment_variant", "full")
        )
        self.memory = (
            AppendOnlyProfileMemory()
            if self.experiment_variant == "append_only_memory"
            else EvidenceStateMemory()
        )
        self.history: list[dict[str, Any]] = []
        self.oracle_ledger: list[dict[str, Any]] = []
        self.business_cost = 0.0
        self.verification_cost = 0.0
        self.high_impact_count = 0
        self.attack_mitigated = False
        self.attack_success = False
        self.last_tool_result: dict[str, Any] | None = None
        self.public_probe_state: list[dict[str, Any]] = []
        self._observed_times: set[int] = set()
        self._last_retrieved_ids: set[str] = set()
        self._observation_cache: dict[int, dict[str, Any]] = {}
        self._source_priors = {
            str(profile["source_id"]): float(profile.get("public_prior", 0.5))
            for profile in scenario.get("source_profiles", [])
        }

    def _target_criticality(self, target: str) -> float:
        for asset in self.scenario.get("network_context", {}).get("assets", []):
            if asset.get("id") == target:
                return float(asset.get("criticality", 0.5))
        return 0.5

    def _internal_events(self, *, time: int | None = None) -> list[dict[str, Any]]:
        current = self.t if time is None else int(time)
        return [
            copy.deepcopy(event)
            for event in self.scenario.get("event_schedule", [])
            if int(event.get("time", -1)) == current
        ]

    def _register_current_events(self) -> list[dict[str, Any]]:
        internal_events = self._internal_events()
        public_events: list[dict[str, Any]] = []
        for internal in internal_events:
            public = project_event(internal)
            evidence_id = self.evidence_store.add_event(public, time=self.t)
            public["evidence_id"] = evidence_id
            source_id = str(public.get("source_id") or public.get("source") or "unknown")
            self.trust_manager.ensure_source(source_id, public_prior=self._source_priors.get(source_id, 0.5), time=self.t)
            self.trust_manager.register_claim(public, time=self.t)
            public_events.append(public)
            if self.t not in self._observed_times:
                self.oracle_ledger.append(
                    {
                        "time": self.t,
                        "event_id": str(internal.get("event_id", "")),
                        "source_id": source_id,
                        "truth_value": bool(internal.get("truth_value", not internal.get("is_fake", False))),
                        "objective_hint": str(internal.get("objective_hint", "")),
                        "claim_semantics": copy.deepcopy(internal.get("claim_semantics", {})),
                    }
                )
        self._observed_times.add(self.t)
        return public_events

    def observe(self) -> dict[str, Any]:
        cached = self._observation_cache.get(self.t)
        if cached is not None:
            return copy.deepcopy(cached)
        events = self._register_current_events()
        retrieval = retrieve_memory(self.memory, events, time=self.t)
        self._last_retrieved_ids = set(retrieval.get("retrieved_memory_ids", []))
        constraints = self.scenario.get("defense_constraints", {})
        observation = make_observation_v4(
            time=self.t,
            observed_events=events,
            evidence_snapshot=self.evidence_store.public_snapshot(time=self.t),
            trust_snapshot=self.trust_manager.public_snapshot(),
            memory_retrieval=retrieval,
            remaining_business_budget=float(constraints.get("business_budget", 5.0)) - self.business_cost,
            verification_remaining=float(constraints.get("verification_budget", 4)) - self.verification_cost,
            last_tool_result=self.last_tool_result,
            public_probe_state=self.public_probe_state,
        )
        assert_public(observation)
        self._observation_cache[self.t] = copy.deepcopy(observation)
        return copy.deepcopy(observation)

    def _find_internal_event(self, event_id: str, snapshot: list[dict[str, Any]]) -> dict[str, Any]:
        for event in snapshot:
            if str(event.get("event_id")) == str(event_id):
                return event
        return {
            "event_id": str(event_id),
            "source_id": "unknown",
            "source": "unknown",
            "spoofability": 0.5,
            "truth_value": True,
        }

    def _execute_tool(self, tool_call: dict[str, Any], snapshot: list[dict[str, Any]]) -> dict[str, Any]:
        tool_call = tool_call if isinstance(tool_call, dict) else {"name": "None", "args": {}}
        name = str(tool_call.get("name", "None"))
        args = tool_call.get("args", {}) if isinstance(tool_call.get("args", {}), dict) else {}
        if name != "None":
            self.verification_cost += 1.0
        event = self._find_internal_event(str(args.get("event_id", "unknown")), snapshot)
        if name == "LogQuery":
            return project_public(log_query(snapshot, source=args.get("source"), time=args.get("time")))
        if name == "CrossCheck":
            return project_public(cross_check(event, args.get("sources", [])))
        if name == "ProvenanceCheck":
            return project_public(provenance_check(event))
        if name == "GraphQuery":
            return project_public(graph_query(self.scenario, args.get("node", self.scenario["true_attack"]["target"])))
        if name == "BusinessImpactEstimator":
            action = args.get("action", {"action": "Observe"})
            target = str(action.get("target", self.scenario["true_attack"]["target"]))
            return project_public(estimate_business_impact(action, self._target_criticality(target)) | {"tool": name})
        if name == "SourceChallenge":
            spoof = float(event.get("spoofability", 0.5))
            contradiction = max(0.0, min(1.0, spoof + (0.2 if not event.get("truth_value", True) else -0.1)))
            return {
                "tool": name,
                "event_id": str(event.get("event_id", "unknown")),
                "source": str(event.get("source_id") or event.get("source") or "unknown"),
                "challenge_consistency": max(0.0, min(1.0, 1.0 - spoof)),
                "contradiction_risk": contradiction,
                "verdict": "challenge_failed" if contradiction >= 0.6 else "challenge_passed",
                "active_probe": True,
            }
        if name == "CanaryProbe":
            spoof = float(event.get("spoofability", 0.5))
            triggered = bool(not event.get("truth_value", True) or spoof >= 0.6)
            return {
                "tool": name,
                "event_id": str(event.get("event_id", "unknown")),
                "source": str(event.get("source_id") or event.get("source") or "unknown"),
                "canary_triggered": triggered,
                "leakage_risk": max(0.0, min(1.0, spoof)),
                "verdict": "source_reacted_to_canary" if triggered else "no_canary_reaction",
                "active_probe": True,
            }
        if name in {"DecoyProbe", "DeployDecoy"}:
            probe = {
                "probe_id": f"probe-{self.t}-{len(self.public_probe_state)}",
                "type": "decoy",
                "zone": str(args.get("zone", "server")),
                "effective_at": self.t + 1,
            }
            self.public_probe_state.append(probe)
            return {"tool": name, "deployed": True, "active_probe": True, **probe}
        if name in {"ShadowActionProbe", "ShadowResponse"}:
            action = args.get("action", {"action": "Observe"})
            target = str(action.get("target", self.scenario["true_attack"]["target"]))
            impact = estimate_business_impact(action, self._target_criticality(target))
            return project_public(impact) | {
                "tool": name,
                "shadow": True,
                "active_probe": True,
                "overresponse_risk": min(1.0, float(impact.get("estimated_cost", 0.0)) / 2.0),
            }
        return {"tool": "None", "message": "no tool executed"}

    def step(self, action_packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
        action_packet = action_packet if isinstance(action_packet, dict) else {}
        # The rollout server recreates the environment after the initial
        # prompt was built; materialize the identical public state locally.
        self.observe()
        # Freeze s_t before any response or persistent probe effect is committed.
        snapshot = self._internal_events()
        trust_events = self.trust_manager.apply(
            action_packet.get("trust_operations", []),
            evidence_store=self.evidence_store,
            time=self.t,
        )
        memory_events = self.memory.apply(
            action_packet.get("memory_operations", []),
            evidence_store=self.evidence_store,
            trust_manager=self.trust_manager,
            time=self.t,
        )
        accepted_memory = self.memory.record_usage(
            action_packet.get("memory_usage", []),
            retrieved_ids=self._last_retrieved_ids,
            time=self.t,
        )

        response = action_packet.get("response", {}) if isinstance(action_packet.get("response", {}), dict) else {}
        target = str(response.get("target", self.scenario["true_attack"]["target"]))
        impact = estimate_business_impact(response, self._target_criticality(target))
        cost = float(impact.get("estimated_cost", 0.0))
        self.business_cost += cost
        action = str(response.get("action", "Observe"))
        if action in HIGH_IMPACT_ACTIONS:
            self.high_impact_count += 1
        belief = action_packet.get("belief", {}) if isinstance(action_packet.get("belief", {}), dict) else {}
        top_belief = max(belief, key=lambda key: float(belief.get(key, 0.0))) if belief else "unknown"
        if top_belief == self.true_objective and action in MITIGATING_ACTIONS:
            self.attack_mitigated = True

        # Compute the probe after response submission, but only against the
        # frozen pre-response snapshot. Its evidence is unavailable until t+1.
        tool_result = self._execute_tool(action_packet.get("tool_call", {}), snapshot)
        parent_refs: list[str] = []
        event_id = str(tool_result.get("event_id", ""))
        parent = self.evidence_store.evidence_for_event(event_id) if event_id else None
        if parent:
            parent_refs.append(parent)

        tool_evidence_id = self.evidence_store.add_tool_result(
            project_public(tool_result),
            time=self.t,
            parent_evidence_ids=parent_refs,
        )
        self.last_tool_result = project_public(tool_result) | {
            "evidence_id": tool_evidence_id,
            "available_at": self.t + 1,
        }
        action_authorized = True
        if action in HIGH_IMPACT_ACTIONS:
            assessed = [
                self.trust_manager.claim_for(str(item.get("event_id", "")))
                for item in action_packet.get("evidence_assessment", [])
                if isinstance(item, dict)
            ]
            best_claim = max((float(item.get("score", 0.0)) for item in assessed if item), default=0.0)
            action_authorized = best_claim >= 0.80

        public_step = {
            "time": self.t,
            "action_packet": copy.deepcopy(action_packet),
            "trust_events": trust_events,
            "memory_events": memory_events,
            "accepted_memory_ids": accepted_memory,
            "response_result": {"action": action, "target": target, "business_cost": cost, "authorized": action_authorized},
            "tool_result": copy.deepcopy(self.last_tool_result),
        }
        assert_public(public_step)
        self.history.append(public_step)
        self.t += 1
        done = self.t >= self.max_steps or self.attack_mitigated
        if done and not self.attack_mitigated:
            self.attack_success = True
        return ({} if done else self.observe()), copy.deepcopy(self.last_tool_result), done
