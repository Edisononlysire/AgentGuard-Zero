from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from agentguard_zero.defender_state.evidence_store import EvidenceStore
from agentguard_zero.defender_state.append_only_memory import AppendOnlyProfileMemory
from agentguard_zero.defender_state.memory_fsm import EvidenceStateMemory
from agentguard_zero.defender_state.retriever import retrieve_memory
from agentguard_zero.defender_state.trust_manager import ContextualTrustManager
from agentguard_zero.governance.authorization import (
    ResponseAuthorization,
    authorize_public_response,
)
from agentguard_zero.schemas.action_schema import OBJECTIVE_KEYS
from agentguard_zero.schemas.observation_schema_v4 import make_observation_v4
from agentguard_zero.schemas.scenario_schema_v2 import validate_scenario_v2
from agentguard_zero.runtime_policy import (
    HIGH_IMPACT_ACTIONS,
    TARGETED_RESPONSE_ACTIONS,
    TOOL_COSTS,
)
from agentguard_zero.tools.business_impact import estimate_business_impact
from agentguard_zero.tools.graph_query import graph_query
from agentguard_zero.tools.log_query import log_query
from agentguard_zero.variants import (
    ACTIVE_PROBE_TOOLS,
    PASSIVE_VERIFICATION_TOOLS,
    experiment_variant,
)
from agentguard_zero.world.public_projector import assert_public, project_event, project_public
from agentguard_zero.world.hidden_world import HiddenWorld
from agentguard_zero.world.oracle_ledger import OracleLedger


MITIGATING_ACTIONS = {"LimitSession", "ShadowBlock", "DeployDecoy", "Isolate"}


class CyberDefenseEnvV2:
    """TMCD-v2 symbolic environment with strict hidden/public state isolation."""

    protocol_version = "tmcd-v2"

    def __init__(self, scenario: dict[str, Any], max_steps: int | None = None) -> None:
        valid, reason = validate_scenario_v2(scenario)
        if not valid:
            raise ValueError(f"invalid TMCD-v2 scenario: {reason}")
        self.scenario = copy.deepcopy(scenario)
        latest_event_time = max(
            (int(event.get("time", -1)) for event in scenario.get("event_schedule", [])),
            default=-1,
        )
        configured_horizon = int(
            scenario.get("defense_constraints", {}).get("horizon", 0) or 0
        )
        requested_horizon = int(
            max_steps
            or configured_horizon
            or max(5, len(scenario.get("true_attack", {}).get("phase_schedule", [])) + 2)
        )
        self.max_steps = max(requested_horizon, latest_event_time + 2)
        self.hidden_world = HiddenWorld(scenario, horizon=self.max_steps)
        self.true_objective = self.hidden_world.true_objective
        self.t = 0
        self.evidence_store = EvidenceStore()
        self.trust_manager = ContextualTrustManager()
        self.experiment_variant = str(
            scenario.get("metadata", {}).get("experiment_variant", "full")
        )
        self.variant = experiment_variant(self.experiment_variant)
        self.state_layer_enabled = self.variant.state_layer
        self.memory = (
            AppendOnlyProfileMemory()
            if not self.variant.memory_lifecycle
            else EvidenceStateMemory()
        )
        self.history: list[dict[str, Any]] = []
        self.oracle_ledger = OracleLedger()
        self.business_cost = 0.0
        self.verification_cost = 0.0
        self.high_impact_count = 0
        self.attack_mitigated = False
        self.attack_success = False
        self.last_tool_result: dict[str, Any] | None = None
        self.public_probe_state: list[dict[str, Any]] = []
        self._observed_event_ids: set[str] = set()
        self._observed_internal_events: dict[str, dict[str, Any]] = {}
        self._last_retrieved_ids: set[str] = set()
        self._observation_cache: dict[int, dict[str, Any]] = {}
        self._source_priors = {
            str(profile["source_id"]): float(profile.get("public_prior", 0.5))
            for profile in scenario.get("source_profiles", [])
        }

    def _public_assets(self) -> list[dict[str, Any]]:
        return [
            {
                "id": str(asset.get("id", "")),
                "zone": str(asset.get("zone", "unknown")),
                "criticality": float(asset.get("criticality", 0.5)),
            }
            for asset in self.scenario.get("network_context", {}).get("assets", [])
            if str(asset.get("id", "")).strip()
        ]

    def _stable_draw(self, namespace: str, payload: Any) -> float:
        raw = json.dumps(
            {
                "namespace": namespace,
                "scenario": self.scenario.get("scenario_id", "unknown"),
                "time": self.t,
                "payload": payload,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        value = int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)
        return value / float(0xFFFFFFFFFFFFFFFF)

    def _noisy_negative_signal(
        self,
        event: dict[str, Any],
        *,
        tool: str,
        independent_sources: int = 0,
    ) -> bool:
        hidden_negative = not bool(event.get("truth_value", True))
        spoofability = max(0.0, min(1.0, float(event.get("spoofability", 0.5))))
        base_accuracy = {
            "CrossCheck": 0.70,
            "ProvenanceCheck": 0.72,
            "SourceChallenge": 0.76,
            "CanaryProbe": 0.70,
        }.get(tool, 0.65)
        accuracy = max(
            0.55,
            min(0.90, base_accuracy - 0.18 * spoofability + 0.05 * independent_sources),
        )
        correct = self._stable_draw(
            tool,
            {"event_id": event.get("event_id"), "sources": independent_sources},
        ) < accuracy
        return hidden_negative if correct else not hidden_negative

    def _resolve_due_probes(self) -> list[dict[str, Any]]:
        generated: list[dict[str, Any]] = []
        assets = {asset["id"]: asset for asset in self._public_assets()}
        target_zone = str(assets.get(self.hidden_world.target, {}).get("zone", "unknown"))
        active_phases = {"initial_access", "discovery", "lateral_movement", "collection"}
        for probe in self.public_probe_state:
            if probe.get("status") != "pending" or int(probe.get("effective_at", -1)) > self.t:
                continue
            zone = str(probe.get("zone", "unknown"))
            match = zone == target_zone or zone == self.hidden_world.target
            probability = 0.75 if match and self.hidden_world.current_phase in active_phases else 0.25
            interaction = self._stable_draw("DecoyProbe", probe.get("probe_id", "")) < probability
            probe["status"] = "resolved"
            probe["resolved_at"] = int(self.t)
            probe["result"] = "interaction_observed" if interaction else "no_interaction_observed"
            generated.append(
                {
                    "event_id": f"{probe.get('probe_id', 'probe')}-result",
                    "time": int(self.t),
                    "type": "decoy_probe_result",
                    "source_id": "decoy-sensor",
                    "entity_id": zone,
                    "claim": probe["result"],
                    "claim_semantics": {
                        "entity_id": zone,
                        "predicate": "decoy_interaction",
                        "object": probe["result"],
                        "scope": "active_probe",
                    },
                    "probe_id": str(probe.get("probe_id", "")),
                    "probe_generated": True,
                }
            )
        return generated

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
        internal_events = [*self._internal_events(), *self._resolve_due_probes()]
        public_events: list[dict[str, Any]] = []
        for internal in internal_events:
            internal_event_id = str(internal.get("event_id", ""))
            if internal_event_id:
                self._observed_internal_events.setdefault(
                    internal_event_id,
                    copy.deepcopy(internal),
                )
            public = project_event(internal)
            evidence_id = self.evidence_store.add_event(public, time=self.t)
            public["evidence_id"] = evidence_id
            source_id = str(public.get("source_id") or public.get("source") or "unknown")
            self.trust_manager.register_claim(
                public,
                time=self.t,
                public_prior=self._source_priors.get(source_id, 0.5),
            )
            public_events.append(public)
            event_id = str(internal.get("event_id", ""))
            if (
                event_id
                and event_id not in self._observed_event_ids
                and not internal.get("probe_generated")
            ):
                self.oracle_ledger.append(
                    {
                        "time": self.t,
                        "event_id": event_id,
                        "source_id": source_id,
                        "truth_value": bool(internal.get("truth_value", not internal.get("is_fake", False))),
                        "objective_hint": str(internal.get("objective_hint", "")),
                        "claim_semantics": copy.deepcopy(internal.get("claim_semantics", {})),
                    }
                )
                self._observed_event_ids.add(event_id)
        return public_events

    def observe(self) -> dict[str, Any]:
        cached = self._observation_cache.get(self.t)
        if cached is not None:
            return copy.deepcopy(cached)
        self.trust_manager.advance_time(self.t)
        events = self._register_current_events()
        retrieval = retrieve_memory(self.memory, events, time=self.t)
        self._last_retrieved_ids = set(retrieval.get("retrieved_memory_ids", []))
        constraints = self.scenario.get("defense_constraints", {})
        observation = make_observation_v4(
            time=self.t,
            observed_events=events,
            evidence_snapshot=self.evidence_store.public_snapshot(time=self.t),
            trust_snapshot=(self.trust_manager.public_snapshot() if self.state_layer_enabled else {}),
            memory_retrieval=(retrieval if self.state_layer_enabled else {}),
            public_assets=self._public_assets(),
            remaining_business_budget=float(constraints.get("business_budget", 5.0)) - self.business_cost,
            verification_remaining=float(constraints.get("verification_budget", 4)) - self.verification_cost,
            remaining_high_impact_actions=max(
                0,
                int(constraints.get("max_high_impact_actions", 1)) - self.high_impact_count,
            ),
            last_tool_result=self.last_tool_result,
            public_probe_state=self.public_probe_state,
        )
        assert_public(observation)
        self._observation_cache[self.t] = copy.deepcopy(observation)
        return copy.deepcopy(observation)

    def _find_internal_event(self, event_id: str, snapshot: list[dict[str, Any]]) -> dict[str, Any] | None:
        for event in snapshot:
            if str(event.get("event_id")) == str(event_id):
                return event
        return None

    def _execute_tool(self, tool_call: dict[str, Any], snapshot: list[dict[str, Any]]) -> dict[str, Any]:
        tool_call = tool_call if isinstance(tool_call, dict) else {"name": "None", "args": {}}
        name = str(tool_call.get("name", "None"))
        args = tool_call.get("args", {}) if isinstance(tool_call.get("args", {}), dict) else {}
        if name in ACTIVE_PROBE_TOOLS and not self.variant.active_probing:
            return {"tool": name, "error": "active_probing_disabled_by_ablation"}
        if name in PASSIVE_VERIFICATION_TOOLS and not self.variant.passive_verification:
            return {"tool": name, "error": "passive_verification_disabled_by_ablation"}
        if name != "None":
            tool_cost = float(TOOL_COSTS.get(name, 1.0))
            budget = float(self.scenario.get("defense_constraints", {}).get("verification_budget", 4.0))
            if self.verification_cost + tool_cost > budget + 1e-9:
                return {
                    "tool": name,
                    "status": "budget_exhausted",
                    "error": "verification_budget_exhausted",
                    "executed": False,
                    "cost": 0.0,
                }
            self.verification_cost += tool_cost
        requested_event_id = str(args.get("event_id", "unknown"))
        event = self._observed_internal_events.get(requested_event_id)
        if event is None:
            event = self._find_internal_event(requested_event_id, snapshot)
        event_tools = {"CrossCheck", "ProvenanceCheck", "SourceChallenge", "CanaryProbe"}
        event_available = requested_event_id in self._observed_internal_events
        if name in event_tools and not event_available:
            return {
                "tool": name,
                "event_id": requested_event_id,
                "status": "invalid_reference",
                "error": "event_not_available_in_current_snapshot",
                "evidence_created": False,
            }
        if name in event_tools and event is None:
            return {
                "tool": name,
                "event_id": requested_event_id,
                "status": "invalid_reference",
                "error": "unknown_event_id",
                "evidence_created": False,
            }
        if name == "LogQuery":
            return project_public(log_query(snapshot, source=args.get("source"), time=args.get("time")))
        if name == "CrossCheck":
            evidence_ids = sorted(
                {str(item) for item in args.get("evidence_ids", []) if str(item)}
            )
            if not evidence_ids:
                return {
                    "tool": name,
                    "event_id": requested_event_id,
                    "status": "invalid_reference",
                    "error": "crosscheck_requires_evidence_ids",
                    "evidence_created": False,
                }
            valid, reason = self.evidence_store.validate_refs(evidence_ids, time=self.t)
            if not valid:
                return {
                    "tool": name,
                    "event_id": requested_event_id,
                    "status": "invalid_reference",
                    "error": reason,
                    "evidence_created": False,
                }
            compatible, reason = self.evidence_store.refs_support_claim(
                evidence_ids,
                event.get("claim_semantics", {}),
                time=self.t,
            )
            if not compatible:
                return {
                    "tool": name,
                    "event_id": requested_event_id,
                    "status": "incompatible_evidence",
                    "error": reason,
                    "evidence_created": False,
                }
            event_source = str(event.get("source_id") or event.get("source") or "unknown")
            derived_roots = self.evidence_store.root_sources(evidence_ids, time=self.t)
            independent_sources = len(set(derived_roots) - {event_source})
            negative = self._noisy_negative_signal(
                event,
                tool=name,
                independent_sources=independent_sources,
            )
            return {
                "tool": name,
                "event_id": requested_event_id,
                "checked_evidence_ids": evidence_ids,
                "consistency_signal": "conflict" if negative else "support",
                "confidence_band": "medium",
                "verdict": "suspicious" if negative else "supported",
                "root_source_ids": derived_roots,
            }
        if name == "ProvenanceCheck":
            negative = self._noisy_negative_signal(event, tool=name)
            return {
                "tool": name,
                "event_id": requested_event_id,
                "provenance_signal": "anomalous" if negative else "plausible",
                "confidence_band": "medium",
                "verdict": "suspicious" if negative else "plausible",
            }
        if name == "GraphQuery":
            node = str(args.get("node", "")).strip()
            if not node:
                return {"tool": name, "error": "missing_required_arg:node"}
            if node not in {asset["id"] for asset in self._public_assets()}:
                return {"tool": name, "error": "unknown_public_target:node"}
            return project_public(graph_query(self.scenario, node))
        if name == "BusinessImpactEstimator":
            action = args.get("action", {"action": "Observe"})
            if not isinstance(action, dict):
                return {"tool": name, "error": "invalid_action_argument"}
            target = str(action.get("target", "")).strip()
            if not target:
                return {"tool": name, "error": "missing_required_arg:action.target"}
            if target not in {asset["id"] for asset in self._public_assets()}:
                return {"tool": name, "error": "unknown_public_target:action.target"}
            return project_public(estimate_business_impact(action, self._target_criticality(target)) | {"tool": name})
        if name == "SourceChallenge":
            negative = self._noisy_negative_signal(event, tool=name)
            return {
                "tool": name,
                "event_id": str(event.get("event_id", "unknown")),
                "source": str(event.get("source_id") or event.get("source") or "unknown"),
                "consistency_signal": "weak_conflict" if negative else "weak_support",
                "confidence_band": "medium",
                "verdict": "challenge_failed" if negative else "challenge_passed",
                "active_probe": True,
            }
        if name == "CanaryProbe":
            triggered = self._noisy_negative_signal(event, tool=name)
            return {
                "tool": name,
                "event_id": str(event.get("event_id", "unknown")),
                "source": str(event.get("source_id") or event.get("source") or "unknown"),
                "canary_triggered": triggered,
                "confidence_band": "medium",
                "verdict": "inconsistent" if triggered else "consistent",
                "active_probe": True,
            }
        if name == "DecoyProbe":
            zone = str(args.get("zone", "server"))
            public_zones = {asset["zone"] for asset in self._public_assets()}
            if zone not in public_zones:
                return {"tool": name, "error": "unknown_public_zone"}
            probe = {
                "probe_id": f"probe-{self.t}-{len(self.public_probe_state)}",
                "type": "decoy",
                "zone": zone,
                "effective_at": self.t + 1,
                "status": "pending",
            }
            self.public_probe_state.append(probe)
            return {"tool": name, "deployed": True, "active_probe": True, **probe}
        if name == "DeployDecoy":
            zone = str(args.get("zone", "server"))
            if zone not in {asset["zone"] for asset in self._public_assets()}:
                return {"tool": name, "error": "unknown_public_zone"}
            return {
                "tool": name,
                "deployed": True,
                "defensive_deployment": True,
                "zone": zone,
            }
        if name in {"ShadowActionProbe", "ShadowResponse"}:
            action = args.get("action", {"action": "Observe"})
            if not isinstance(action, dict):
                return {"tool": name, "error": "invalid_action_argument"}
            target = str(action.get("target", "")).strip()
            if not target:
                return {"tool": name, "error": "missing_required_arg:action.target"}
            if target not in {asset["id"] for asset in self._public_assets()}:
                return {"tool": name, "error": "unknown_public_target:action.target"}
            impact = estimate_business_impact(action, self._target_criticality(target))
            return project_public(impact) | {
                "tool": name,
                "shadow": True,
                "active_probe": name == "ShadowActionProbe",
                "overresponse_risk": min(1.0, float(impact.get("estimated_cost", 0.0)) / 2.0),
            }
        return {"tool": "None", "message": "no tool executed"}

    def _authorize_response(
        self,
        response: dict[str, Any],
        action_packet: dict[str, Any],
    ) -> ResponseAuthorization:
        constraints = self.scenario.get("defense_constraints", {})
        return authorize_public_response(
            response=response,
            evidence_assessment=action_packet.get("evidence_assessment", []),
            trust_snapshot=self.trust_manager.public_snapshot(),
            evidence_snapshot=self.evidence_store.public_snapshot(time=self.t),
            public_assets=self._public_assets(),
            business_state={
                "remaining_business_budget": (
                    float(constraints.get("business_budget", 5.0)) - self.business_cost
                ),
                "remaining_high_impact_actions": (
                    int(constraints.get("max_high_impact_actions", 1))
                    - self.high_impact_count
                ),
            },
            oracle_override=bool(
                self.scenario.get("metadata", {}).get("oracle_defender", False)
            ),
        )

    def step(self, action_packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
        action_packet = action_packet if isinstance(action_packet, dict) else {}
        # The rollout server recreates the environment after the initial
        # prompt was built; materialize the identical public state locally.
        self.observe()
        # Freeze s_t before any response or persistent probe effect is committed.
        snapshot = self._internal_events()
        proposed_response = (
            copy.deepcopy(action_packet.get("response", {}))
            if isinstance(action_packet.get("response", {}), dict)
            else {}
        )
        # Response authorization is a pre-state decision. Operations proposed
        # in this packet commit only after the response has executed.
        authorization = self._authorize_response(
            proposed_response,
            action_packet,
        )
        action_authorized = authorization.allowed
        authorization_reason = authorization.reason
        executed_response = (
            proposed_response
            if action_authorized
            else {"tier": "L0", "action": "Observe", "target": "none"}
        )
        target = str(executed_response.get("target", "none"))
        impact = estimate_business_impact(executed_response, self._target_criticality(target))
        cost = float(impact.get("estimated_cost", 0.0))
        self.business_cost += cost
        action = str(executed_response.get("action", "Observe"))
        if action in HIGH_IMPACT_ACTIONS:
            self.high_impact_count += 1
        belief = action_packet.get("belief", {}) if isinstance(action_packet.get("belief", {}), dict) else {}
        numeric_belief: dict[str, float] = {}
        for key in OBJECTIVE_KEYS:
            try:
                value = float(belief.get(key, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            numeric_belief[key] = value if math.isfinite(value) else 0.0
        top_belief = (
            max(OBJECTIVE_KEYS, key=numeric_belief.get)
            if any(value > 0.0 for value in numeric_belief.values())
            else "unknown"
        )
        belief_matches = top_belief == self.true_objective
        self.hidden_world.apply_response(
            action=action,
            belief_matches=belief_matches,
            target_matches=target == self.hidden_world.target,
            time=self.t,
        )
        self.attack_mitigated = self.hidden_world.mitigated

        if self.state_layer_enabled:
            trust_operations = action_packet.get("trust_operations", [])
            if self.variant.trust_recalibration:
                trust_events = self.trust_manager.apply(
                    trust_operations,
                    evidence_store=self.evidence_store,
                    time=self.t,
                )
            else:
                trust_events = [
                    {
                        "committed": False,
                        "op": str(operation.get("op", "")),
                        "reason": "trust_recalibration_disabled_by_ablation",
                        "time": int(self.t),
                    }
                    for operation in trust_operations
                    if isinstance(operation, dict)
                ]
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
        else:
            trust_events = []
            memory_events = []
            accepted_memory = []

        # Compute the probe after response submission, but only against the
        # frozen pre-response snapshot. Its evidence is unavailable until t+1.
        tool_result = self._execute_tool(action_packet.get("tool_call", {}), snapshot)
        parent_refs: list[str] = []
        event_id = str(tool_result.get("event_id", ""))
        parent = self.evidence_store.evidence_for_event(event_id) if event_id else None
        if parent:
            parent_refs.append(parent)
        parent_refs.extend(
            str(item)
            for item in tool_result.get("checked_evidence_ids", []) or []
            if str(item)
        )

        public_tool_result = project_public(tool_result)
        tool_name = str((action_packet.get("tool_call", {}) or {}).get("name", "None"))
        if tool_name == "None" or public_tool_result.get("error"):
            # A no-op or failed call is feedback, not independently referencable evidence.
            self.last_tool_result = (
                {"tool": "None"} if tool_name == "None" else public_tool_result
            )
        else:
            tool_evidence_id = self.evidence_store.add_tool_result(
                public_tool_result,
                time=self.t,
                parent_evidence_ids=parent_refs,
            )
            self.last_tool_result = public_tool_result | {
                "evidence_id": tool_evidence_id,
                "available_at": self.t + 1,
            }
        public_step = {
            "time": self.t,
            # The schema layer rejects forbidden fields. Project again here so
            # a direct environment caller can never turn malformed model data
            # into a public-state leak or a rollout-server 500 response.
            "action_packet": project_public(action_packet),
            "trust_events": trust_events,
            "memory_events": memory_events,
            "accepted_memory_ids": accepted_memory,
            "response_result": {
                "action": action,
                "target": target,
                "proposed_action": str(proposed_response.get("action", "Observe")),
                "proposed_target": str(proposed_response.get("target", "none")),
                "executed_action": action,
                "executed_target": target,
                "blocked": not action_authorized,
                "authorized": action_authorized,
                "authorization_reason": authorization_reason,
                "proposed_business_cost": float(authorization.estimated_cost),
                "business_cost": cost,
                "authorization_evidence_ids": list(authorization.evidence_ids),
            },
            "tool_result": copy.deepcopy(self.last_tool_result),
        }
        assert_public(public_step)
        self.history.append(public_step)
        self.hidden_world.advance_attack()
        self.attack_success = self.hidden_world.succeeded
        self.t += 1
        done = self.t >= self.max_steps or self.attack_mitigated or self.attack_success
        if done and not self.attack_mitigated:
            self.attack_success = True
            self.hidden_world.succeeded = True
        return ({} if done else self.observe()), copy.deepcopy(self.last_tool_result), done
