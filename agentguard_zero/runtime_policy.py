"""Shared public execution constraints for TMCD-v2."""

HIGH_IMPACT_ACTIONS = frozenset({"Isolate", "Restore", "Remove"})
TARGETED_RESPONSE_ACTIONS = frozenset(
    {"DeployDecoy", "LimitSession", "ShadowBlock", "Isolate", "Restore", "Remove"}
)
TOOL_COSTS = {
    "LogQuery": 1.0,
    "CrossCheck": 1.0,
    "ProvenanceCheck": 1.0,
    "GraphQuery": 1.0,
    "BusinessImpactEstimator": 1.0,
    "SourceChallenge": 1.5,
    "CanaryProbe": 2.0,
    "DecoyProbe": 2.5,
    "ShadowActionProbe": 2.5,
    "DeployDecoy": 2.0,
    "ShadowResponse": 2.0,
}
