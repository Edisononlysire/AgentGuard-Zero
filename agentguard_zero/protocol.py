"""Version identifiers for fail-closed TMCD experiment artifacts."""

TMCD_PROTOCOL_VERSION = "tmcd-v2"
TMCD_RELEASE_REVISION = "tmcd-v2.4.2-20260716"

TASK_FAMILY_MAP = {
    "T1": "active_probe_ambiguity",
    "T2": "trust_betrayal",
    "T3": "profile_poisoning",
    "T4": "business_overresponse",
}
FAMILY_TASK_MAP = {family: task_id for task_id, family in TASK_FAMILY_MAP.items()}

# Scenario content is untrusted curriculum data. Runtime privilege must never be
# conveyed through generated or serialized scenario metadata.
PRIVILEGED_METADATA_FIELDS = frozenset(
    {"oracle_defender", "oracle_mode", "privileged"}
)

# DCA-generated events are untrusted claims, not verification results. Keep
# this vocabulary exact so curriculum data cannot inject tool evidence fields.
RAW_EVENT_ALLOWED_FIELDS = frozenset(
    {
        "event_id",
        "time",
        "type",
        "source_id",
        "source",
        "entity_id",
        "claim",
        "claim_semantics",
        "objective_hint",
        "truth_value",
        "is_fake",
        "spoofability",
    }
)
RAW_EVENT_FORBIDDEN_SIGNAL_FIELDS = frozenset(
    {
        "verdict",
        "consistency",
        "challenge_consistency",
        "source_reliability",
        "contradiction_risk",
        "canary_triggered",
        "leakage_risk",
        "verifier_id",
        "root_source_ids",
        "parent_evidence_ids",
        "evidence_id",
        "evidence_origin",
    }
)
RAW_EVENT_RESERVED_TYPES = frozenset(
    {
        "crosscheck",
        "provenancecheck",
        "graphquery",
        "businessimpactestimator",
        "sourcechallenge",
        "canaryprobe",
        "decoyprobe",
        "shadowactionprobe",
    }
)

EVIDENCE_ORIGIN_RAW_EVENT = "raw_event"
EVIDENCE_ORIGIN_TOOL_GENERATED = "tool_generated"


def task_id_from_focus(task_focus: str) -> str:
    task_id = str(task_focus).strip().split(maxsplit=1)[0].upper()
    return task_id if task_id in TASK_FAMILY_MAP else "unknown"
