"""Version identifiers for fail-closed TMCD experiment artifacts."""

TMCD_PROTOCOL_VERSION = "tmcd-v2"
TMCD_RELEASE_REVISION = "tmcd-v2.4.1-20260716"

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


def task_id_from_focus(task_focus: str) -> str:
    task_id = str(task_focus).strip().split(maxsplit=1)[0].upper()
    return task_id if task_id in TASK_FAMILY_MAP else "unknown"
