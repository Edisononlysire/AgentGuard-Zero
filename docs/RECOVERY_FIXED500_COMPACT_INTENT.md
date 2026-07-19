# Fixed-500 Compact-Intent VDA Recovery Pilot

## Outcome

The fixed-500 pilot establishes one positive result and two useful negative
results. A Qwen3.5-4B VDA can learn executable defensive behavior when the
learned output is reduced to one short, public-only action intent and a
deterministic compiler supplies Action Schema v4 boilerplate. The selected
VDA mitigated 48.5% of attacks on the complete 200-scenario xplay set, versus
0% for an all-`Observe` policy on the identical scenarios.

This is evidence that the VDA training problem is recoverable. It is not yet a
paper-ready final policy: overresponse is 47%, T4 mitigation is 0%, and the
sealed TMCD-Test was not used.

## Simplified architecture

The model receives the existing public decision instance and returns exactly
one of six short intent forms:

```text
tool | trust | memory | memory_use | response | observe
```

The model never emits the full state/action packet. A deterministic compiler
fills fixed protocol fields and rejects illegal tools, actions, argument
shapes, or response tiers. Hidden state and Oracle truth are not part of the
model prompt. The pilot uses no ECRG, no DCA update, no PPO, no co-evolution,
and no model-performance-based scenario selection.

This design removes the conflict in the previous target: the prompt required
an exact full-schema key order while the training target moved the action to a
different position. It also stops token-average SFT from spending most of its
capacity reproducing invariant schema/state boilerplate instead of learning
the semantic decision.

## Three variants

All variants started from Qwen3.5-4B or the explicitly recorded parent adapter
and used the same fixed Round-1 source. The quick comparison used the same
task-balanced eight xplay scenarios, K=1 greedy decoding, and no ECRG.

| Variant | Training | Valid | Mitigation | Overresponse | Safe utility | Verdict |
|---|---|---:|---:|---:|---:|---|
| V1 compact intent SFT | 2,317 sequence-balanced Teacher records, 3 epochs | 100.0% | 62.5% | 62.5% | -0.153 | Selected |
| V2 balanced compact SFT | 3,000 balanced Teacher records, 2 epochs | 70.2% | 25.0% | 50.0% | -0.815 | Rejected: invalid/hallucinated actions |
| V3 bidirectional DPO | V1 + 2,130 public-action preference pairs, 1 epoch | 100.0% | 0.0% | 0.0% | -1.061 | Rejected: collapsed to `Observe` |

V3 is an important negative result. Of the 2,130 preference pairs, 1,697
preferred `Observe`; a single DPO epoch was enough to erase the useful action
support learned by V1. Lower preference loss was therefore not evidence of a
better cyber-defense policy.

## Complete 200-scenario evaluation of V1

Job `344526` evaluated all 200 fixed xplay scenarios (T1–T4: 50 each). It used
K=1 greedy inference with thinking disabled. All four shards completed without
Traceback, OOM, NCCL, or DataProto errors.

| Scope | Valid | Mitigation | Overresponse | Business cost | Safe utility |
|---|---:|---:|---:|---:|---:|
| Overall | 94.45% | 48.5% | 47.0% | 1.765 | -0.382 |
| T1 | 96.89% | 72.0% | 50.0% | 2.122 | 0.098 |
| T2 | 100.0% | 76.0% | 52.0% | 2.179 | 0.172 |
| T3 | 94.46% | 46.0% | 48.0% | 2.219 | -0.432 |
| T4 | 88.35% | 0.0% | 38.0% | 0.538 | -1.367 |
| All-`Observe` baseline | 100.0% | 0.0% | 0.0% | 0.000 | -1.058 |

The aggregate Safe Utility improvement over all-`Observe` is +0.676 on the
same 200 scenarios. This is a meaningful policy improvement, but the action
distribution is still too aggressive: 84.1% of decisions are mitigation
actions. The next optimization target is a constrained reduction in
overresponse that preserves action support, with special attention to T4.

## Reproducibility

- Fixed train parquet SHA256:
  `16b7b2e8e77c8186b71d118995d60a75b1d46c63487c71b9976387a90dc56487`
- Fixed xplay parquet SHA256:
  `9e48a6c39d401d089c9f6e5c90980fbdc7b4dc325c83a9a55f8f050e6e4dc2a2`
- Selected V1 adapter SHA256:
  `34e940ec13f52f311765a4a9de76742ba78ff8400d5df46961f8cbd703414192`
- Full-200 metrics SHA256:
  `199f0403c0d48b80bb259d86cd9b27df7ca023841285ed03976f1be7cb03bbe1`
- Full-200 job: `344526`
- Training jobs: V1 `344504`, V2 `344505`, V3 `344508`
- Machine-readable result:
  `docs/results/RECOVERY_FIXED500_COMPACT_INTENT_V1_FULL200.json`

## Research basis

The exploration used recent work as design guidance rather than as evidence
for the result. The preference branch follows the general direction of
[Direct Multi-Turn Preference Optimization (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.138/),
while the explicit offline action feedback follows the motivation of
[ETO (ACL 2024)](https://aclanthology.org/2024.acl-long.409/). The rejected
DAgger branch was motivated by the original
[DAgger analysis](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).
The public-only action interface and explicit executable-action auditing are
consistent with the threat-model discipline emphasized by
[Agent Security Bench (ICLR 2025)](https://openreview.net/forum?id=V4y0CpX4hK).

The empirical verdict remains repository-local and protocol-specific: compact
intent SFT worked in this pilot; balanced SFT and the tested DPO construction
did not.
