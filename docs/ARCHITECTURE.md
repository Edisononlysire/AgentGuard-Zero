# Existing T1/T2 Model Architecture

This page describes the historical result-producing model. See the
[audited existing results](RESULTS.md) and the
[revised overall plan](PLAN.md) for the current evidence
boundary and proposed next steps; the proposal has not produced these results.

## 1. Decision Interface

At each step, the environment exposes a public observation and the candidate
generator constructs legal structured defense actions. The model performs two
related decisions:

1. select an action family from `observe`, `passive_verification`,
   `active_probe`, `trust`, and `mitigation`;
2. rank candidates inside the selected family by expected safe utility.

Explicit task labels, hidden attack state, oracle values, and teacher scores
are omitted from the encoded policy input. This is a field-level statement,
not a proof that the input has no answer shortcuts: the audited encoder still
includes `require_*` workflow hints, and the environment and teacher enforce
related workflow constraints. See [the current limitations](STATUS.md).

## 2. Encoder

The encoder is Qwen3.5-4B loaded through `transformers.AutoModel`. A rank-16
LoRA adapter is applied to:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

The frozen base architecture is shared. The current T1/T2 expert is initialized
from the common D0 adapter/head state rather than a later T3/T4 checkpoint.

The implementation creates two textual views:

- a **public-state view** for action-family prediction;
- a **public-state + candidate view** for candidate utility prediction.

The last non-padding hidden representation is pooled and passed to learned
heads.

## 3. Heads

The result-producing architecture is `shared_local_long_v1` with the
`branched_defense` objective. T1/T2 use its local decision branch.

Primary heads:

- state action-family logits;
- candidate utility;
- local family and local utility;
- residual family and utility calibration.

Outcome heads trained by the current `branched_defense` objective include
information gain, terminal mitigation, business cost, overresponse, final safe
success, and trajectory safe utility. The source also retains compatibility
heads for support, belief, uncertainty, probe value, business risk, and safety
risk; these are not explicit optimization terms in the current T1/T2 run.
Final action selection remains hierarchical family-then-utility.

The existing `probe_value` head is present but was not explicitly supervised by
the current `branched_defense` loss. The revised plan would test explicit
Value-of-Information supervision only in the matched `AEP-Policy+VoIaux` arm.
The core `AEP-Policy` arm would learn public-value teacher actions without this
auxiliary loss. Neither proposed training arm produced the historical result.

## 4. Training Contract

The frozen current run used:

| Item | Value |
|---|---:|
| Backbone | Qwen3.5-4B |
| LoRA rank | 16 |
| Train candidate sets | 4,000 |
| Held-out data-development sets | 400 |
| T1/T2 train split | 2,000 / 2,000 |
| Epochs | 4 |
| Optimizer steps | 1,000 |
| Global batch | 16 |
| Maximum sequence length | 2,048 |
| Selection | hierarchical family then utility |
| ECRG during parameter training | disabled |
| ECRG in the audited evaluation | disabled |
| Selected checkpoint | epoch 2, optimizer step 500 |

The 400-trajectory epoch-selection suite is distinct from the 400 offline
development records and separate from the retention suites in [RESULTS](RESULTS.md).
The four completed epochs do not mean that the reported policy uses epoch 4.
Checkpoint selection is lexicographic, led by macro T1/T2 Safe Success.

## 5. Scenario And Supervision Path

```text
canonical_recovery_group(T1 or T2)
  -> cyber_grounded_recovery_group_v2
  -> paired hidden-world execution
  -> public projection
  -> robust teacher search
  -> legal candidate set
  -> lifecycle-constrained target
  -> family/stage-balanced train and dev records
```

The builder records semantic scenario fingerprints, public-state digests,
candidate-set hashes, group lineage, and train/dev disjointness checks.

## 6. Known Limitation Motivating vNext

The current scenario contract exposes requirement fields that can correlate a
task state with a particular probe family, and the current teacher contains
probe preferences tied to those requirements. The existing result is therefore
evidence for the historical T12 candidate policy, not for the proposed
VoI-supervised AEP or task-independent causal probing.

The proposed vNext implementation would remove those public shortcuts, give
both tasks the same probe registry, return raw delayed evidence from causal
mechanisms, and train the teacher to prefer a probe when its Value of
Information is positive.

The 2026-09-07 CPU review also reproduced loss of an older evidence record when
the candidate references its evidence ID but the compactor prioritizes its
event ID. Minimal inputs differing only in root-source dependencies or evidence
availability time can map to the same policy text. These defects remain in the
published runtime; their frequency in historical rollouts is not yet measured.
See [STATUS](STATUS.md) for evidence levels and [PLAN](PLAN.md) for the staged
repair and diagnostic gates. The plan does not mandate replacing hierarchical
selection or extending the existing architecture before measuring actual losses.
