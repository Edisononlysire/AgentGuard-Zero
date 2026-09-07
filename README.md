# AgentGuard-Zero: T1/T2 Research Release

This repository contains the public implementation scope needed to inspect the
current AgentGuard T1/T2 result and the proposed next-generation active probing
system.

The focused tasks are:

- **T1: ambiguity-driven active investigation**;
- **T2: trust build-up followed by source betrayal or legitimate change**.

The release includes the public-state candidate ranker, T1/T2 scenario
generator, robust teacher, data construction, frozen training/evaluation
contracts, tests, and an audited aggregate-only historical results snapshot.
It intentionally excludes model weights, LoRA adapters, generated datasets,
raw trajectories, logs, credentials, and scheduler outputs.

## Latest Plan And Existing Results

- **[新版整体规划（中文，2026-09-07）](docs/T12_OVERALL_PLAN_20260907.md)**:
  revised active probing, public-information teacher, ECRG, data boundaries,
  matched baselines, causal tests, external investigation transfer, and delivery.
- **[现有 T1/T2 结果（中文，2026-09-07）](docs/T12_CURRENT_RESULTS_20260907.md)**:
  verified counts, all four development epochs, probe usage, and limitations.
- **[Machine-readable historical results](results/t12_legacy_retention_20260907.json)**:
  aggregate counts and source hashes, without private paths or raw traces.

| Historical T12, ECRG disabled | Safe Success | Attack Mitigation |
|---|---:|---:|
| T1 | 50/150 (33.33%) | 65/150 (43.33%) |
| T2 | 100/150 (66.67%) | 132/150 (88.00%) |

These are **legacy development/retention results**, not untouched final tests
for the new protocol. The original retention gate **did not pass**, including
overresponse and intent-retention checks. No vNext causal-probing or T12+ECRG
gain is claimed. Read the results page for negative evidence and denominators.

## Release Status

Two method states are kept separate:

1. **Current T1/T2 model**: the architecture and data path that produced the
   existing T1/T2 result. It uses a Qwen3.5-4B LoRA candidate ranker with
   hierarchical action-family and candidate-utility selection.
2. **Independent active probing vNext**: a revised design proposal. It removes
   task-to-probe shortcuts, gives T1 and T2 the same probe registry, produces
   delayed raw observations through causal simulator mechanisms, and teaches
   probing through value of information. It has not yet produced the current
   result and is not presented as implemented training output.

See:

- [`docs/T12_MODEL_ARCHITECTURE.md`](docs/T12_MODEL_ARCHITECTURE.md)
- [`docs/T12_RELEASE_SCOPE.md`](docs/T12_RELEASE_SCOPE.md)
- [`docs/T12_OVERALL_PLAN_20260907.md`](docs/T12_OVERALL_PLAN_20260907.md)
- [`docs/T1_T2_ACTIVE_PROBING_DESIGN_20260904.md`](docs/T1_T2_ACTIVE_PROBING_DESIGN_20260904.md)
  (superseded proposal, retained for provenance)

## Current T1/T2 Pipeline

```text
T1/T2 hidden-world scenario groups
        |
        v
cyber-grounded public observations
        |
        v
public-only robust teacher + legal candidate generator
        |
        v
4,000 train / 400 held-out candidate sets
        |
        v
Qwen3.5-4B + LoRA public-state/candidate encoder
        |
        +--> action-family head
        +--> candidate utility head
        +--> public auxiliary heads
        |
        v
hierarchical family-then-utility decision
```

The policy input excludes hidden state, oracle labels, teacher scores, and task
labels. Training metadata may retain those fields for offline audit, but the
public projector is the model-input boundary.

## Repository Map

| Path | Purpose |
|---|---|
| `agentguard_zero/candidate/` | candidate generation, encoding, heads, ranking, and branch supervision |
| `agentguard_zero/recovery/canonical_scenarios.py` | canonical hidden-world T1/T2 construction |
| `agentguard_zero/candidate/cyber_grounding.py` | public cyber grounding and ATT&CK/telemetry mappings |
| `agentguard_zero/recovery/public_teacher.py` | public-state robust teacher used for current supervision |
| `scripts/build_v11_t12_native_v2.py` | exact current 4,000/400 T1/T2 dataset builder |
| `scripts/train_candidate_ranker.py` | Qwen/LoRA multi-head candidate-ranker trainer |
| `scripts/eval_v11_single_expert_policy.py` | single-expert trajectory evaluation |
| `configs/t12/t12_ranker_public.json` | portable description of the frozen model/training contract |
| `scripts/smoke_t12_scenarios.py` | CPU smoke for T1/T2 scenario generation and leakage checks |
| `results/t12_legacy_retention_20260907.json` | audited aggregate historical evidence, not new-protocol results |

## Quick Start

Python 3.12 was used for the experiments. The CPU inspection path is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/smoke_t12_scenarios.py --groups-per-task 2
python -m pytest -q \
  tests/test_v11_t12_native_v2.py \
  tests/test_v11_t12_epoch_selection.py
```

The full ranker needs a local Qwen3.5-4B checkpoint plus CUDA-compatible
PyTorch, Transformers, PEFT, and the packages in
`requirements-training.txt`. The repository does not redistribute the base
model or trained adapter.

The exact result-producing hyperparameters are recorded in
`configs/t12/t12_ranker_public.json`. Machine-specific frozen manifests are not
published because they contain private filesystem paths and do not improve the
portable source release.

## Active Probing vNext

T1 and T2 will share four tool interfaces:

- `AttestationChallenge` (`SourceChallenge` compatibility alias);
- `SensorCanaryProbe` (`CanaryProbe` alias);
- `DecoyInteractionProbe` (`DecoyProbe` alias);
- `ShadowEnforcementProbe` (`ShadowActionProbe` alias).

Canary and decoy are the primary environment interventions. Attestation is
active source verification; shadow execution estimates response impact. Valid
authentication is not proof of truth, and raw-looking output fields must not
encode hidden attack labels.

The teacher will actively cover beneficial probing opportunities while keeping
the same policy at indistinguishable public histories. Cost is counted once;
confirmation of a correct decision can be useful without changing the action.
No task ID or `require_*_probe` field may route the model to a probe. Result
masking, no-probe rollouts, and result corruption are separate evaluations.

The complete proposed contract, causal mechanisms, data composition, losses,
counterfactual metrics, and acceptance gates are in
[`docs/T12_OVERALL_PLAN_20260907.md`](docs/T12_OVERALL_PLAN_20260907.md).
This is a plan, not a claim that the new runtime or training has been implemented.

## Safety And Reproducibility

- The simulator uses abstract, safety-bounded defensive actions.
- It does not produce exploit payloads or execute attacks against real systems.
- Train/dev/test groups must be split by public-group identity.
- Public policy inputs must pass the hidden-state leakage audit.
- Checkpoints and data artifacts are hash-bound but excluded from Git.
- Aggregate results permit table recalculation, not full experiment reproduction
  without the separately held data and checkpoints.
- Current results and proposed vNext behavior are never conflated.

## License

AgentGuard-Zero project code is released under the Apache License 2.0.
Vendored third-party components remain under their respective upstream
licenses; see `THIRD_PARTY_NOTICES.md`.
