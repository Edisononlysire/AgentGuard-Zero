# T1/T2 Focused Release Scope

## Included

This release is the source-level dependency closure for the current T1/T2
result:

- public observation and hidden-world environment state;
- canonical and cyber-grounded T1/T2 scenario generation;
- public-only teacher and candidate generation;
- candidate compiler and action schema validation;
- Qwen3.5-4B LoRA candidate-ranker architecture;
- current branched-defense supervision used by the T1/T2 local branch;
- 4,000/400 data construction and audit code;
- checkpoint selection and trajectory evaluation code;
- portable frozen method metadata and unit tests;
- independent active probing vNext design.
- revised Chinese overall plan and audited T1/T2 aggregate results, including
  historical gate failures and source hashes (2026-09-07).

## Excluded

The following are intentionally not part of the source release:

- Qwen or cybersecurity-LLM weights;
- LoRA adapters and learned head tensors;
- generated JSONL/Parquet datasets;
- server addresses, passwords, tokens, or API keys;
- scheduler job outputs and runtime logs;
- raw evaluation trajectories and sealed final-test material;
- T3/T4 expert training and three-expert router experiments.

Some shared simulator modules retain generic T3/T4-compatible branches because
the current T1/T2 checkpoint was trained using that exact shared code. They are
dependencies, not entrypoints in this focused release.

## Provenance Boundary

Use `configs/t12/t12_ranker_public.json` for the portable method contract.
Machine-specific run manifests remain private because they contain internal
filesystem paths. Checkpoints and generated data remain hash-audited in the
private experiment archive but are not part of this source release.

The aggregate-only snapshot in `results/t12_legacy_retention_20260907.json`
contains historical development/retention evidence. It excludes private
absolute paths and records the original failed retention gate. It is not a new
benchmark release or a claim that vNext has completed training. Recomputing the
published tables is possible from its counts; reproducing the complete runs
still requires the unpublished data and checkpoints.

## Current Versus Proposed

The current result uses the implementations in:

```text
agentguard_zero/candidate/
agentguard_zero/recovery/
agentguard_zero/env/
scripts/build_v11_t12_native_v2.py
scripts/train_candidate_ranker.py
```

The proposed independent active probing system is specified in:

```text
docs/T12_OVERALL_PLAN_20260907.md
```

That document is deliberately marked as a design. It must not be cited as the
implementation behind the existing T1/T2 numbers until it is implemented,
retrained, and evaluated from a new frozen lineage.

The earlier `docs/T1_T2_ACTIVE_PROBING_DESIGN_20260904.md` is retained as a
superseded proposal. The September 7 plan governs conflicting definitions.
