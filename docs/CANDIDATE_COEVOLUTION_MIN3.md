# Probe-Centric Candidate Co-Evolution

This branch is a complete project copy that adds a public-state candidate VDA
and a capability-gated DCA-VDA co-evolution path. It does not replace or delete
the historical DCA implementation.

## Architecture

```text
public observation + defender state
  -> public CandidateGenerator (8-24 legal options)
  -> Qwen shared encoder + utility/family/belief/uncertainty/probe/risk heads
  -> utility argmax over the current legal set
  -> deterministic V4 CandidateCompiler
  -> public evidence, trust, memory, and mitigation state transition
  -> DCA frontier/skill-gap curriculum
```

Candidate keys and order are remapped per state and seed. Stable semantic IDs,
Teacher Q values, hidden worlds, and oracle labels are audit-only fields and are
never included in ranker input. Final action selection uses only the utility
head; auxiliary heads cannot override it.

Active probing is a first-class action family. `SourceChallenge`, `CanaryProbe`,
`DecoyProbe`, and `ShadowActionProbe` have distinct candidates. Probe-followup
records supervise both the next non-Observe action and preference for candidates
that cite the newly returned public evidence.

## Code Map

| Path | Responsibility |
|---|---|
| `agentguard_zero/candidate/` | candidate types, generation, compilation, ranker, policy, metrics, gates |
| `agentguard_zero/recovery/public_teacher.py` | public-state robust listwise Q, core-first filtering, skill ablations |
| `agentguard_zero/recovery/canonical_scenarios.py` | T1-T4 skill-identifiable scenarios, including longitudinal T3 |
| `scripts/build_candidate_dataset.py` | listwise/multi-head/probe-chain records with `world_count >= 2` |
| `scripts/train_candidate_ranker.py` | A1 family, A2 listwise, A3 preference, and joint/probe-chain training |
| `scripts/collect_candidate_dagger.py` | semantic-ID-safe error-focused DAgger |
| `scripts/eval_candidate_policy.py` | fixed T1-T4 trajectory evaluation and multi-label metrics |
| `scripts/run_candidate_coevolution_min3.py` | resumable three-round diagnostic pilot with acceptance and rollback |
| `scripts/run_candidate_coevolution_round.py` | formal LLM-DCA integration path |
| `agentguard_zero/rewards/candidate_dca_reward.py` | G-rollout frontier/novelty/skill-gap/regret reward |

Candidate-level trajectory RL remains locked until Gate A and supervised
co-evolution pass. This is intentional: the completed pilot did not satisfy the
precondition, so running PPO would turn infrastructure-valid actions into an
unsupported formal lineage.

## Reproduction

Prepare the six-source warm start only after skill-identifiability passes:

```bash
python scripts/prepare_candidate_warmstart.py \
  --output-dir data/candidate_coevolution_min3/warmstart \
  --skill-gate evaluations/candidate_coevolution_min3/skill_identifiability.json
```

Run A1, A2, and A3 with `scripts/train_candidate_ranker.py`, using objectives
`family`, `listwise`, and `preference`. The trainer requires an accepted, hashed
data manifest and writes adapter/head hashes in every checkpoint manifest.

The minimum three-round command is:

```bash
source scripts/qwen35_env.sh
python scripts/run_candidate_coevolution_min3.py \
  --model-path "$AGZ_QWEN35_4B_PATH" \
  --warmstart-manifest outputs/candidate_min3/warmstart_a3/manifest.json \
  --canonical-replay data/candidate_coevolution_min3/warmstart/split/train.jsonl \
  --canonical-manifest data/candidate_coevolution_min3/warmstart/split/train_manifest.json \
  --diagnostic-override-gates
```

`--diagnostic-override-gates` is only for failure analysis. Without it, Gate A
or DCA feedback rejection stops the run. Each candidate checkpoint is evaluated
on a fixed T1-T4 suite, its fresh pool, and the previous pool. A rejected round
is rolled back and cannot become the parent of the next round.

## Pilot Result

The 2026-07-19 minimum run completed three rounds. All candidate/compiler audits
passed, and active probing remained nonzero. Gate A failed because probe yield,
Memory, mitigation success, and safe success were zero. None of the three round
checkpoints was promoted. See
`evaluations/candidate_coevolution_min3/RESULTS.md` and the hashed JSON reports
under `evaluations/candidate_coevolution_min3/pilot3/`.
