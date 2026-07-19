# Evaluation Requirements Gap Analysis

Analysis date: 2026-07-19. Source requirements: the proposed TMCD main,
progressive-ablation, active-probing, co-evolution, ECRG, and CAGE evaluation
protocol. This file distinguishes existing evaluation infrastructure from what
the three-round Candidate pilot actually demonstrated.

## Current Evidence Status

The Candidate pilot is a diagnostic run, not a paper result. Its fixed set has
eight scenarios, two per T1-T4 task. It validates implementation contracts and
detects complete skill collapse, but it cannot estimate formal NSU, confidence
intervals, baseline superiority, transfer, or co-evolution trends.

The repository's formal TMCD path already supports several requirements that
the Candidate pilot did not exercise:

- `scripts/eval_tmcd_systems.py` scopes Probe Yield to T1, Betrayal Detection to
  T2, Poison Success to T3, and Overresponse to T4. Business Cost and Safe
  Utility are task-macro averages.
- `scripts/validate_tmcd_main_full.py` requires a sealed, balanced 2,400-scenario
  TMCD-Test, exactly 600 scenarios per task, fixed horizons, frozen VDA3 and
  ECRG artifacts, K=6 Full inference, and preserved candidate hashes.
- `scripts/export_tmcd_tables.py` computes NSU from random and oracle bounds and
  exports task-level results.
- Existing tests enforce task-specific metric scopes and frozen ECRG
  invariants.

These capabilities do not turn the pilot into a formal evaluation. No Candidate
checkpoint passed Gate A, so formal TMCD, ECRG, and CAGE evaluation must remain
blocked.

## Requirement Matrix

| Requirement | Current Candidate evidence | Status |
|---|---|---|
| Train learns defense without ECRG | Safe Success, Probe Yield, Memory use all 0 | Fail |
| T1 active probing | Probe-shaped actions nonzero; T1 Probe Yield 0 | Fail |
| T2 trust betrayal | No formal Betrayal Detection result | Not measured |
| T3 poisoning defense | Memory operation/use 0; T3 regret about 1.61 | Fail |
| T4 constrained response | Mitigation frequent, but Safe Success 0 | Fail |
| Overall/task NSU | No random/oracle normalization in pilot | Not measured |
| 2,400 sealed TMCD-Test | Pilot uses 8 canonical scenarios | Not formal |
| 95% paired bootstrap CI | No Candidate paired-bootstrap artifact | Missing |
| Progressive ablation fairness | No equal-budget Candidate ablations | Missing |
| Probe-budget causality | No 0/1/2/3 budget sweep on one Train checkpoint | Missing |
| DCA-VDA cross-play | All candidate rounds rolled back | Not established |
| Frozen ECRG contribution | Correctly locked because Train failed | Blocked |
| CAGE Challenge 2 transfer | No promotable Train/Full checkpoint | Blocked |

## How Evaluation Should Drive Better Training

### T1: Optimize information use, not probe frequency

The training target must be a complete chain: ambiguous public state, selected
probe and target, returned probe evidence, changed belief/trust, and a better
follow-up action. Report probe information gain, evidence-grounded follow-up,
and final NSU together. A probe action with no later decision change receives no
positive skill reward.

During development, sweep active-probe budgets 0/1/2/3 on the same checkpoint.
Do not promote a model unless `NSU(1) > NSU(0)` and the gain exceeds verification
cost. This directly supports the paper's active-probing claim.

### T2: Train current-claim verification against historical trust

Construct matched pairs with identical source history but different current
claim consistency. Add hard negatives that follow the historically trusted
source without current evidence. Supervise trust recalibration and require a
verification/probe before high-impact acceptance. Optimize T2 Betrayal Detection
and NSU, not generic trust-operation rate.

### T3: Make Memory causally necessary

Use two-phase poisoning and clean-decision trajectories. The clean phase must be
unsolvable from the current observation alone and solvable from correctly
quarantined or confirmed memory. Keep the complete trajectory in one training
group and score the final decision. Raise the Teacher-vs-No-Memory utility gap
above the pilot's marginal 0.1139 before generating large data.

### T4: Distinguish correct mitigation from overresponse

Create matched attack/benign worlds with the same high-severity public claim and
different corroborating evidence. Train same-family wrong-target and
wrong-severity negatives. Promote only when T4 NSU improves while Overresponse
and Business Cost do not regress. Mitigation rate alone is not a success metric.

## Evaluation Ladder

1. **Learnability test:** 256 balanced states, including all task/family cells.
   Require at least 95% train Top-1. This tests the optimizer and scorer.
2. **Static skill dev:** 200 disjoint scenarios with task-specific metrics.
   Require every T1-T4 skill to be nonzero and mean Teacher regret to decrease.
3. **Gate A:** 200 scenarios x 3 seeds, ECRG disabled. Require Safe Success at
   least 10%, T1 Probe Yield at least 10%, Memory use at least 5%, and no task
   skill collapse.
4. **Curriculum admission:** use G=4 actual trajectory outcomes and admit only
   Teacher-solvable scenarios with VDA success in `[0.2, 0.8]`.
5. **Three-round cross-play:** evaluate every VDA0-VDA3 against every held-out
   DCA1-DCA3 pool. Require later VDA improvement and early-pool retention.
6. **Formal attribution:** run equal-budget progressive ablations and the T1
   probe-budget sweep before enabling ECRG.
7. **Formal Full:** freeze VDA3 and ECRG, seal TMCD-Test after training, run all
   baselines on identical scenarios/seeds, and calculate 10,000 paired-bootstrap
   confidence intervals.
8. **External transfer:** only a promoted Full checkpoint proceeds to frozen
   CybORG CAGE Challenge 2 adapters and paired episode seeds.

## Claim Boundary

Until steps 1-5 pass, the supported claim is limited to: dynamic candidates and
deterministic compilation remove output-contract errors while preserving access
to active-probe actions. The current evidence does not support claims that the
VDA learned defensive active probing, safe Memory governance, or DCA-VDA
co-evolution.
