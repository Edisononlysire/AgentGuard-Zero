# Candidate Co-Evolution Improvement Analysis

Analysis date: 2026-07-19. This document uses the committed warm-start,
three-round training, fixed T1-T4 evaluation, and fresh-curriculum artifacts.

## Decision

The candidate representation should be retained, including active probing and
deterministic compilation. It removed parser and reference failures, but the
current pilot did not learn defense. The next run should not scale the current
configuration unchanged. It should first repair ranker learnability and
trajectory supervision, then pass Gate A before DCA or RL is unlocked.

## Verified Pattern

- Compiler validity, public-reference validity, Teacher-best recall, and
  near-optimal recall were 100% in every audited round. The correct action was
  available, so candidate generation was not the limiting factor.
- Active-probe selection was 33.3%-39.4%, but probe yield was 0%. The policy
  learned to select probe-shaped candidates, not to exploit probe evidence.
- Mitigation selection was 55.2%-78.8%, while attack mitigation and Safe
  Success remained 0%. The failure is target and sequence selection, not lack
  of mitigation-shaped output.
- Memory operation and Memory use were 0% in every fixed evaluation. T3 also
  had the largest per-task regret (about 1.61).
- A1, A2, and A3 produced identical decisions on the held-out set. Three round
  candidates were all rejected, leaving the warm-start checkpoint active.

## Root Causes

### 1. The scoring heads did not receive a realistic optimization budget

The utility and auxiliary linear heads are randomly initialized, but they use
the same learning rate as LoRA: `1e-5`, `5e-6`, and `2e-6` in A1-A3. Each stage
ran only four optimizer steps on 14 training states. This is too small to
establish a calibrated ranking head, and the unchanged held-out argmax confirms
that the updates did not alter policy decisions.

### 2. Part of A1 rewards format recognition instead of defensive choice

Half of `family_loss` classifies each candidate's own declared action family.
That is easy to infer from candidate text and does not answer which family is
appropriate for the public state. It reinforces the behavior already observed
in free-JSON training: recognizing or reproducing an action schema without
learning defensive utility.

The state-dependent decision-family loss exists, but it is diluted by this
candidate-type classification and by auxiliary losses that are active from the
first step. Final inference uses only the utility head, so good auxiliary-head
loss does not guarantee better action selection.

### 3. The warm-start split cannot measure all required skills

Warm-start training contained only 14 states. It had no passive-verification
target, while the six-state development split had no active-probe or Memory
target. All warm-start records also used `task_id=unknown`. Consequently, the
reported development set cannot verify active probing, Memory, or T1-T4
generalization.

### 4. Probe and Memory supervision is isolated rather than longitudinal

The round training mixes contained zero probe-follow-up states in all three
rounds, even though the fresh pools for rounds 2 and 3 each contained one. The
probe-grounded follow-up loss was therefore inactive during round training.
Memory targets numbered only 1, 4, and 2 in the three 16-state mixes, with no
complete t0-t4 Memory trajectory retained as a unit.

This explains the metric split: active-probe rate is nonzero, but probe yield
and Memory use are zero. The policy sees action labels but not enough
cause-and-effect chains.

### 5. Listwise targets are too diffuse for the available sample count

Each state has about 20 candidates. Teacher top-candidate probability averaged
0.141 in round 2 and 0.157 in round 3; entropy was 2.642 and 2.560, close to the
maximum `log(20)=2.996`. This preserves near-optimal alternatives, but with only
four updates it produces weak pressure on the correct target and parameter.

### 6. The curriculum is not on the VDA learning frontier

The DCA feedback frontier rate was 0% in round 2 and 13.3% in round 3, below the
30% gate. These values were based on an offline regret proxy rather than four
actual Safe-Success rollouts. Round 2's Top-1 increase occurred while mean
regret worsened from 0.0836 to 0.2679, so it is not evidence of improvement.

The fixed evaluation has only eight scenarios, two per task. It is useful as a
smoke test but is too small for checkpoint promotion.

## Required Changes Before Another Three-Round Run

### P0: Prove ranker learnability

1. Add optimizer parameter groups: train new utility/family heads at `3e-4` to
   `1e-3`, and LoRA at `1e-5`. Freeze LoRA for the first 200-500 head updates.
2. Remove candidate-type classification from A1, or reduce it to at most 5%.
   Train state-conditioned family choice directly.
3. Log each loss component, utility margin, gradient norm, score variance, and
   argmax changes. Auxiliary losses must be disabled until utility ranking can
   overfit a controlled set.
4. Run a 256-state balanced overfit test. Require at least 95% train Top-1 and
   90% train family accuracy. Failure blocks larger training and indicates a
   model, pooling, masking, or optimizer defect.

### P1: Build skill-complete supervised data

1. Build 8,000-12,000 warm-start states with stratification by T1-T4, action
   family, attack/benign world, target, and trajectory position.
2. Give every held-out split active-probe, probe-follow-up, Memory lifecycle,
   Memory-use, correct mitigation, and benign Observe examples.
3. Preserve trajectories as groups. At least 25% of batches should contain a
   probe -> returned evidence -> grounded follow-up chain. T3 batches must keep
   complete t0-t4 Memory sequences.
4. Add same-family hard negatives: wrong probe target, low-information probe,
   wrong mitigation target, premature trust update, and Memory operation without
   later use.

### P2: Make ranking pressure match defensive utility

1. Combine soft listwise distillation with hard argmax and utility-gap losses:
   `KL(teacher || policy) + CE(best) + gap-weighted pairwise margin`.
2. Use an adaptive Teacher temperature or Top-k truncation so tied near-optimal
   candidates remain soft while clearly inferior candidates receive little
   mass.
3. Add explicit target accuracy and probe-evidence grounding metrics. A
   candidate counts as an active-probe success only when its evidence changes a
   later belief, trust, Memory, or mitigation decision.
4. Separate pure information-gathering probes from composite probe+mitigation
   actions in metrics, so a bundled mitigation cannot satisfy the probe gate.

### P3: Reopen capability-gated co-evolution

1. Keep DCA frozen until a 200-scenario, three-seed Gate A passes.
2. Estimate frontier probability with `G=4` actual Safe-Success trajectories.
   Admit only Teacher-solvable scenarios with VDA success in `[0.2, 0.8]`.
3. Reject a curriculum before VDA training when task coverage or frontier gates
   fail. Do not use rejected samples even for a candidate checkpoint.
4. Promote only when fixed Safe Utility, Safe Success, validity, per-skill
   floors, fresh mean regret, and previous-pool retention all pass. Top-1 alone
   is insufficient.
5. Unlock candidate-level trajectory RL only after two consecutive supervised
   round gates pass. Keep BC replay and KL active during RL.

## Next Minimal Experiment

The next run should be a learnability pilot, not another co-evolution run:

| Phase | Scale | Required result |
|---|---:|---|
| Head-only overfit | 256 balanced states | Train Top-1 >=95% |
| LoRA supervised pilot | 2,000 states, 200 dev | Dev Top-1 >=40%, regret decreases |
| Probe-chain pilot | 500 complete chains | Probe yield >=10% |
| T3 Memory pilot | 500 complete trajectories | Memory use >=5%, T3 success >0 |
| Gate A | 200 scenarios x 3 seeds | Safe Success >=10%, no skill zero |

Only after these checks pass should the project spend compute on another three
DCA-VDA rounds. This sequence tests whether the ranker can learn actual
state-action utility before dynamic curricula add distribution shift.
