# AgentGuard-Zero Framework Lock

This project is no longer an Agent0 reproduction. Agent0 is used only as the
training substrate: GRPO/ADPO infrastructure, vLLM rollout plumbing, and cluster
scripts. The research object, reward source, and task semantics are different.

## Frozen Model Framework

AgentGuard-Zero is a two-agent, zero-human-label training framework for
adversarial observation defense:

- DCA: Deception Curriculum Agent.
  Generates safe, symbolic, non-operational cyber deception scenarios.
- VDA: Verification Defender Agent.
  Produces structured defensive decisions under poisoned observations.
- Level 1 simulator:
  Maintains hidden attack state, poisoned observations, profile memory,
  business cost, verification cost, and action consequences.
- CFC: Conservative Frontier Co-Evolution.
  Selects scenarios that are safe, oracle-solvable, diverse, and difficult for
  the current VDA.
- CA-ADPO: Consequence-Aware ADPO.
  Trains VDA with simulator-derived rewards and quality-scaled advantages.

## Difference From Agent0

| Agent0 | AgentGuard-Zero |
|---|---|
| Math/general reasoning tasks | Adversarial observation defense scenarios |
| Executor solves questions | VDA updates trust, verifies evidence, controls response |
| Python interpreter as tool | Abstract cyber verification tools |
| Majority-vote pseudo-label | Hidden-state simulator consequence reward |
| Self-consistency frontier | Safe-success / overresponse / poisoning frontier |
| Answer accuracy | Safe Utility and trajectory outcome |

The common outer loop is intentional: both systems use zero-label co-evolution.
The contribution is not another Agent0 domain port. The contribution is changing
the learning signal from pseudo-label agreement to consequence-grounded safety
policy evolution under corrupted observations.

## Training Stages

1. VDA action warmup:
   Train valid JSON, legal tool calls, evidence quarantine, and low-impact
   responses from scenario-conditioned observations.
2. VDA consequence RL:
   Use simulator-grounded rewards for model-generated VDA action JSON.
3. Multi-step VDA rollout:
   Interleave model actions, abstract tool results, environment transitions, and
   trajectory reward.
4. DCA GRPO:
   Train DCA to generate frontier scenarios that are safe, oracle-solvable,
   diverse, and hard for the current VDA.
5. Full CFC:
   Alternate DCA frontier generation and VDA CA-ADPO training for 3-4 rounds.

## Non-Negotiable Safety Boundary

DCA outputs only abstract cyber scenarios. It must not generate payloads,
exploit code, malware behavior, real IPs, real organizations, or operational
attack instructions.
