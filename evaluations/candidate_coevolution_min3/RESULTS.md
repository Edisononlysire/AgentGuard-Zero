# Candidate Co-Evolution Minimum Pilot

Run date: 2026-07-19. Backbone: Qwen3.5-4B. Fixed evaluation: 8 scenarios,
two hidden worlds for each T1-T4 group. Every round used 16 mixed candidate
sets and four effective distributed optimization steps.

## Fixed T1-T4 Evaluation

| Checkpoint | Safe utility | Safe success | Attack mitigation | Probe yield | Mean regret | Active probe | Memory/use |
|---|---:|---:|---:|---:|---:|---:|---:|
| Warm start | -1.5089 | 0% | 0% | 0% | 0.3437 | 37.9% | 0% / 0% |
| Round 1 candidate | -1.4717 | 0% | 0% | 0% | 0.8898 | 39.4% | 0% / 0% |
| Round 2 candidate | -1.3033 | 0% | 0% | 0% | 0.7465 | 33.3% | 0% / 0% |
| Round 3 candidate | -1.4227 | 0% | 0% | 0% | 0.7465 | 33.3% | 0% / 0% |

## Fresh Curriculum Evaluation

| Round | Regret start | Regret end | Top-1 start | Top-1 end | DCA feedback gate | Promoted |
|---|---:|---:|---:|---:|---|---|
| 1 | 0.7929 | 0.8823 | 0% | 0% | Reject: task coverage | No |
| 2 | 0.0836 | 0.2679 | 0% | 6.25% | Reject: frontier rate | No |
| 3 | 0.2023 | 0.1974 | 13.33% | 20.00% | Reject: frontier rate | No |

## Assessment

The architecture fixes the output-contract failure: candidate compilation,
public references, permutation consistency, Teacher-best recall, and
near-optimal recall were 100% in every audited pool. Active probing also remains
available and nonzero.

The pilot does **not** show learned defense. Safe success, actual attack
mitigation, probe yield, Memory operations, and Memory use remain zero in all
three rounds. Safe utility gains come from lower-cost action sequences, not
successful defense. Round 2's isolated top-1 gain accompanied worse mean
regret, which is not a robust improvement.

The result supports the representation change but rejects the current tiny
training scale and curriculum admission policy. Before another formal run:

1. Make DCA admission use actual `G=4` safe-success samples, not an offline
   regret proxy.
2. Expand Probe-chain and T3 Memory-use states substantially; four Memory
   positives in a 16-state round did not change the policy.
3. Require fresh mean-regret reduction in addition to argmax/top-1 gain.
4. Pass fixed 200-scenario, three-seed Gate A before enabling LLM-DCA updates or
   candidate-level RL.
