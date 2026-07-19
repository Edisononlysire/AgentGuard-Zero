# API Select Optimization Notes - 2026-07-07

## Summary

We optimized the API black-box AgentGuard-Zero-Select evaluator from
`mitigation_v3` to `mitigation_v4`, then ran a V5 framework search over three
non-oracle selector objectives.

The v4 selector keeps the same fairness boundary: it only sees public
observations, generated candidate JSON, allowed tools/actions, and running
public cost annotations. It never reads hidden simulator state, true objective,
true attack path, or oracle labels.

## Main Results: v3/v4 Pilot

All rows use GLM-5.1 API, N=16, max_turns=3, max_new_tokens=768.

| Split | Method | Safe Utility | Delta vs Zero | Attack Mitigation | Avg Steps | Verification Cost |
|---|---:|---:|---:|---:|---:|---:|
| Dev | Zero-shot VDA | 0.3994 | 0.0000 | 0.7500 | 2.9375 | 2.1875 |
| Dev | Select v3 | 0.4882 | +0.0888 | 1.0000 | 2.5000 | 1.8750 |
| Dev | Select v4 | 0.4953 | +0.0959 | 1.0000 | 2.0625 | 1.3750 |
| Holdout-16 | Zero-shot VDA | 0.4242 | 0.0000 | 0.8125 | 2.8125 | 2.7500 |
| Holdout-16 | Select v3 | 0.4865 | +0.0623 | 1.0000 | 2.7500 | 2.4375 |
| Holdout-16 | Select v4 | 0.4943 | +0.0701 | 1.0000 | 2.0000 | 1.2500 |
| Hard-Frontier-16 | Zero-shot VDA | 0.4412 | 0.0000 | 0.9375 | 2.8125 | 1.8125 |
| Hard-Frontier-16 | Select v4 | 0.4731 | +0.0319 | 1.0000 | 2.0000 | 1.1875 |

## Interpretation

Select v4 reaches 100% attack mitigation on dev, holdout, and the
frontier-score hard slice. This gives +25.0 points mitigation on dev, +18.75
points on holdout, and +6.25 points on the hard slice.

The Safe Utility target of +0.20 absolute points is outside the current metric
ceiling for the dev/holdout slices. A post-hoc zero-cost ceiling analysis
estimates the maximum possible average Safe Utility delta over Zero-shot at
about +0.143 on dev and +0.119 on holdout. Select v4 reaches about 67% of that
dev ceiling and about 59% of that holdout ceiling. The hard-frontier slice did
not create a larger Safe Utility gap because GLM zero-shot already mitigated
15/16 hard-frontier cases.

## v4 Change

`mitigation_v4` adds confidence-gated early mitigation:

```text
if public risk is high
and at least one public verification has been spent
and all sampled candidates are still verification-only
and the best generated belief has confidence >= 0.70
and uncertainty <= 0.60:
    synthesize a reversible LimitSession action from public belief
```

If this early gate does not fire, v4 falls back to the v3 late-stage bounded
verification governor.

This reduced average steps and verification cost while preserving 100%
mitigation and 0% overresponse in the current GLM API pilot.

## V5 Framework Search

All V5 variants preserve the same fairness boundary: the selector only uses the
public observation, generated VDA JSON candidates, allowed schema, and public
running costs. It never reads hidden simulator state or oracle labels.

The tested V5 selector objectives were:

- `v5_a_constrained`: Lagrangian-style constrained safe-utility proxy. This
  heavily penalizes unverified profile confirmation, high business risk,
  overresponse risk, and repeated verification delay.
- `v5_b_belief_q`: public belief-state Q proxy. This estimates action value
  from public risk, belief confidence, belief margin, uncertainty, verification
  spend, and low-impact mitigation readiness.
- `v5_c_frontier_minimax`: robust frontier/minimax selector. This treats weak
  source reliability, cross-source conflict, and spoofability as adversarial
  public evidence until verification or quarantine reduces risk.

### Small Pilot Selection

GLM-5.1 API, N=48, offset=32, K=2, max_turns=2.

| Selector | Safe Utility | Attack Mitigation | Safe Success | Reward | Business Cost | Verification Cost | Overresponse | Raw JSON Fail | Governor Override |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_a_constrained` | 0.420705 | 0.770833 | 0.770833 | 2.975990 | 0.564687 | 1.375000 | 0.000000 | 0.276042 | 0.239583 |
| `v5_b_belief_q` | 0.494330 | 1.000000 | 1.000000 | 4.513333 | 0.740000 | 1.166667 | 0.000000 | 0.244792 | 0.385417 |
| `v5_c_frontier_minimax` | 0.494430 | 1.000000 | 1.000000 | 4.511823 | 0.734687 | 1.208333 | 0.000000 | 0.265625 | 0.375000 |

`v5_c_frontier_minimax` won the small pilot by a narrow Safe Utility margin.
`v5_b_belief_q` is effectively tied and remains the most training-friendly
interpretation for Qwen LoRA because it directly matches a learnable
belief-state value model.

### Medium Holdout

GLM-5.1 API, N=96, offset=80, max_turns=3. Select uses K=4.

| Method | Selector | Safe Utility | Delta SU | Attack Mitigation | Delta Mitigation | Reward | Business Cost | Verification Cost | Delta Verification | Overresponse | JSON Fail | Raw JSON Fail | Governor Override | Avg Steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot VDA |  | 0.425764 | +0.000000 | 0.822917 | +0.000000 | 3.225469 | 0.565729 | 2.520833 | +0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.906250 |
| Select v4 | `mitigation_v4` | 0.496651 | +0.070888 | 1.000000 | +0.177083 | 4.515260 | 0.700729 | 1.343750 | -1.177083 | 0.000000 | 0.000000 | 0.400990 | 0.321782 | 2.104167 |
| Select V5-C | `v5_c_frontier_minimax` | 0.496707 | +0.070943 | 1.000000 | +0.177083 | 4.517786 | 0.743594 | 1.104167 | -1.416667 | 0.000000 | 0.000000 | 0.394531 | 0.427083 | 2.000000 |

Medium-holdout interpretation:

- V5-C matches v4's 100% mitigation and 0% overresponse.
- V5-C slightly improves Safe Utility and trajectory reward over v4.
- V5-C reduces verification cost from 1.3438 to 1.1042 and reduces average
  steps from 2.1042 to 2.0000.
- Relative to Zero-shot VDA, V5-C gives +7.09 Safe Utility points, +17.71
  Attack Mitigation points, and -1.42 verification steps.

### Cost

The confirmed API usage for this search was about 3.35M total tokens, including
one sanity check, three small V5 pilots, and the medium zero-shot/v4/V5-C
evaluation. Under the GLM-5.1 price estimates used in the experiment log, this
is roughly 45-80 RMB. The exact charge depends on the provider's metering and
cached-token policy.

## Training Implication

For Qwen LoRA training, do not imitate API actions. Instead, keep the Level-1
trajectory reward as the optimization target and encode the V5 principle in the
training objective/prompt:

```text
learn public belief-state active verification
+ penalize repeated verification after sufficient public evidence
+ penalize unverified profile confirmation
+ reward reversible low-impact mitigation under high public risk
+ constrain overresponse and business cost
```

The cleanest final paper framing is:

```text
Public Belief-State Active Verification
+ Constrained Trajectory Safe-Utility Optimization
+ DCA Frontier/Minimax Curriculum
```
