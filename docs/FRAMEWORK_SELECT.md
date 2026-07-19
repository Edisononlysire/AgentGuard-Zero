# AgentGuard-Zero-Select

AgentGuard-Zero-Select is the training-free instantiation of the AgentGuard-Zero
framework. It uses the same Level-1 simulator, scenario frontier, VDA JSON
schema, tool/action space, multi-step environment consequences, and trajectory
reward as AgentGuard-Zero-Train, but freezes the base model parameters.

At each simulator step, the frozen base model samples a candidate set:

```text
C_t = {a_t^1, ..., a_t^K} ~ pi_theta(. | h_t)
```

where theta is fixed. A non-oracle selector chooses the executed action:

```text
a_t* = argmax_{a in C_t} S_public(o_t, a)
```

`S_public` can inspect only the public observation, the generated action JSON,
the allowed tool/action schema, and running public cost annotations. It must not
inspect hidden simulator state such as the true objective, true attack path,
oracle labels, or whether an observed event is fake.

## Selector Components

The production selector used for API transfer after the V5 search is
`v5_c_frontier_minimax`: a public belief-state active-verification selector
with constrained safe-utility scoring and a frontier/minimax safety governor.
The central design choice is that verification is a means, not the final
defense action. When public evidence remains risky after one or two checks, the
selector prefers reversible controls such as `LimitSession`, `ShadowBlock`, or
`DeployDecoy` over further verification-only actions.

The selector score has four layers.

```text
S_public(o, a) =
  S_schema(a)
  + S_verification(o, a)
  + S_safety(o, a)
  + S_utility_proxy(o, a)
  + S_v5_frontier(o, a)
```

`S_schema` rewards parseable strict JSON, required VDA fields, legal tools, and
legal response actions. It penalizes JSON parse failures, invalid tools, missing
fields, and invalid response actions.

`S_verification` rewards active verification when the public observation shows
weak source reliability, cross-source conflict, high spoofability, or an
unverified profile update. It rewards `CrossCheck`, `ProvenanceCheck`,
`GraphQuery`, `BusinessImpactEstimator`, and conservative quarantine of
unverified profile evidence early in the rollout. After the public running
verification cost reaches one or two checks, additional verification-only
actions are penalized unless uncertainty remains very high.

`S_safety` penalizes high-impact actions such as `Isolate`, `Restore`, and
`Remove` unless the action itself reports strong verified evidence and low
uncertainty. It rewards shadow or low-impact controls under uncertainty.

`S_utility_proxy` rewards low-impact mitigation actions, avoids unnecessary
delay under high public risk, and discourages excessive business risk and
overresponse risk. This proxy is used only for candidate selection; final
results are still measured by simulator trajectory reward and Safe Utility.

`S_v5_frontier` is the V5 addition. It treats weak source reliability,
cross-source conflict, high spoofability, unverified profile memory, and
critical-asset risk as a public adversarial frontier. It rewards quarantine or
rejection of suspicious profile evidence, rewards verified or bounded
low-impact mitigation, and penalizes unverified profile confirmation, repeated
verification after public readiness, high-impact actions under uncertainty, and
business-risk-heavy responses.

If every sampled candidate is malformed, the Select safety governor may create
a compact fallback action from public context and partial generated beliefs.
The fallback never uses hidden simulator state. It chooses conservative
cross-checking before evidence exists, and low-impact mitigation only after
public risk, verification spend, or generated belief confidence is high enough.
If all sampled candidates remain verification-only after one public
verification and the generated belief is high-confidence with low uncertainty,
the public governor may synthesize a reversible `LimitSession` action early. If
confidence is lower, the same governor waits until the late rollout stage before
mitigating. Evaluations report both executed-action parse failures and raw
candidate parse failures, plus the selector fallback and governor override
rates.

## V5 API Selection Result

On GLM-5.1 API medium holdout (N=96, offset=80, max_turns=3), V5-C was selected
as the production API selector:

| Method | Safe Utility | Attack Mitigation | Verification Cost | Overresponse |
|---|---:|---:|---:|---:|
| Zero-shot VDA | 0.4258 | 0.8229 | 2.5208 | 0.0000 |
| Select v4 | 0.4967 | 1.0000 | 1.3438 | 0.0000 |
| Select V5-C | 0.4967 | 1.0000 | 1.1042 | 0.0000 |

V5-C gives +7.09 Safe Utility points and +17.71 Attack Mitigation points over
Zero-shot VDA while reducing verification cost by 1.42 steps and preserving 0%
overresponse. It also slightly improves Safe Utility over the already-strong
v4 selector while using fewer verification steps.

## Relationship To Other Variants

```text
Zero-shot VDA:
  K = 1. The first generated action is executed.

AgentGuard-Zero-Select:
  K > 1. The base model is frozen, and a public safety-utility selector chooses
  one candidate action at each step.

AgentGuard-Zero-Train:
  The VDA receives Level-1 trajectory rewards and updates LoRA/model parameters.
```

Select can participate in the same outer DCA-VDA co-evolution loop because DCA
can generate new frontier scenarios against the current frozen-model-plus-selector
policy. It does not include parameter-level VDA evolution.

The same policy principle is used in the Train prompt and rollout reward path:
the learned VDA should quarantine suspicious profile evidence, spend a bounded
verification budget, and then converge to reversible mitigation instead of
spending the full trajectory on verification-only behavior.

For parameter training, the API selector is not a teacher-label source. Qwen
LoRA should be optimized against the Level-1 trajectory reward, with the V5
principle expressed as loss shaping and curriculum design:

```text
maximize trajectory Safe Utility
subject to low JSON/tool invalidity,
           low overresponse and business cost,
           low unverified profile confirmation,
           bounded verification delay.
```
