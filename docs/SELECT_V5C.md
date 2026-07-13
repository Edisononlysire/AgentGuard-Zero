# AgentGuard-Zero-Select V5-C

AgentGuard-Zero-Select is the frozen-parameter deployment variant. It uses the
same structured VDA action schema, active probing actions, profile-memory
partitions, Level-1 simulator, and trajectory metrics as AgentGuard-Zero-Train,
but it does not update model parameters.

At each step, a frozen model samples candidate actions:

```text
C_t = {a_t^1, ..., a_t^K} ~ pi_theta(. | h_t)
a_t* = argmax S_public(o_t, a),  a in C_t
```

The V5-C frontier-minimax selector can inspect only public observations,
generated action JSON, legal tool/action definitions, public history, and
running cost annotations. It cannot inspect the true objective, hidden attack
path, oracle labels, or whether an event was generated as fake evidence.

## Public Safety Score

```text
S_public(o, a) =
    S_schema(a)
  + S_verification(o, a)
  + S_poison_guard(o, a)
  + S_business_safety(o, a)
  + S_frontier_minimax(o, a)
```

- `S_schema` checks strict JSON, required fields, tools, and response actions.
- `S_verification` values cross-checks and low-risk active probes under public
  uncertainty, while penalizing repeated verification-only delay.
- `S_poison_guard` favors quarantine or rejection of suspicious profile claims
  and penalizes unverified confirmation.
- `S_business_safety` favors reversible controls and penalizes unsupported
  high-impact responses.
- `S_frontier_minimax` selects actions robust to public trust-conflict,
  spoofability, poisoning, and overresponse indicators.

If suspicious evidence appears at the first turn and sampled candidates contain
no active probe, V5-C may add a `SourceChallenge` candidate. The augmentation
uses only public state, quarantines the associated claim, and has low declared
business and overresponse risk. Later public governors can prefer reversible
actions such as `LimitSession`, `ShadowBlock`, or `DeployDecoy` when additional
verification would mostly add delay.

## Relationship To Train

```text
Zero-shot VDA:       K = 1, frozen parameters, no selector.
ReAct / Base+Tools:  frozen parameters with ordinary tool use.
Select V5-C:         frozen parameters plus public safety selection.
Train:               independently trained VDA LoRA with trajectory rewards.
Train + V5-C:        trained VDA candidates plus the same public selector.
```

Select is calibrated only on training data and finalized on dev data. Test and
held-out data are never used to tune selector weights or prompts.
