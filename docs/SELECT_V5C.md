# AgentGuard-Zero V5-C Evidence-Constrained Robust Public-State Governor

AgentGuard-Zero-Select is the frozen-parameter deployment variant. It uses the
same structured VDA action schema, active probing actions, profile-memory
partitions, Level-1 simulator, and trajectory metrics as AgentGuard-Zero-Train,
but it does not update model parameters.

At each step, a frozen model samples candidate actions:

```text
C_t = {a_t^1, ..., a_t^K} ~ pi_theta(. | h_t)
a_t* = argmax S_public(o_t, a),  a in C_t
```

The V5-C runtime governor can inspect only public observations, generated action
JSON, legal tool/action definitions, public history, and running cost
annotations. It cannot inspect the true objective, hidden attack path, oracle
labels, or whether an event was generated as fake evidence. V5-C is not a
training-time policy and does not update model parameters.

## Two-Stage Public Governance

V5-C first applies a non-negotiable admission gate. A candidate is inadmissible
when it violates the action schema, cites unavailable evidence, exceeds the
remaining verification budget, invents a target, confirms memory without
independent support and provenance, or proposes a high-impact action without
the required public trust support. Inadmissible candidates cannot be rescued by
a high soft score.

Only admitted candidates enter public-state robust ranking:

```text
S_public(o, a) =
    S_verification(o, a)
  + S_poison_guard(o, a)
  + S_business_safety(o, a)
  + S_robustness(o, a)
```

- `S_verification` values cross-checks and low-risk active probes under public
  uncertainty, while penalizing repeated verification-only delay.
- `S_poison_guard` favors quarantine or rejection of suspicious profile claims
  and penalizes unverified confirmation.
- `S_business_safety` favors reversible controls and penalizes unsupported
  high-impact responses.
- `S_robustness` favors actions robust to public trust conflict, telemetry
  inconsistency, poisoning, and overresponse indicators.

Candidate-declared uncertainty, business cost, and overresponse risk remain
model outputs; V5-C does not treat them as ground-truth safety facts. Its gate
and ranking derive authorization, target validity, evidence state, trust state,
memory provenance, budget, and asset criticality from the public environment.
If no candidate is admissible, the governor derives a bounded active probe from
the current public claim, trust, budget, and asset state. It uses `Observe` only
when no legal probe is available or the verification budget is exhausted. It
never executes the least-invalid proposal.

If suspicious evidence appears and sampled candidates contain no active probe,
V5-C may add a `SourceChallenge` candidate using only public state. Later, the
governor can prefer reversible actions such as `LimitSession`, `ShadowBlock`,
or `DeployDecoy` when additional verification would mostly add delay.

## Relationship To Train

```text
Zero-shot VDA:       K = 1, frozen parameters, no selector.
ReAct / Base+Tools:  frozen parameters with ordinary tool use.
Select V5-C:         frozen parameters plus public safety selection.
Train:               independently trained VDA LoRA with trajectory rewards.
Train + V5-C:        trained VDA candidates plus the same public selector.
```

Select is calibrated only on training data and finalized on dev data. Test and
held-out data are never used to tune governor weights or prompts. We describe
the implementation as evidence-constrained hard gating followed by soft robust
ranking; we do not claim a formal minimax optimization guarantee.
