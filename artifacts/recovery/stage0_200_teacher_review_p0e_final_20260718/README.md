# Recovery Stage-0 acceptance artifact

This directory records the final accepted, model-free Stage-0 gate for the
`action-support-bootstrap-v1` recovery protocol. The run used the implementation
at commit `7ccfd035e67436f09bc3a7dc0cda12f37deb95e0`.

## Scope

- 200 deterministic canonical scenarios: 50 each for T1-T4.
- 100 public-equivalent counterfactual groups with two hidden worlds per group.
- Fixed-policy comparison of Oracle, the Finite-Counterfactual Public-State
  Robust Teacher, Random Legal, No-op, and Overreact.
- No language-model inference and no parameter update (`model_calls=0`,
  `parameter_updates=0`).

This is a teacher/environment learnability gate. It is **not** a Qwen3.5-4B
zero-shot evaluation, a trained-model result, a DCA-generated evaluation, or an
approval to start Bootstrap SFT, DAgger, RL, or co-evolution.

## Accepted result

| Policy | Safe Utility | Attack Mitigation | Safe Success | Business Cost |
|---|---:|---:|---:|---:|
| Oracle | 0.7789375 | 75.0% | 75.0% | 1.34125 |
| Public-State Robust Teacher | 0.7675075 | 75.0% | 75.0% | 1.03985 |
| Random Legal | -1.2784829 | 3.5% | 2.5% | 0.911925 |
| No-op | -1.0750000 | 0.0% | 0.0% | 0.0 |
| Overreact | -1.3250000 | 0.0% | 0.0% | 0.0 |

Additional audited values:

- Teacher minus No-op Safe Utility: `1.8425075`.
- Teacher/core utility mean rank correlation: `0.9177742211` across 940
  decision states.
- Teacher selected actions: Observe 465, Mitigation 342, Active Probe 44,
  Passive Verification 54, Trust 35, Memory 0.
- Gate verdict: `accepted`; the only permitted next operation recorded by the
  gate is `bootstrap_data_build_and_audit`.

`Random Legal > No-op` was false, but that ordering is explicitly diagnostic
and is not a Stage-0 hard acceptance condition.

## Provenance and reproduction

The canonical scenario source SHA256 is:

```text
54f3c794f20ba4151b93f2f52efd1238ac2c80eddaca2cb52edb28f58844d6d4
```

The source can be reproduced from the committed deterministic generator:

```bash
python scripts/generate_recovery_canonical.py \
  --scenario-count 200 \
  --output canonical_200.json
```

The exact server-side audit, console log, source hash, code hashes, and original
checksum list are preserved in this directory. `SHA256SUMS.github` verifies the
files under their GitHub artifact names; `SHA256SUMS` is the unmodified
server-side checksum record and therefore retains the original server-relative
paths.

## Limitations

The canonical scenarios are deterministic protocol fixtures rather than
DCA-generated scenarios. This accepted gate does not establish transfer to the
DCA distribution. The Stage-0 artifact also does not contain a formal
fingerprint-overlap audit against historical training data or sealed TMCD-Test;
such an audit is required before any subsequently approved Bootstrap data is
used for model training. Memory actions had zero selected support in this run
and must be addressed by the independent Bootstrap data audit rather than being
treated as already validated.
