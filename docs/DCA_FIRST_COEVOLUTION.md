# DCA-First Alternating Co-Evolution

AgentGuard-Zero uses two independently parameterized roles initialized from the
same frozen backbone:

```text
DCA_0 = B + Delta_D_0
VDA_0 = B + Delta_V_0
```

The adapters are never shared. Round `r + 1` follows a strict causal order:

```text
DCA_r generates on-policy scenarios
  -> VDA_r produces trajectory feedback
  -> update and save DCA_{r+1}
  -> DCA_{r+1} generates a fresh scenario pool
  -> hard checks and security-CFC curriculum filtering
  -> VDA_r rolls out on the fresh train split
  -> update and save VDA_{r+1}
```

This differs from updating both roles from one shared trajectory batch. The
4,000-scenario DCA feedback batch is used only to optimize DCA. The updated DCA
then generates a new 10,000-scenario candidate pool for VDA. Scenario
fingerprints prevent any DCA feedback scenario from entering VDA train, dev, or
xplay.

## Formal Round

`scripts/run_dca_first_round.py` is the authoritative one-round orchestrator.
Its stages are resumable and recorded in `round_state.json`:

1. Build balanced DCA prompts for T1-T4.
2. Generate DCA scenarios and evaluate them against the current VDA.
3. Optimize the DCA LoRA adapter with online hard-but-solvable rewards.
4. Reload the new DCA adapter and generate a fresh four-shard candidate pool.
5. Apply format, validity, safety, solvability, uniqueness, and security-CFC checks.
6. Select a task-balanced top curriculum, then distribute difficulty strata
   across disjoint VDA train/dev/xplay splits and save their hashes.
7. Optimize the VDA LoRA adapter with trajectory-level safety rewards.
8. Independently reload both adapters and verify their hashes differ.

The formal defaults per backbone and round are:

| Artifact | Count |
|---|---:|
| DCA feedback candidates | 4,000 |
| Fresh VDA candidate pool | 10,000 |
| VDA train | 2,400 |
| VDA dev | 400 |
| VDA xplay | 800 |

The four task families are active probing (T1), trust betrayal (T2), profile
memory poisoning (T3), and business-constrained overresponse (T4).

The current VDA influences the next curriculum through the 4,000-scenario
feedback batch used to optimize DCA. The later 10,000-candidate pool filter is
a deterministic security-aware hard-but-solvable selector over generated
scenario metadata and simulator checks. It is not a direct multi-rollout
estimate of current-VDA failure probability, and the paper should use the same
terminology.

## Three Rounds

For each backbone, `scripts/run_three_rounds.py` invokes source rounds 0, 1,
and 2 serially:

```text
DCA_0 + feedback(VDA_0) -> DCA_1 -> fresh pool -> VDA_1
DCA_1 + feedback(VDA_1) -> DCA_2 -> fresh pool -> VDA_2
DCA_2 + feedback(VDA_2) -> DCA_3 -> fresh pool -> VDA_3
```

The 4B and 9B processes may run on separate four-GPU workers. Within one
backbone, the stage order above must remain serial.

## Artifact Lineage

```text
checkpoints/<backbone>/dca/round_<r>/
checkpoints/<backbone>/vda/round_<r>/
data/co_evolution/<backbone>/round_<r>/dca_feedback/
data/co_evolution/<backbone>/round_<r>/vda_candidates/
data/co_evolution/<backbone>/round_<r>/vda_train/
data/co_evolution/<backbone>/round_<r>/vda_dev/
data/co_evolution/<backbone>/round_<r>/vda_xplay/
```

Every checkpoint manifest records the role, backbone, round, parent manifest,
training-data manifest, configuration hash, adapter path, and adapter hash.
Formal artifacts also record one TMCD release revision. A completed stage from
an older revision cannot be resumed into a newer protocol run.
`scripts/audit_dca_first_lineage.py` reconstructs both role chains and checks
the data dependencies. `--expected-host` is optional and can be used when an
experiment protocol binds a run to a specific worker.

## Held-Out Generation

After DCA_3 is sealed, `scripts/generate_final_heldout.py` generates a separate
four-shard pool. It excludes fingerprints from every feedback batch, candidate
pool, train/dev/xplay split, and any earlier backbone heldout. Formal settings
select 200 scenarios per task, for 800 final-heldout scenarios per backbone.

Final-heldout, TMCD-Test-Mix, and CAGE-style transfer data must not be used for
training, prompt revision, checkpoint selection, or selector calibration.
