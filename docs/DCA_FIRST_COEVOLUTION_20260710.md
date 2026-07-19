# AgentGuard-Zero DCA-First Co-Evolution Implementation

## Frozen protocol

Each backbone follows the same alternating dependency:

```text
DCA_r generates feedback scenarios
-> VDA_r produces real multi-turn rollout feedback
-> update and save DCA_{r+1}
-> independently reload DCA_{r+1}
-> DCA_{r+1} generates a fresh candidate pool
-> hard safety, validity, solvability, CFC, duplicate, and leakage filtering
-> train and save VDA_{r+1}
-> independently reload VDA_{r+1}
```

The DCA feedback batch is never reused by VDA train, dev, or xplay. Semantic
scenario fingerprints exclude only scenario IDs and lineage metadata, so
renaming a feedback scenario cannot bypass the leakage check.

## Runtime

- Environment: `agent0-gpu`, loaded by `scripts/qwen35_env.sh`.
- Qwen3.5 gated-delta-rule kernels use the isolated `fla-core==0.5.1` overlay;
  the base conda environment is unchanged. ARM has no compatible prebuilt
  `causal-conv1d` wheel, so causal convolution keeps the safe Torch fallback.
  Rebuild it with `scripts/install_qwen35_fla_overlay.sh`.
- Every concurrent FLA process uses an isolated node-local Triton cache under
  `/tmp/agentguard_zero_triton_${USER}`. Ray workers are separated by PID;
  feedback services and candidate generators are separated by service/shard.
- Formal node: `cyclone001-agent-217`.
- Every Pilot and formal dsub entry point has a runtime hostname gate and exits
  before model loading if the scheduler places it anywhere other than node 217.
- Allocation: one backbone job at a time, using all four A100 40 GB GPUs and up
  to 230 GB host memory on `cyclone001-agent-217`. The other backbone remains
  pending in the scheduler.
- 4B and 9B DCA updates use four-rank FSDP. The independently loaded current VDA
  feedback service shares the last allocated GPU.
- Four feedback services bind their health endpoints before loading VDA
  weights, one service per GPU. A DCA reward batch is partitioned across them;
  each loads the same independently saved `VDA_r` only for its assigned
  feedback subset and moves the weights back to CPU before the DCA optimizer
  step. This preserves the strict serial dependency while avoiding simultaneous
  full-model residency during DCA generation and update. The later VDA update
  uses all four GPUs.
- After `DCA_{r+1}` is saved, its fresh VDA candidate pool is generated as four
  data-parallel shards, one per GPU. The merge gate requires one shared DCA
  manifest hash and exact, duplicate-free index coverage before CFC filtering.
- Fixed reproducibility seed: `20260709`; no multi-seed run is planned yet.
- Qwen3.5 activation offload is disabled because its gradient-checkpoint state
  can contain tuples unsupported by the current VerL offload hook. Gradient
  checkpointing, layer-wise FSDP, and 9B reference-policy offload remain enabled.

## Task horizons

The training parquet records carry task-specific `max_env_steps`:

| Task | Horizon |
| --- | ---: |
| T1 active probing | 10 |
| T2 trust-building betrayal | 16 |
| T3 profile-memory poisoning | 14 |
| T4 business-constrained overresponse | 10 |

Formal rollout has a global ceiling of 16 and stops at each row's own horizon.
Engineering pilots use three turns only.
The four-GPU Pilot uses four DCA prompts with two rollouts each (eight feedback
candidates), so every FSDP rank receives one prompt in the single optimizer step.
Formal DCA uses batch 16 (2,000 prompts / 125 steps, two rollouts per prompt),
and formal VDA uses batch 16 (2,400 trajectories / 150 steps). The requested
4,000 feedback candidates and 2,400 VDA train scenarios are unchanged.
DCA uses GRPO with two generated scenarios per prompt. VDA keeps one multi-turn
trajectory per train scenario and uses REINFORCE++ outcome advantages, avoiding
a degenerate one-sample group baseline while preserving the planned decision
record count.

## Entry points

- One resumable round: `scripts/run_dca_first_round.py`
- Node wrapper: `scripts/run_dca_first_round_node217.sh`
- Three formal rounds: `scripts/run_dca_first_three_rounds_node217.sh`
- 4B formal job: `scripts/train_dca_first_round_4b_node217_dsub.sh`
- 9B formal job: `scripts/train_dca_first_round_9b_node217_dsub.sh`
- Ordered formal submitter: `scripts/submit_dca_first_formal_node217.sh`; it
  submits 9B with an explicit `4B=SUCCEEDED` scheduler dependency.
- Formal post-processing: `scripts/generate_final_heldout_node217.py`; after
  Round 3 it generates a new-seed pool, excludes every co-evolution feedback,
  candidate, train, dev, and xplay fingerprint, then seals 200 scenarios per
  task (800 per backbone) with hashes. The later 9B job also excludes the
  already sealed 4B heldout fingerprints.
- Final verifier: `scripts/audit_dca_first_lineage.py`; Pilot and formal jobs
  only succeed after this verifier reconstructs every parent edge, data split,
  adapter hash, reload report, four-rank config, and node-217 execution record.
- 4B pilot: `scripts/train_dca_first_round_4b_pilot_node217_dsub.sh`
- 9B pilot: `scripts/train_dca_first_round_9b_pilot_node217_dsub.sh`

Submit a job with:

```bash
dsub -pn cyclone001-agent-217 -s scripts/<job-script>
```

## Artifact and lineage gates

Every round writes role-separated checkpoints and data under:

```text
checkpoints/{backbone}/{dca|vda}/round_{r}/
data/co_evolution/{backbone}/round_{r}/
```

Engineering pilots use the separate `checkpoints_pilot/` and
`data/co_evolution_pilot/` roots and can never satisfy a formal-round resume
check.
Formal VDA splitting is stratified before shuffling and hard-enforces per-task
counts: train 600, dev 100, and xplay 200 for each of T1–T4.

The round is complete only when all of the following hold:

1. DCA and VDA manifests have the expected role, backbone, round, and parent.
   Each also freezes the execution host, four-GPU allocation, batch/rollout
   settings, LoRA settings, optimizer steps, and a canonical configuration hash.
2. Feedback, candidate, train, dev, and xplay files have recorded SHA256 hashes.
3. The DCA checkpoint timestamp predates the fresh VDA candidate pool.
4. Every VDA row traces to `DCA_{r+1}` and its manifest hash.
5. Feedback and VDA split semantic fingerprints have zero overlap.
6. DCA and VDA adapter hashes differ.
7. Both adapters independently reload through PEFT; reload reports are stored in
   the round data directory.

## Validation status

- Python and shell compile checks pass in `agent0-gpu`.
- `python -m unittest -v tests.test_dca_first_coevolution` passes fifteen tests.
- Qwen3.5 4B and 9B resolve to the conditional-generation model class.
- Thinking is disabled in Qwen prompt rendering for DCA and VDA JSON rollouts.
- V5-C active probing is implemented and an API trajectory selected a real
  `SourceChallenge` before bounded mitigation. Evidence:
  `outputs/tmcd_api_eval_active_probe/api_v5c_active_probe_aug_t1_l1_c2_tok384_20260709/summary.json`
  records four GLM requests, 6,472 total tokens, zero JSON/tool errors,
  safe-success 1.0, and attack-mitigation 1.0.
- Lily-Cybersecurity-7B is deployed; its functional VDA JSON smoke uses the
  model's native `### Instruction / ### Input / ### Response` prompt format.
  Evidence:
  `outputs/tmcd_lily_smoke/tmcd_lily_cybersecurity_vda_functional_smoke_342528/summary.json`
  records a two-step trajectory with zero JSON/tool errors and intent accuracy
  1.0.
