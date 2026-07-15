# AgentGuard-Zero

Official research code for **AgentGuard-Zero: Zero-Label Co-Evolution for Safe
Cyber Defense under Trust Deception and Profile Poisoning**.

AgentGuard-Zero studies autonomous defensive decision-making when observations,
source trust, and long-term profile memory may be manipulated. It contains an
abstract safety-bounded simulator, a Deception Curriculum Agent (DCA), a
Verification Defense Agent (VDA), the frozen-parameter V5-C Evidence-Constrained
Runtime Governor, and the
strict DCA-first alternating co-evolution pipeline used by the paper.

The repository intentionally excludes model weights, LoRA checkpoints,
generated datasets, API credentials, site-specific scheduler outputs, logs, and
paper result artifacts. It contains the method, example launchers, and
experiment code needed to inspect and reproduce the pipeline with separately
obtained models and compute.

## Method At A Glance

```text
DCA_r proposes 4,000 scenarios
  -> VDA_r rollout feedback
  -> train independent DCA_{r+1} LoRA
  -> DCA_{r+1} generates 10,000 new candidates
  -> safety + validity + solvability + security-CFC curriculum filtering
  -> train/dev/xplay = 2,400/400/800
  -> train independent VDA_{r+1} LoRA
```

The process runs for three rounds independently on Qwen3.5-4B and Qwen3.5-9B.
DCA and VDA share a frozen backbone architecture but never share adapters.
Every batch and checkpoint is hashed, role-labelled, and linked to its parent.
The DCA update is conditioned on current-VDA rollout feedback. The fresh VDA
pool is then selected with safety, consistency, oracle-solvability, uniqueness,
and security-CFC checks. This pool filter is not described as a direct
multi-rollout current-VDA frontier estimator.

VDA can use passive verification and four defensive active probes:
`SourceChallenge`, `CanaryProbe`, `DecoyProbe`, and `ShadowActionProbe`. These
return seeded, noisy qualitative evidence rather than hidden truth. DecoyProbe
also changes the next public observation through an explicit probe state. These
are abstract, low-risk simulator actions; the project does not generate attack
payloads, exploit real systems, or execute network attacks.

## Repository Map

| Path | Purpose |
|---|---|
| `agentguard_zero/` | schemas, simulator, memory, rewards, tools, lineage |
| `scripts/run_dca_first_round.py` | authoritative resumable one-round pipeline |
| `scripts/run_three_rounds.py` | serial three-round launcher for one backbone |
| `scripts/eval_tmcd_systems.py` | baselines, Train, Select, and Train+V5-C evaluation |
| `agentguard_zero/governance/v5c.py` | V5-C hard admission and public-state robust ranking |
| `curriculum/reward_function/` | online DCA and trajectory VDA rewards |
| `docs/` | protocol and V5-C method details |
| `third_party/` | patched VerL and Verl-Tool runtime subset |

## Quick Start

Python 3.12 was used for the GPU experiments. A lightweight CPU setup for
schemas, simulator logic, dataset construction, and tests is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

For full LoRA/GRPO training, install the CUDA-compatible packages listed in
`requirements-training.txt`. PyTorch, `causal-conv1d`, and FlashAttention
wheels are platform specific, so install builds matching the local CUDA driver.
The reference Qwen3.5 runtime used Transformers 5.13.0, FLA-Core 0.5.1, and
causal-conv1d 1.6.2.post1. Target-directory installs can be supplied through
`AGZ_TRANSFORMERS_OVERLAY`, `AGZ_FLA_OVERLAY`, and
`AGZ_CAUSAL_CONV_OVERLAY`. The vendored `third_party/verl` and
`third_party/verl_tool` directories are added to `PYTHONPATH` by
`scripts/env.sh`.

Set external model paths before training:

```bash
export AGZ_QWEN35_4B_PATH=/path/to/Qwen3.5-4B
export AGZ_QWEN35_9B_PATH=/path/to/Qwen3.5-9B
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Run a reduced engineering pilot:

```bash
python scripts/run_three_rounds.py \
  --backbone qwen3.5-4b \
  --artifact-scope pilot \
  --dca-feedback-candidates 256 \
  --vda-candidates 1024 \
  --vda-train-size 256 \
  --vda-dev-size 64 \
  --vda-xplay-size 64
```

Run the formal three-round defaults:

```bash
python scripts/run_three_rounds.py \
  --backbone qwen3.5-4b \
  --allocated-gpus 0,1,2,3
```

Run the 9B chain in a separate four-GPU process. Do not run two backbone chains
on the same four devices.

After round 3, generate a sealed DCA_3 heldout split and audit lineage:

```bash
python scripts/generate_final_heldout.py \
  --backbone qwen3.5-4b \
  --allocated-gpus 0,1,2,3

python scripts/audit_dca_first_lineage.py \
  --backbone qwen3.5-4b \
  --artifact-scope formal \
  --max-round 3
```

See [`docs/DCA_FIRST_COEVOLUTION.md`](docs/DCA_FIRST_COEVOLUTION.md) for the
data contract and [`docs/SELECT_V5C.md`](docs/SELECT_V5C.md) for the
frozen-parameter selector.

## Evaluation

`eval_tmcd_systems.py` supports the paper's rule, tool, memory, trust-score,
cybersecurity-LLM, trained VDA, Select, and Train+V5-C systems. For example:

```bash
python scripts/run_tmcd_eval_four_gpu.py \
  --data /path/to/TMCD-Test-Mix.parquet \
  --system agentguard_zero_full \
  --model-path "$AGZ_QWEN35_4B_PATH" \
  --adapter-path /path/to/checkpoints/qwen3.5-4b/vda/round_3/adapter \
  --run-name qwen35_4b_full
```

Use `python scripts/eval_tmcd_systems.py --help` for all system identifiers and
backend options. API credentials are read only from the environment variable
named by `--api_key_env`; no credential value is written to source code.
The evaluation-only controls `agentguard_zero_train_random_k`,
`agentguard_zero_train_generic_best_of_k`, and
`agentguard_zero_train_soft_v5c` reuse the same trained adapter and candidate
count; they do not require additional co-evolution training.

## Reproducibility Contract

- DCA feedback data never enters VDA train/dev/xplay.
- A VDA pool must cite the newly updated DCA checkpoint.
- Every formal artifact must match the active TMCD release revision; legacy
  DCA feedback, checkpoints, candidate pools, and splits fail closed.
- DCA and VDA adapters have distinct paths and SHA-256 hashes.
- Selector calibration uses train and dev only.
- Final-heldout, TMCD-Test-Mix, and CAGE-style data are test-only.
- Runs record a fixed seed even when multi-seed reporting is not requested.

The code does not include claimed numerical results. Tables should be produced
from sealed evaluation outputs with `scripts/export_tmcd_tables.py`.

## Acknowledgements

The dual-agent curriculum/executor idea is inspired by Agent0. The training
runtime includes modified snapshots of VerL and Verl-Tool; their upstream
licenses and notices are preserved under `third_party/` and summarized in
`THIRD_PARTY_NOTICES.md`.

## License

AgentGuard-Zero project code is released under the Apache License 2.0. Vendored
third-party components remain under their respective upstream licenses.
