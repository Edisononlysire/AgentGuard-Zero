#!/bin/bash
#DSUB -n AGZV2_9B_FULL
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo logs/tmcd_v2/%J.out
#DSUB -eo logs/tmcd_v2/%J.err

set -euo pipefail
ROOT="${AGZ_ROOT:-${PWD}}"
if [[ ! -d "${ROOT}/agentguard_zero" ]]; then
  echo "Set AGZ_ROOT to the AgentGuard-Zero repository root" >&2
  exit 71
fi
EXPECTED_NODE=cyclone001-agent-217
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
mkdir -p "${ROOT}/logs/tmcd_v2" "${ROOT}/outputs/tmcd_v2/preflight"
export AGZ_ROOT="${ROOT}"
export AGZ_DCA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_VDA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_VDA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_RESHARD_AFTER_FORWARD=true
export AGZ_VDA_FEEDBACK_MAX_TURNS=16
export AGZ_VDA_FEEDBACK_MAX_INPUT_TOKENS=4096
export AGZ_VDA_FEEDBACK_MAX_NEW_TOKENS=384
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"

python -s "${ROOT}/scripts/preflight_tmcd_v2_job.py" \
  --backbone qwen3.5-9b \
  --model-path "${AGZ_QWEN35_9B_PATH}" \
  --variant full \
  --expected-node agent-217 \
  --output "${ROOT}/outputs/tmcd_v2/preflight/node217_9b_full.json"

python -s "${ROOT}/scripts/run_three_rounds.py" \
  --root "${ROOT}" \
  --backbone qwen3.5-9b \
  --experiment-variant full \
  --artifact-scope tmcd_v2 \
  --model-path "${AGZ_QWEN35_9B_PATH}" \
  --allocated-gpus "${CUDA_VISIBLE_DEVICES}" \
  --seed 20260709 \
  --dca-feedback-candidates 4000 \
  --dca-rollout-n 2 \
  --dca-batch-size 40 \
  --dca-steps 50 \
  --vda-candidates 10000 \
  --vda-train-size 2400 \
  --vda-dev-size 400 \
  --vda-xplay-size 800 \
  --vda-batch-size 32 \
  --vda-steps 75 \
  --vda-rollout-n 1 \
  --vda-max-turns 16 \
  --candidate-batch-size 8

touch "${ROOT}/outputs/tmcd_v2/node217_9b_full.SUCCEEDED"
