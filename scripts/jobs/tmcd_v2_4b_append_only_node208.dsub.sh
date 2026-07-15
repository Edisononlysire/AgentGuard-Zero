#!/bin/bash
#DSUB -n AGZV2_4B_APPEND
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-208
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
mkdir -p "${ROOT}/logs/tmcd_v2" "${ROOT}/outputs/tmcd_v2/preflight"
export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"

# Keep ablation runtime controls identical to the 4B full run.
export AGZ_DCA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_VDA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_VDA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_RESHARD_AFTER_FORWARD=true
export AGZ_MAX_ACTOR_CKPT_TO_KEEP=1
export AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES=16
export AGZ_DCA_REWARD_FSYNC_EVERY_BATCHES=8
export AGZ_ENABLE_GRADIENT_CHECKPOINTING=true
export AGZ_VDA_FEEDBACK_MAX_TURNS=16
export AGZ_VDA_FEEDBACK_MAX_INPUT_TOKENS=4096
export AGZ_VDA_FEEDBACK_MAX_NEW_TOKENS=384

echo "Effective AgentGuard configuration:"
env | grep '^AGZ_' | sed -E '/(KEY|TOKEN|SECRET|PASSWORD)=/s/=.*/=<redacted>/' | LC_ALL=C sort

python -s "${ROOT}/scripts/preflight_tmcd_v2_job.py" \
  --backbone qwen3.5-4b \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --variant append_only_memory \
  --expected-node agent-208 \
  --output "${ROOT}/outputs/tmcd_v2/preflight/node208_4b_append_only.json"

python -s "${ROOT}/scripts/run_three_rounds.py" \
  --root "${ROOT}" \
  --backbone qwen3.5-4b \
  --experiment-variant append_only_memory \
  --artifact-scope tmcd_v2 \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
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
  --candidate-batch-size 16

touch "${ROOT}/outputs/tmcd_v2/node208_4b_append_only.SUCCEEDED"
