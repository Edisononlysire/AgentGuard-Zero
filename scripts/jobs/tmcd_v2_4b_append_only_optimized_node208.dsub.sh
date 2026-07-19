#!/bin/bash
#DSUB -n AGZV2_4B_APPEND_OPT
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-208
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2_optimized/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2_optimized/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-208
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

mkdir -p "${ROOT}/logs/tmcd_v2_optimized" "${ROOT}/outputs/tmcd_v2/preflight"
export AGZ_ROOT="${ROOT}"

# Preserve all 4,000 DCA feedback examples while retaining 100 optimizer
# mini-updates: 80 prompts x 2 rollouts x 25 steps, mini-batch 40.
export AGZ_DCA_PPO_MINI_BATCH_SIZE=40
export AGZ_DCA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_MAX_PROMPT_LENGTH=2048
export AGZ_DCA_MAX_RESPONSE_LENGTH=1024

export AGZ_VDA_FEEDBACK_MAX_TURNS=16
export AGZ_VDA_FEEDBACK_MAX_INPUT_TOKENS=4096
export AGZ_VDA_FEEDBACK_MAX_NEW_TOKENS=320
export AGZ_VDA_FEEDBACK_CONTINUATION_PROMPT_MODE=snapshot
export AGZ_VDA_FEEDBACK_INVALID_ACTION_PATIENCE=2
export AGZ_VDA_FEEDBACK_ATTN_IMPLEMENTATION=sdpa

export AGZ_DCA_CANDIDATE_ATTN_IMPLEMENTATION=sdpa
export AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES=8
export AGZ_DCA_CANDIDATE_MAX_ATTEMPTS=3

export AGZ_VDA_ACTION_TOKENS=320
export AGZ_VDA_OBSERVATION_TOKENS=1280
export AGZ_VDA_TRAJECTORY_TOKENS=11264
export AGZ_VDA_MODEL_TOKENS=15360
export AGZ_VDA_MAX_NUM_SEQS=8
export AGZ_VDA_PPO_MINI_BATCH_SIZE=32
export AGZ_VDA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_VDA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_RESHARD_AFTER_FORWARD=true
export AGZ_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AGZ_VDA_SAVE_FREQ=25

source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"

python -s "${ROOT}/scripts/preflight_tmcd_v2_job.py" \
  --backbone qwen3.5-4b \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --variant append_only_memory \
  --expected-node agent-208 \
  --output "${ROOT}/outputs/tmcd_v2/preflight/node208_4b_append_only_optimized.json"

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
  --dca-batch-size 80 \
  --dca-steps 25 \
  --vda-candidates 10000 \
  --vda-train-size 2400 \
  --vda-dev-size 400 \
  --vda-xplay-size 800 \
  --vda-batch-size 32 \
  --vda-steps 75 \
  --vda-rollout-n 1 \
  --vda-max-turns 16 \
  --candidate-batch-size 72

touch "${ROOT}/outputs/tmcd_v2/node208_4b_append_only_optimized.SUCCEEDED"
