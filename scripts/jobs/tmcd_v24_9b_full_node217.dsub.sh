#!/bin/bash
#DSUB -n AGZV242_9B_FULL
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-217
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-217
DATA_ROOT=${ROOT}/data/tmcd_v242/qwen3.5-9b
CHECKPOINT_ROOT=${ROOT}/checkpoints/tmcd_v242/qwen3.5-9b
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ -e "${DATA_ROOT}" || -e "${CHECKPOINT_ROOT}" ]]; then
  if [[ "${AGZ_FORMAL_RESUME:-0}" != "1" ]]; then
    echo "Refusing to overwrite formal TMCD v2.4.2 outputs: ${DATA_ROOT} ${CHECKPOINT_ROOT}" >&2
    exit 73
  fi
  echo "Explicitly resuming preserved TMCD v2.4.2 outputs: ${DATA_ROOT} ${CHECKPOINT_ROOT}"
fi

mkdir -p "${ROOT}/logs/tmcd_v242" "${ROOT}/outputs/tmcd_v242/preflight"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"

export AGZ_DCA_PPO_MINI_BATCH_SIZE=40
export AGZ_DCA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_MAX_PROMPT_LENGTH=2048
export AGZ_DCA_MAX_RESPONSE_LENGTH=1024
export AGZ_DCA_REWARD_FSYNC_EVERY_BATCHES=8
export AGZ_DCA_MAX_NUM_SEQS=20
export AGZ_DCA_GPU_MEMORY_UTILIZATION=0.50

export AGZ_REQUIRE_INSTRUCTION_PRESERVATION=1
export AGZ_VDA_FEEDBACK_CONTINUATION_PROMPT_MODE=snapshot
export AGZ_VDA_FEEDBACK_HISTORY_WINDOW=6
export AGZ_VDA_FEEDBACK_MAX_TURNS=16
export AGZ_VDA_FEEDBACK_MAX_INPUT_TOKENS=4096
export AGZ_VDA_FEEDBACK_MAX_NEW_TOKENS=384
export AGZ_VDA_FEEDBACK_INVALID_ACTION_PATIENCE=2
export AGZ_VDA_FEEDBACK_ATTN_IMPLEMENTATION=sdpa

export AGZ_DCA_CANDIDATE_ATTN_IMPLEMENTATION=sdpa
export AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES=16
export AGZ_DCA_CANDIDATE_MAX_ATTEMPTS=3

# 9B settings passed dedicated DCA, candidate-generation, and VDA update gates.
export AGZ_ROLLOUT_BACKEND=hf
export AGZ_AGENT_NUM_WORKERS=4
export AGZ_MAX_NUM_SEQS=16
export AGZ_GPU_MEMORY_UTILIZATION=0.35
export AGZ_VDA_GENERATION_BATCH_SIZE=32
export AGZ_VDA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_ROLLOUT_TOP_K=0
export AGZ_ROLLOUT_SERVER_MAX_PARALLEL_TRAJECTORIES=16
export AGZ_ROLLOUT_SERVER_MAX_STATES=512
export AGZ_VDA_ACTION_TOKENS=320
export AGZ_VDA_OBSERVATION_TOKENS=1280
export AGZ_VDA_TRAJECTORY_TOKENS=11264
export AGZ_VDA_MODEL_TOKENS=15360
export AGZ_VDA_PPO_MINI_BATCH_SIZE=32
export AGZ_VDA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_RESHARD_AFTER_FORWARD=true
export AGZ_MAX_ACTOR_CKPT_TO_KEEP=1
export AGZ_ENABLE_GRADIENT_CHECKPOINTING=true
export AGZ_REQUIRE_TRAJECTORY_REWARD=1

echo "Effective AgentGuard configuration:"
env | grep '^AGZ_' | sed -E '/(KEY|TOKEN|SECRET|PASSWORD)=/s/=.*/=<redacted>/' | LC_ALL=C sort

python -s "${ROOT}/scripts/preflight_tmcd_v2_job.py" \
  --backbone qwen3.5-9b \
  --model-path "${AGZ_QWEN35_9B_PATH}" \
  --variant full \
  --expected-node agent-217 \
  --output "${ROOT}/outputs/tmcd_v242/preflight/node217_9b_full.json"

python -s "${ROOT}/scripts/run_three_rounds.py" \
  --root "${ROOT}" \
  --backbone qwen3.5-9b \
  --experiment-variant full \
  --artifact-scope tmcd_v242 \
  --model-path "${AGZ_QWEN35_9B_PATH}" \
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
  --candidate-batch-size 32

touch "${ROOT}/outputs/tmcd_v242/node217_9b_full.SUCCEEDED"
