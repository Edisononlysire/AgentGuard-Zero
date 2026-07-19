#!/bin/bash
#DSUB -n AGZV2_4B_VDA_STABLE_GATE
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-175
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2_optimized/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2_optimized/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-175
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"

GATE_ROOT="${ROOT}/outputs/tmcd_v2/gates/vda_stable_batch32_20260715"
GATE_DATA="${GATE_ROOT}/train.parquet"
mkdir -p "${GATE_ROOT}" "${ROOT}/logs/tmcd_v2_optimized"

# Exercise the failed long-tail region with the stable 9B VDA schedule:
# one global batch of 32, split across four GPUs.
python -s -c 'import pyarrow.parquet as pq, sys; table=pq.read_table(sys.argv[1]).slice(800, 32); assert table.num_rows == 32; pq.write_table(table, sys.argv[2])' \
  "${ROOT}/data/tmcd_v2/qwen3.5-4b/round_1/vda_train/train.parquet" \
  "${GATE_DATA}"

export AGZ_MODEL_PATH="${AGZ_QWEN35_4B_PATH}"
export AGZ_TRAIN_FILE="${GATE_DATA}"
export AGZ_VAL_FILE="${GATE_DATA}"
export AGZ_RUN_NAME=agz_tmcd_v2_full_qwen3.5-4b_vda_stable_gate
export AGZ_CHECKPOINT_DIR="${GATE_ROOT}/checkpoint"
export AGZ_MAX_STEPS=1
export AGZ_RESUME_MODE=disable
export AGZ_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export AGZ_BACKBONE=qwen3.5-4b
export AGZ_EXPERIMENT_VARIANT=full
export AGZ_N_GPUS_PER_NODE=4
export AGZ_BATCH_SIZE=32
export AGZ_PPO_MINI_BATCH_SIZE=32
export AGZ_ROLLOUT_N=1
export AGZ_ADV_ESTIMATOR=reinforce_plus_plus
export AGZ_TOOL_SERVER_MODE=level1
export AGZ_BUILD_SMOKE_DATASET=0
export AGZ_AGENT_MAX_TURNS=16
export AGZ_MAX_PROMPT_LENGTH=2048
export AGZ_MAX_RESPONSE_LENGTH=11264
export AGZ_MAX_MODEL_LENGTH=15360
export AGZ_MAX_ACTION_LENGTH=320
export AGZ_MAX_OBS_LENGTH=1280
export AGZ_GPU_MEMORY_UTILIZATION=0.35
export AGZ_MAX_NUM_SEQS=8
export AGZ_LORA_RANK=16
export AGZ_LORA_ALPHA=32
export AGZ_LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
export AGZ_ACTOR_LR=2e-5
export AGZ_ACTOR_CPU_OFFLOAD=false
export AGZ_ACTOR_PARAM_OFFLOAD=false
export AGZ_ACTOR_OPTIMIZER_OFFLOAD=false
export AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_REF_PARAM_OFFLOAD=false
export AGZ_RESHARD_AFTER_FORWARD=true
export AGZ_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AGZ_SEED=20260709
export AGZ_VAL_BEFORE_TRAIN=False
export AGZ_DATA_SHUFFLE=false
export AGZ_SAVE_FREQ=1
export AGZ_TEST_FREQ=0

/usr/bin/bash "${ROOT}/scripts/train_vda_qwen35_lora.sh"
touch "${GATE_ROOT}/SUCCEEDED"
