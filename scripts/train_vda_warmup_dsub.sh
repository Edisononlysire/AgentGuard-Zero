#!/bin/bash
#DSUB -n AGZVDAWarmup
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
# Expose all local GPUs because the cluster scheduler may assign a busy
# GPU when only gpu=1 is requested. The training script then selects the
# least busy GPUs by live nvidia-smi free-memory checks.
#DSUB -R "cpu=16;gpu=4;mem=180000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
mkdir -p "${ROOT}/logs" "${ROOT}/data/smoke" "${ROOT}/data/level1" "${ROOT}/outputs/checkpoints"

export AGZ_ROOT="${ROOT}"
export AGZ_MODEL_PATH="${AGZ_MODEL_PATH:-${ROOT}/data/smoke/tiny-qwen3}"
export AGZ_RUN_NAME="agentguard_vda_warmup_${BATCH_JOB_ID:-manual}"
export AGZ_CHECKPOINT_DIR="${AGZ_CHECKPOINT_DIR:-${ROOT}/outputs/checkpoints/${AGZ_RUN_NAME}}"
export AGZ_LEVEL1_FRONTIER_FILE="${AGZ_LEVEL1_FRONTIER_FILE:-${ROOT}/data/level1/level1_seed20260706_n500_frontier_vda.parquet}"
export AGZ_TRAIN_FILE="${AGZ_TRAIN_FILE:-${AGZ_LEVEL1_FRONTIER_FILE}}"
export AGZ_VAL_FILE="${AGZ_VAL_FILE:-${AGZ_TRAIN_FILE}}"
export AGZ_TOOL_SERVER_MODE="${AGZ_TOOL_SERVER_MODE:-level1}"
export AGZ_MAX_STEPS="${AGZ_MAX_STEPS:-2}"
export AGZ_N_GPUS_PER_NODE="${AGZ_N_GPUS_PER_NODE:-1}"
export AGZ_ROLLOUT_N="${AGZ_ROLLOUT_N:-1}"
export AGZ_BATCH_SIZE="${AGZ_BATCH_SIZE:-1}"
export AGZ_PPO_MINI_BATCH_SIZE="${AGZ_PPO_MINI_BATCH_SIZE:-1}"
export AGZ_RAY_NUM_CPUS="${AGZ_RAY_NUM_CPUS:-8}"
export AGZ_MAX_PROMPT_LENGTH="${AGZ_MAX_PROMPT_LENGTH:-2048}"
export AGZ_MAX_RESPONSE_LENGTH="${AGZ_MAX_RESPONSE_LENGTH:-256}"
export AGZ_MAX_OBS_LENGTH="${AGZ_MAX_OBS_LENGTH:-384}"
export AGZ_AGENT_MAX_TURNS="${AGZ_AGENT_MAX_TURNS:-3}"
export AGZ_GPU_MEMORY_UTILIZATION="${AGZ_GPU_MEMORY_UTILIZATION:-0.10}"
export AGZ_MAX_NUM_SEQS="${AGZ_MAX_NUM_SEQS:-2}"
export AGZ_GPU_MIN_FREE_MB="${AGZ_GPU_MIN_FREE_MB:-16000}"
export AGZ_BUILD_SMOKE_DATASET="${AGZ_BUILD_SMOKE_DATASET:-0}"
export AGZ_FORCE_REBUILD_SMOKE="${AGZ_FORCE_REBUILD_SMOKE:-0}"
export AGZ_SMOKE_ROWS="${AGZ_SMOKE_ROWS:-${AGZ_BATCH_SIZE}}"
export AGZ_VAL_BEFORE_TRAIN="${AGZ_VAL_BEFORE_TRAIN:-False}"
export AGZ_SAVE_FREQ="${AGZ_SAVE_FREQ:-1}"
export AGZ_TEST_FREQ="${AGZ_TEST_FREQ:-0}"
export AGZ_ACTOR_CPU_OFFLOAD="${AGZ_ACTOR_CPU_OFFLOAD:-false}"
export AGZ_ACTOR_PARAM_OFFLOAD="${AGZ_ACTOR_PARAM_OFFLOAD:-false}"
export AGZ_ACTOR_OPTIMIZER_OFFLOAD="${AGZ_ACTOR_OPTIMIZER_OFFLOAD:-false}"
export AGZ_REF_PARAM_OFFLOAD="${AGZ_REF_PARAM_OFFLOAD:-false}"
export AGZ_ACTOR_USE_ORIG_PARAMS="${AGZ_ACTOR_USE_ORIG_PARAMS:-true}"

cd "${ROOT}"
if [[ ! -f "${AGZ_MODEL_PATH}/config.json" ]]; then
  /usr/bin/bash -lc "source ${ROOT}/scripts/agentguard_env.sh && python -s ${ROOT}/scripts/build_tiny_qwen3_smoke_model.py --base /home/share/huadjyin/home/s_qinhua2/02code/guozhihan/InfectModel/model/Qwen/Qwen3-8B --output ${AGZ_MODEL_PATH}"
fi
/usr/bin/bash "${ROOT}/scripts/train_vda_warmup_smoke.sh"
