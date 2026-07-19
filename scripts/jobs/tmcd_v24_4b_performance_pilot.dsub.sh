#!/bin/bash
#DSUB -n AGZV24_4B_PERF
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v24_pilot/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v24_pilot/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
case "$(hostname)" in
  cyclone001-agent-175|cyclone001-agent-208|cyclone001-agent-217) ;;
  *)
    echo "Refusing to run outside nodes 175/208/217: $(hostname)" >&2
    exit 72
    ;;
esac

mkdir -p "${ROOT}/logs/tmcd_v24_pilot" "${ROOT}/outputs/tmcd_v24_pilot/preflight"
export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

NODE_SUFFIX=$(hostname)
NODE_SUFFIX=${NODE_SUFFIX#cyclone001-}
python -s "${ROOT}/scripts/preflight_tmcd_v2_job.py" \
  --backbone qwen3.5-4b \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --variant full \
  --expected-node "${NODE_SUFFIX}" \
  --output "${ROOT}/outputs/tmcd_v24_pilot/preflight/${NODE_SUFFIX}.json"

DATA_DIR="${ROOT}/data/tmcd_v24_pilot"
TRAIN="${DATA_DIR}/vda_train_320.parquet"
mkdir -p "${DATA_DIR}"
python -s - "${TRAIN}" <<'PY'
import pathlib
import sys

import pandas as pd

from agentguard_zero.training.vda_dataset import scenario_to_training_row
from scripts.smoke_tmcd_v2_protocol import make_scenario

target = pathlib.Path(sys.argv[1])
rows = []
for task_id in ("T1", "T2", "T3", "T4"):
    for index in range(80):
        rows.append(scenario_to_training_row(make_scenario(task_id, index), split="train"))
pd.DataFrame(rows).to_parquet(target, index=False)
print({"rows": len(rows), "output": str(target)})
PY

export AGZ_MODEL_PATH="${AGZ_QWEN35_4B_PATH}"
export AGZ_TRAIN_FILE="${TRAIN}"
export AGZ_VAL_FILE="${TRAIN}"
export AGZ_MAX_STEPS=10
export AGZ_RESUME_MODE=disable
export AGZ_CUDA_VISIBLE_DEVICES=0,1,2,3
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
export AGZ_LORA_RANK=16
export AGZ_LORA_ALPHA=32
export AGZ_LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
export AGZ_ACTOR_LR=2e-5
export AGZ_ACTOR_CPU_OFFLOAD=false
export AGZ_ACTOR_PARAM_OFFLOAD=false
export AGZ_ACTOR_OPTIMIZER_OFFLOAD=false
export AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_REF_PARAM_OFFLOAD=false
export AGZ_RESHARD_AFTER_FORWARD=true
export AGZ_ENABLE_GRADIENT_CHECKPOINTING=true
export AGZ_SEED=20260709
export AGZ_VAL_BEFORE_TRAIN=False
export AGZ_DATA_SHUFFLE=false
export AGZ_SAVE_FREQ=10
export AGZ_MAX_ACTOR_CKPT_TO_KEEP=1
export AGZ_TEST_FREQ=0
export AGZ_STOP_ON_COMPLETE_JSON=true
export AGZ_REQUIRE_TRAJECTORY_REWARD=1
export AGZ_ROLLOUT_SERVER_MAX_PARALLEL_TRAJECTORIES=8

run_pilot() {
  local name=$1
  local backend=$2
  local workers=$3
  local max_num_seqs=$4
  local utilization=$5
  local logprob_microbatch=$6
  local output="${ROOT}/outputs/tmcd_v24_pilot/${name}"
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite pilot output: ${output}" >&2
    return 73
  fi
  mkdir -p "${output}"
  export AGZ_RUN_NAME="agz_v24_${name}"
  export AGZ_CHECKPOINT_DIR="${output}/checkpoints"
  export AGZ_ROLLOUT_BACKEND="${backend}"
  export AGZ_AGENT_NUM_WORKERS="${workers}"
  export AGZ_MAX_NUM_SEQS="${max_num_seqs}"
  export AGZ_GPU_MEMORY_UTILIZATION="${utilization}"
  export AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${logprob_microbatch}"
  if [[ "${backend}" == "vllm" ]]; then
    export AGZ_ROLLOUT_TOP_K=-1
  else
    export AGZ_ROLLOUT_TOP_K=0
  fi
  echo "PILOT_START name=${name} backend=${backend} workers=${workers} max_num_seqs=${max_num_seqs} utilization=${utilization} logprob_microbatch=${logprob_microbatch}"
  bash "${ROOT}/scripts/train_vda_qwen35_lora.sh"
  python -s "${ROOT}/scripts/validate_vda_training_log.py" \
    --log "${ROOT}/logs/${AGZ_RUN_NAME}.log" \
    --output "${output}/validation.json" \
    --expected-step 10 \
    --action-budget 320 \
    --observation-budget 1280
  echo "PILOT_SUCCESS name=${name}"
}

run_pilot hf_safe hf 1 8 0.35 1
run_pilot hf_workers4 hf 4 8 0.35 1
run_pilot hf_concurrent hf 4 24 0.50 1
run_pilot hf_logprob2 hf 4 24 0.50 2
run_pilot vllm_concurrent vllm 4 24 0.50 2

touch "${ROOT}/outputs/tmcd_v24_pilot/PILOT_SUCCEEDED"
