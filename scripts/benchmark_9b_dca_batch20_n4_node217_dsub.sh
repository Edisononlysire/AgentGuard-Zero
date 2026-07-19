#!/bin/bash
#DSUB -n AGZ9BDCA20N4Gate
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
if [[ "$(hostname)" != "cyclone001-agent-217" ]]; then
  echo "Refusing to run outside cyclone001-agent-217: $(hostname)" >&2
  exit 72
fi
source "${ROOT}/scripts/qwen35_env.sh"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

GATE_ROOT="${ROOT}/outputs/optimization_gates/qwen3.5-9b-dca-b20-n4-m4"
PROMPTS="${GATE_ROOT}/prompts.parquet"
mkdir -p "${GATE_ROOT}"
python -s - "${PROMPTS}" <<'PY'
import sys
from agentguard_zero.training.dca_dataset import write_dca_prompt_dataset
write_dca_prompt_dataset(sys.argv[1], num_rows=20, seed=20260709, backbone="qwen3.5-9b", source_round=0)
PY

export AGZ_ROOT="${ROOT}"
export AGZ_BACKBONE=qwen3.5-9b
export AGZ_MODEL_PATH="${AGZ_QWEN35_9B_PATH}"
export AGZ_VDA_MODEL_PATH="${AGZ_QWEN35_9B_PATH}"
export AGZ_TRAIN_FILE="${PROMPTS}"
export AGZ_VAL_FILE="${PROMPTS}"
export AGZ_DCA_FEEDBACK_LOG="${GATE_ROOT}/feedback.jsonl"
export AGZ_RUN_NAME=agz_gate_qwen3.5-9b_dca_b20_n4_m4
export AGZ_CHECKPOINT_DIR="${GATE_ROOT}/trainer"
export AGZ_MAX_STEPS=1
export AGZ_RESUME_MODE=disable
export AGZ_RESUME_FROM_PATH=null
export AGZ_ALLOCATED_GPU_IDS=0,1,2,3
export AGZ_BATCH_SIZE=20
export AGZ_PPO_MINI_BATCH_SIZE=20
export AGZ_ROLLOUT_N=4
export AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU=4
export AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
export AGZ_VDA_FEEDBACK_MAX_TURNS=4
export AGZ_VDA_FEEDBACK_MAX_INPUT_TOKENS=2048
export AGZ_SAVE_FREQ=1

/usr/bin/bash "${ROOT}/scripts/train_dca_qwen35_lora.sh"
