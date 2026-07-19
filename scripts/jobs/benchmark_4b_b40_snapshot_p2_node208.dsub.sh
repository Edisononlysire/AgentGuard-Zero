#!/bin/bash
#DSUB -n AGZ4B_B40_SNAPSHOT_P2_GATE
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-208
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

GATE_ROOT="${ROOT}/outputs/optimization_gates/qwen3.5-4b-dca-b40-n2-snapshot-p2"
if [[ -e "${GATE_ROOT}" ]]; then
  echo "Refusing to overwrite existing gate: ${GATE_ROOT}" >&2
  exit 73
fi
mkdir -p "${GATE_ROOT}"
PROMPTS="${GATE_ROOT}/prompts.parquet"
python -s - "${PROMPTS}" <<'PY'
import sys
from agentguard_zero.training.dca_dataset import write_dca_prompt_dataset
write_dca_prompt_dataset(
    sys.argv[1],
    num_rows=80,
    seed=20260709,
    backbone="qwen3.5-4b",
    source_round=0,
)
PY

export AGZ_BACKBONE=qwen3.5-4b
export AGZ_MODEL_PATH="${AGZ_QWEN35_4B_PATH}"
export AGZ_VDA_MODEL_PATH="${AGZ_QWEN35_4B_PATH}"
export AGZ_VDA_ADAPTER_PATH=
export AGZ_TRAIN_FILE="${PROMPTS}"
export AGZ_VAL_FILE="${PROMPTS}"
export AGZ_DCA_FEEDBACK_LOG="${GATE_ROOT}/feedback.jsonl"
export AGZ_RUN_NAME=agz_gate_qwen3.5-4b_dca_b40_n2_snapshot_p2
export AGZ_CHECKPOINT_DIR="${GATE_ROOT}/trainer"
export AGZ_MAX_STEPS=2
export AGZ_RESUME_MODE=disable
export AGZ_RESUME_FROM_PATH=null
export AGZ_ALLOCATED_GPU_IDS="${CUDA_VISIBLE_DEVICES}"
export AGZ_BATCH_SIZE=40
export AGZ_PPO_MINI_BATCH_SIZE=40
export AGZ_ROLLOUT_N=2
export AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_MAX_PROMPT_LENGTH=2048
export AGZ_MAX_RESPONSE_LENGTH=1024
export AGZ_VDA_FEEDBACK_MAX_TURNS=16
export AGZ_VDA_FEEDBACK_MAX_INPUT_TOKENS=4096
export AGZ_VDA_FEEDBACK_MAX_NEW_TOKENS=384
export AGZ_VDA_FEEDBACK_ATTN_IMPLEMENTATION=flash_attention_2
export AGZ_VDA_FEEDBACK_CONTINUATION_PROMPT_MODE=snapshot
export AGZ_VDA_FEEDBACK_INVALID_ACTION_PATIENCE=2
export AGZ_VDA_FEEDBACK_OFFLOAD=1
export AGZ_RESHARD_AFTER_FORWARD=true
export AGZ_DCA_REQUIRE_VALID_BATCH=0
export AGZ_SAVE_FREQ=2
export AGZ_MAX_ACTOR_CKPT_TO_KEEP=1
export AGZ_SEED=20260709

/usr/bin/bash "${ROOT}/scripts/train_dca_qwen35_lora.sh"
test "$(wc -l < "${GATE_ROOT}/feedback.jsonl")" -eq 160
echo "4B batch40 snapshot-p2 gate passed"
