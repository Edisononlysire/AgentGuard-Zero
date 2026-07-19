#!/bin/bash
#DSUB -n AGZFastDcaStep
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
EXPECTED_NODE=${AGZ_EXPECTED_NODE:-cyclone001-agent-217}
ROUND_DIR=${ROOT}/data/co_evolution/qwen3.5-4b/round_1
CHECKPOINT_DIR=${ROOT}/checkpoints/qwen3.5-4b/dca/round_1/trainer
FEEDBACK_LOG=${ROUND_DIR}/dca_feedback/feedback.jsonl

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

source "${ROOT}/scripts/qwen35_env.sh"
python -s -c 'from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available; assert is_fast_path_available; print("Qwen3.5 fast path enabled")'
python -s -c "from pathlib import Path; from scripts.run_dca_first_round import _reconcile_feedback_log; _reconcile_feedback_log(feedback_log_path=Path('${FEEDBACK_LOG}'), checkpoint_root=Path('${CHECKPOINT_DIR}'), parent_step=0, batch_size=16, rollout_n=2)"

export AGZ_ROOT=${ROOT}
export AGZ_MODEL_PATH=${AGZ_QWEN35_4B_PATH}
export AGZ_VDA_MODEL_PATH=${AGZ_QWEN35_4B_PATH}
export AGZ_VDA_ADAPTER_PATH=
export AGZ_TRAIN_FILE=${ROUND_DIR}/dca_feedback/prompts.parquet
export AGZ_VAL_FILE=${ROUND_DIR}/dca_feedback/prompts.parquet
export AGZ_DCA_FEEDBACK_LOG=${FEEDBACK_LOG}
export AGZ_RUN_NAME=agz_qwen3.5-4b_dca_r1_fastpath_gate
export AGZ_CHECKPOINT_DIR=${CHECKPOINT_DIR}
export AGZ_MAX_STEPS=4
export AGZ_RESUME_MODE=auto
export AGZ_RESUME_FROM_PATH=null
export AGZ_ALLOCATED_GPU_IDS=0,1,2,3
export AGZ_BATCH_SIZE=16
export AGZ_PPO_MINI_BATCH_SIZE=16
export AGZ_ROLLOUT_N=2
export AGZ_MAX_PROMPT_LENGTH=896
export AGZ_MAX_RESPONSE_LENGTH=512
export AGZ_ROLLOUT_TEMPERATURE=0.7
export AGZ_ROLLOUT_TOP_P=1.0
export AGZ_ROLLOUT_TOP_K=0
export AGZ_SEED=20260709
export AGZ_VDA_FEEDBACK_PORT_A=31501
export AGZ_VDA_FEEDBACK_MAX_TURNS=16
export AGZ_DCA_REQUIRE_VALID_BATCH=0
export AGZ_SAVE_FREQ=25
export AGZ_MAX_ACTOR_CKPT_TO_KEEP=1
export AGZ_DYNAMIC_BSZ=false
export AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU=4
export AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4

/usr/bin/bash "${ROOT}/scripts/train_dca_qwen35_lora.sh"

test -d "${CHECKPOINT_DIR}/global_step_4"
test "$(wc -l < "${FEEDBACK_LOG}")" -eq 128
echo "Fast-path DCA microbatch gate passed"
