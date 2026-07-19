#!/bin/bash
#DSUB -n AGZV2_4B_APPEND_VDA
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=${AGZ_GATE_EXPECTED_NODE:-cyclone001-agent-208}
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

OUT=${AGZ_GATE_OUTPUT_DIR:-${ROOT}/outputs/tmcd_v2/gates/vda-append4-b160}
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite existing VDA gate: ${OUT}" >&2
  exit 73
fi
mkdir -p "${OUT}" "${ROOT}/logs/gates"

export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"

export AGZ_MODEL_PATH="${AGZ_QWEN35_4B_PATH}"
export AGZ_TRAIN_FILE=${AGZ_GATE_TRAIN_FILE:-${ROOT}/data/tmcd_v2/ablations/append_only_memory/qwen3.5-4b/round_1/vda_train/train.parquet}
export AGZ_VAL_FILE=${AGZ_GATE_VAL_FILE:-${ROOT}/data/tmcd_v2/ablations/append_only_memory/qwen3.5-4b/round_1/vda_dev/dev.parquet}
export AGZ_RUN_NAME=${AGZ_GATE_RUN_NAME:-agz_gate_qwen3.5-4b_append_vda_batch160_prompt_v5}
export AGZ_CHECKPOINT_DIR="${OUT}/checkpoints"
export AGZ_MAX_STEPS=1
export AGZ_RESUME_MODE=disable
export AGZ_RESUME_FROM_PATH=null
export AGZ_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export AGZ_BACKBONE=qwen3.5-4b
export AGZ_EXPERIMENT_VARIANT=append_only_memory
export AGZ_N_GPUS_PER_NODE=4
export AGZ_BATCH_SIZE=160
export AGZ_PPO_MINI_BATCH_SIZE=40
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
export AGZ_MAX_NUM_SEQS=40
export AGZ_LORA_RANK=16
export AGZ_LORA_ALPHA=32
export AGZ_LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
export AGZ_ACTOR_LR=2e-5
export AGZ_LR_WARMUP_STEPS=0
export AGZ_ACTOR_CPU_OFFLOAD=false
export AGZ_ACTOR_PARAM_OFFLOAD=false
export AGZ_ACTOR_OPTIMIZER_OFFLOAD=false
export AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
export AGZ_REF_PARAM_OFFLOAD=false
export AGZ_RESHARD_AFTER_FORWARD=true
export AGZ_SEED=20260709
export AGZ_VAL_BEFORE_TRAIN=False
export AGZ_DATA_SHUFFLE=false
export AGZ_SAVE_FREQ=1
export AGZ_TEST_FREQ=0

bash "${ROOT}/scripts/train_vda_qwen35_lora.sh"

python -s "${ROOT}/scripts/validate_vda_training_log.py" \
  --log "${ROOT}/logs/${AGZ_RUN_NAME}.log" \
  --output "${OUT}/training_metrics.json" \
  --expected-step 1 \
  --action-budget 320 \
  --observation-budget 1280

python -s "${ROOT}/scripts/finalize_vda_training_gate.py" \
  --output-dir "${OUT}" \
  --backbone qwen3.5-4b \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --parent-manifest "${ROOT}/checkpoints/tmcd_v2/ablations/append_only_memory/qwen3.5-4b/vda/round_0/manifest.json" \
  --pool-manifest "${AGZ_GATE_POOL_MANIFEST:-${ROOT}/data/tmcd_v2/ablations/append_only_memory/qwen3.5-4b/round_1/vda_pool_manifest.json}" \
  --dca-manifest "${ROOT}/checkpoints/tmcd_v2/ablations/append_only_memory/qwen3.5-4b/dca/round_1/manifest.json" \
  --checkpoint-root "${OUT}/checkpoints" \
  --batch-size 160 \
  --seed 20260709

CUDA_VISIBLE_DEVICES=0 python -s "${ROOT}/scripts/validate_adapter_reload.py" \
  --checkpoint-manifest "${OUT}/checkpoint_manifest.json" \
  --output "${OUT}/adapter_reload.json"

python -s "${ROOT}/scripts/prune_gate_recovery_checkpoint.py" \
  --checkpoint-manifest "${OUT}/checkpoint_manifest.json" \
  --output "${OUT}/recovery_pruned.json"

touch "${OUT}/SUCCEEDED"
