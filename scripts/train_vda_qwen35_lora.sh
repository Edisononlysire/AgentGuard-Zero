#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/env.sh"
ROOT=${AGZ_ROOT}
export AGZ_DISABLE_THINKING=${AGZ_DISABLE_THINKING:-1}

MODEL_PATH=${AGZ_MODEL_PATH:?AGZ_MODEL_PATH is required}
TRAIN_FILE=${AGZ_TRAIN_FILE:-${ROOT}/data/smoke/vda_train.parquet}
VAL_FILE=${AGZ_VAL_FILE:-${TRAIN_FILE}}
BUILD_SMOKE_DATASET=${AGZ_BUILD_SMOKE_DATASET:-}
if [[ -z "${BUILD_SMOKE_DATASET}" ]]; then
  BUILD_SMOKE_DATASET=0
  if [[ "${TRAIN_FILE}" == "${ROOT}/data/smoke/"* ]]; then
    BUILD_SMOKE_DATASET=1
  fi
fi
RUN_NAME=${AGZ_RUN_NAME:-agentguard_vda_warmup_smoke}
CHECKPOINT_DIR=${AGZ_CHECKPOINT_DIR:-${ROOT}/outputs/checkpoints/${RUN_NAME}}
TOOL_SERVER_MODE=${AGZ_TOOL_SERVER_MODE:-smoke}
MAX_STEPS=${AGZ_MAX_STEPS:-2}
VAL_BEFORE_TRAIN=${AGZ_VAL_BEFORE_TRAIN:-False}
SAVE_FREQ=${AGZ_SAVE_FREQ:-1}
MAX_ACTOR_CKPT_TO_KEEP=${AGZ_MAX_ACTOR_CKPT_TO_KEEP:-1}
TEST_FREQ=${AGZ_TEST_FREQ:-0}
SEED=${AGZ_SEED:-20260709}
RESUME_MODE=${AGZ_RESUME_MODE:-auto}
RESUME_FROM_PATH=${AGZ_RESUME_FROM_PATH:-null}
N_GPUS_PER_NODE=${AGZ_N_GPUS_PER_NODE:-2}
GPU_MIN_FREE_MB=${AGZ_GPU_MIN_FREE_MB:-0}
if [[ -n "${AGZ_CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${AGZ_CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  ORIGINAL_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}
  echo "Original CUDA_VISIBLE_DEVICES=${ORIGINAL_CUDA_VISIBLE_DEVICES}"
  echo "GPU memory snapshot before selection:"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
  CUDA_VISIBLE_DEVICES=$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F, -v min_free="${GPU_MIN_FREE_MB}" -v allowed="${ORIGINAL_CUDA_VISIBLE_DEVICES}" '
          BEGIN {
            use_allowed = (allowed != "" && allowed != "unset")
            if (use_allowed) {
              n = split(allowed, ids, ",")
              for (i = 1; i <= n; i++) {
                gsub(/ /, "", ids[i])
                allowed_ids[ids[i]] = 1
              }
            }
          }
          {
            gsub(/ /, "", $1)
            gsub(/ /, "", $2)
            if ((!use_allowed || ($1 in allowed_ids)) && (($2 + 0) >= min_free)) print $1 "," $2
          }' \
      | sort -t, -k2 -nr \
      | head -n "${N_GPUS_PER_NODE}" \
      | awk -F, '{print $1}' \
      | paste -sd, -
  )
  if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    echo "No GPU satisfies AGZ_GPU_MIN_FREE_MB=${GPU_MIN_FREE_MB}; not starting warmup." >&2
    exit 42
  fi
  SELECTED_GPU_COUNT=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  if (( SELECTED_GPU_COUNT < N_GPUS_PER_NODE )); then
    echo "Only selected ${SELECTED_GPU_COUNT} GPU(s), but AGZ_N_GPUS_PER_NODE=${N_GPUS_PER_NODE}; not starting warmup." >&2
    exit 48
  fi
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=0,1
fi
ROLLOUT_N=${AGZ_ROLLOUT_N:-2}
ROLLOUT_TEMPERATURE=${AGZ_ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_TOP_P=${AGZ_ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${AGZ_ROLLOUT_TOP_K:-0}
BATCH_SIZE=${AGZ_BATCH_SIZE:-1}
GEN_BATCH_SIZE=${AGZ_GEN_BATCH_SIZE:-${BATCH_SIZE}}
DATA_SHUFFLE=${AGZ_DATA_SHUFFLE:-true}
PPO_MINI_BATCH_SIZE=${AGZ_PPO_MINI_BATCH_SIZE:-1}
MAX_PROMPT_LENGTH=${AGZ_MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${AGZ_MAX_RESPONSE_LENGTH:-1024}
MAX_ACTION_LENGTH=${AGZ_MAX_ACTION_LENGTH:-${MAX_RESPONSE_LENGTH}}
MAX_OBS_LENGTH=${AGZ_MAX_OBS_LENGTH:-${MAX_RESPONSE_LENGTH}}
DEFAULT_AGENT_MAX_TURNS=1
if [[ "${TOOL_SERVER_MODE}" == "level1" ]]; then
  DEFAULT_AGENT_MAX_TURNS=3
fi
AGENT_MAX_TURNS=${AGZ_AGENT_MAX_TURNS:-${DEFAULT_AGENT_MAX_TURNS}}
MAX_MODEL_LENGTH=${AGZ_MAX_MODEL_LENGTH:-$((MAX_PROMPT_LENGTH + (AGENT_MAX_TURNS + 1) * MAX_ACTION_LENGTH + AGENT_MAX_TURNS * MAX_OBS_LENGTH))}
GPU_MEMORY_UTILIZATION=${AGZ_GPU_MEMORY_UTILIZATION:-0.45}
MAX_NUM_SEQS=${AGZ_MAX_NUM_SEQS:-16}
HF_MAX_BATCH_TOKENS=${AGZ_HF_MAX_BATCH_TOKENS:-196608}
TENSOR_MODEL_PARALLEL_SIZE=${AGZ_TENSOR_MODEL_PARALLEL_SIZE:-1}
USE_KL_LOSS=${AGZ_USE_KL_LOSS:-false}
KL_LOSS_COEF=${AGZ_KL_LOSS_COEF:-0.01}
ADV_ESTIMATOR=${AGZ_ADV_ESTIMATOR:-reinforce_plus_plus}
RAY_NUM_CPUS=${AGZ_RAY_NUM_CPUS:-8}
AGENT_NUM_WORKERS=${AGZ_AGENT_NUM_WORKERS:-1}
ACTOR_MODEL_DTYPE=${AGZ_ACTOR_MODEL_DTYPE:-bf16}
ROLLOUT_BACKEND=${AGZ_ROLLOUT_BACKEND:-hf}
ACTOR_CPU_OFFLOAD=${AGZ_ACTOR_CPU_OFFLOAD:-true}
ACTOR_USE_ORIG_PARAMS=${AGZ_ACTOR_USE_ORIG_PARAMS:-true}
ACTOR_PARAM_OFFLOAD=${AGZ_ACTOR_PARAM_OFFLOAD:-false}
ACTOR_OPTIMIZER_OFFLOAD=${AGZ_ACTOR_OPTIMIZER_OFFLOAD:-false}
REF_PARAM_OFFLOAD=${AGZ_REF_PARAM_OFFLOAD:-false}
ACTOR_LR=${AGZ_ACTOR_LR:-1e-6}
LR_WARMUP_STEPS=${AGZ_LR_WARMUP_STEPS:-1}
LORA_RANK=${AGZ_LORA_RANK:-0}
LORA_ALPHA=${AGZ_LORA_ALPHA:-16}
LORA_TARGET_MODULES=${AGZ_LORA_TARGET_MODULES:-all-linear}
PPO_MICRO_BATCH_SIZE_PER_GPU=${AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
USE_TORCH_COMPILE=${AGZ_USE_TORCH_COMPILE:-False}
DYNAMIC_BSZ=${AGZ_DYNAMIC_BSZ:-false}
RESHARD_AFTER_FORWARD=${AGZ_RESHARD_AFTER_FORWARD:-false}
HF_FULL_ROLLOUT_REPLICA=${AGZ_HF_FULL_ROLLOUT_REPLICA:-true}
STOP_ON_COMPLETE_JSON=${AGZ_STOP_ON_COMPLETE_JSON:-true}
ENABLE_GRADIENT_CHECKPOINTING=${AGZ_ENABLE_GRADIENT_CHECKPOINTING:-true}
ROLLOUT_SERVER_MAX_PARALLEL_TRAJECTORIES=${AGZ_ROLLOUT_SERVER_MAX_PARALLEL_TRAJECTORIES:-8}
ROLLOUT_SERVER_MAX_STATES=${AGZ_ROLLOUT_SERVER_MAX_STATES:-512}

export CUDA_VISIBLE_DEVICES
export VERL_RUN_ID="${RUN_NAME}"
export AGZ_TRITON_CACHE_NAMESPACE="verl_$(basename "${MODEL_PATH}")_vda"
# vLLM 0.9.1 CuMemAllocator rejects expandable_segments. Keep this unset by
# default; allow an explicit override only for non-vLLM debugging runs.
if [[ -n "${AGZ_PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${AGZ_PYTORCH_CUDA_ALLOC_CONF}"
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi
export RAY_memory_usage_threshold=${AGZ_RAY_MEMORY_USAGE_THRESHOLD:-0.999}
export RAY_memory_monitor_refresh_ms=${AGZ_RAY_MEMORY_MONITOR_REFRESH_MS:-0}

echo "Running host=$(hostname)"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using actor fsdp model dtype=${ACTOR_MODEL_DTYPE}"
echo "Using rollout backend=${ROLLOUT_BACKEND}"
echo "Using train batch_size=${BATCH_SIZE} generation batch_size=${GEN_BATCH_SIZE}"
echo "Using actor cpu_offload=${ACTOR_CPU_OFFLOAD}"
echo "Using actor param_offload=${ACTOR_PARAM_OFFLOAD}"
echo "Using actor optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
echo "Using ref param_offload=${REF_PARAM_OFFLOAD}"
echo "Using actor use_orig_params=${ACTOR_USE_ORIG_PARAMS}"
echo "Using actor lr=${ACTOR_LR}"
echo "Using advantage estimator=${ADV_ESTIMATOR}"
echo "Using actor KL loss=${USE_KL_LOSS} coefficient=${KL_LOSS_COEF}"
echo "Using lr warmup steps=${LR_WARMUP_STEPS}"
echo "Using LoRA rank=${LORA_RANK}"
echo "Using LoRA alpha=${LORA_ALPHA}"
echo "Using LoRA target_modules=${LORA_TARGET_MODULES}"
echo "Using torch compile=${USE_TORCH_COMPILE}"
echo "Using gradient checkpointing=${ENABLE_GRADIENT_CHECKPOINTING}"
echo "Using agent num_workers=${AGENT_NUM_WORKERS}"
echo "Using rollout server max_parallel_trajectories=${ROLLOUT_SERVER_MAX_PARALLEL_TRAJECTORIES}"
echo "Using tool server mode=${TOOL_SERVER_MODE}"
echo "Using agent max_turns=${AGENT_MAX_TURNS}"
echo "Using agent lengths: start=${MAX_PROMPT_LENGTH}, action=${MAX_ACTION_LENGTH}, obs=${MAX_OBS_LENGTH}, model=${MAX_MODEL_LENGTH}"
echo "Using train file=${TRAIN_FILE}"
echo "Using val file=${VAL_FILE}"
echo "Using build smoke dataset=${BUILD_SMOKE_DATASET}"
echo "Using checkpoint dir=${CHECKPOINT_DIR}"
echo "Using val_before_train=${VAL_BEFORE_TRAIN}"
echo "Using save_freq=${SAVE_FREQ}"
echo "Keeping ${MAX_ACTOR_CKPT_TO_KEEP} actor checkpoint(s)"
echo "Using test_freq=${TEST_FREQ}"
echo "Using seed=${SEED}"
echo "Using resume_mode=${RESUME_MODE}"
echo "Using resume_from_path=${RESUME_FROM_PATH}"

mkdir -p "${ROOT}/data/smoke" "${CHECKPOINT_DIR}" "${ROOT}/logs"
cd "${ROOT}"

MODEL_PATH_ABS=$(readlink -f "${MODEL_PATH}" 2>/dev/null || echo "${MODEL_PATH}")
CHECKPOINT_DIR_ABS=$(readlink -f "${CHECKPOINT_DIR}" 2>/dev/null || echo "${CHECKPOINT_DIR}")
case "${CHECKPOINT_DIR_ABS}" in
  "${MODEL_PATH_ABS}"|"${MODEL_PATH_ABS}"/*)
    echo "Refusing to write checkpoints inside AGZ_MODEL_PATH: ${CHECKPOINT_DIR_ABS}" >&2
    exit 47
    ;;
esac

TOOL_SERVER_URL=${AGZ_TOOL_SERVER_URL:-}
TOOL_SERVER_PID=""
if [[ -z "${TOOL_SERVER_URL}" && "${AGZ_ENABLE_SMOKE_TOOL_SERVER:-1}" == "1" ]]; then
  TOOL_SERVER_HOST=${AGZ_TOOL_SERVER_HOST:-$(hostname -i | awk '{print $1}')}
  TOOL_SERVER_PORT=${AGZ_TOOL_SERVER_PORT:-$(python -s - <<'PY'
import random
print(random.randint(30000, 31000))
PY
)}
  TOOL_SERVER_URL="http://${TOOL_SERVER_HOST}:${TOOL_SERVER_PORT}/get_observation"
  if [[ "${TOOL_SERVER_MODE}" == "smoke" ]]; then
    TOOL_SERVER_SCRIPT="${ROOT}/scripts/vda_smoke_tool_server.py"
    TOOL_SERVER_EXTRA_ARGS=()
  elif [[ "${TOOL_SERVER_MODE}" == "level1" ]]; then
    TOOL_SERVER_SCRIPT="${ROOT}/scripts/level1_rollout_server.py"
    TOOL_SERVER_EXTRA_ARGS=(
      --max-parallel-trajectories "${ROLLOUT_SERVER_MAX_PARALLEL_TRAJECTORIES}"
      --max-states "${ROLLOUT_SERVER_MAX_STATES}"
    )
  else
    echo "Unsupported AGZ_TOOL_SERVER_MODE=${TOOL_SERVER_MODE}; expected smoke or level1." >&2
    exit 44
  fi
  python -s "${TOOL_SERVER_SCRIPT}" \
    --host "${TOOL_SERVER_HOST}" \
    --port "${TOOL_SERVER_PORT}" \
    "${TOOL_SERVER_EXTRA_ARGS[@]}" \
    > "${ROOT}/logs/${RUN_NAME}_tool_server.log" 2>&1 &
  TOOL_SERVER_PID=$!

  cleanup_tool_server() {
    if [[ -n "${TOOL_SERVER_PID}" ]]; then
      kill "${TOOL_SERVER_PID}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_tool_server EXIT

  TOOL_SERVER_HEALTH_URL="http://${TOOL_SERVER_HOST}:${TOOL_SERVER_PORT}/health"
  for attempt in $(seq 1 60); do
    if python -s - "${TOOL_SERVER_HEALTH_URL}" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=1) as resp:
        raise SystemExit(0 if resp.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      break
    fi
    if [[ "${attempt}" == "60" ]]; then
      echo "VDA ${TOOL_SERVER_MODE} tool server did not become healthy: ${TOOL_SERVER_HEALTH_URL}" >&2
      exit 43
    fi
    sleep 1
  done
fi
echo "Using tool server url=${TOOL_SERVER_URL:-null}"

if [[ "${BUILD_SMOKE_DATASET}" == "1" ]] && [[ "${AGZ_FORCE_REBUILD_SMOKE:-0}" == "1" || ! -f "${TRAIN_FILE}" ]]; then
  export PYTHONPATH="${ROOT}:${PYTHONPATH}"
  export AGZ_SMOKE_ROWS="${AGZ_SMOKE_ROWS:-${BATCH_SIZE}}"
  python -s - <<'PY'
import json
import os
from agentguard_zero.schemas.scenario_schema import minimal_example
rows = max(1, int(os.environ.get("AGZ_SMOKE_ROWS", "1")))
with open("data/smoke/minimal_scenarios.json", "w", encoding="utf-8") as f:
    json.dump([{"scenario": minimal_example()} for _ in range(rows)], f, ensure_ascii=False, indent=2)
PY
  python -s "${ROOT}/curriculum/scenario_evaluate/build_vda_dataset.py" \
    "${ROOT}/data/smoke/minimal_scenarios.json" \
    --output "${TRAIN_FILE}"
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Training parquet not found: ${TRAIN_FILE}" >&2
  echo "Set AGZ_TRAIN_FILE to an existing parquet or run scripts/build_level1_frontier.sh." >&2
  exit 45
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Validation parquet not found: ${VAL_FILE}" >&2
  exit 46
fi

PYTHONUNBUFFERED=1 python -s -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=problem \
    data.train_batch_size="${BATCH_SIZE}" \
    data.val_batch_size="${BATCH_SIZE}" \
    data.gen_batch_size="${GEN_BATCH_SIZE}" \
    data.shuffle="${DATA_SHUFFLE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.truncation=right \
    data.seed="${SEED}" \
    ray_init.num_cpus="${RAY_NUM_CPUS}" \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=False \
    custom_reward_function.path="${ROOT}/curriculum/reward_function/vda_reward.py" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward="${RESHARD_AFTER_FORWARD}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
    +actor_rollout_ref.actor.fsdp_config.cpu_offload="${ACTOR_CPU_OFFLOAD}" \
    +actor_rollout_ref.actor.fsdp_config.use_orig_params="${ACTOR_USE_ORIG_PARAMS}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.agent.enable_agent=True \
    actor_rollout_ref.agent.tool_server_url="${TOOL_SERVER_URL}" \
    actor_rollout_ref.agent.max_turns="${AGENT_MAX_TURNS}" \
    actor_rollout_ref.agent.min_turns=0 \
    actor_rollout_ref.agent.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.agent.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.agent.max_start_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.agent.max_action_length="${MAX_ACTION_LENGTH}" \
    actor_rollout_ref.agent.max_obs_length="${MAX_OBS_LENGTH}" \
    actor_rollout_ref.agent.max_model_len="${MAX_MODEL_LENGTH}" \
    actor_rollout_ref.agent.rolling_with_prompt=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND}" \
    +actor_rollout_ref.rollout.use_full_replica="${HF_FULL_ROLLOUT_REPLICA}" \
    +actor_rollout_ref.rollout.stop_on_complete_json="${STOP_ON_COMPLETE_JSON}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    +actor_rollout_ref.rollout.hf_max_batch_tokens="${HF_MAX_BATCH_TOKENS}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH}" \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
    +actor_rollout_ref.rollout.agent.max_start_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    critic.optim.lr=1e-5 \
    critic.strategy=fsdp \
    critic.model.path="${MODEL_PATH}" \
    critic.model.fsdp_config.fsdp_size=-1 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ulysses_sequence_parallel_size=1 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.logger=['console'] \
    trainer.project_name=agentguard_zero \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${MAX_STEPS}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.resume_from_path="${RESUME_FROM_PATH}" \
    2>&1 | tee "${ROOT}/logs/${RUN_NAME}.log"
