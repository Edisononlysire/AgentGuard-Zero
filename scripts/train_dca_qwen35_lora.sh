#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/env.sh"
ROOT=${AGZ_ROOT}
export PYTHONHASHSEED=${AGZ_SEED:-20260709}
export AGZ_DISABLE_THINKING=${AGZ_DISABLE_THINKING:-1}

MODEL_PATH=${AGZ_MODEL_PATH:?AGZ_MODEL_PATH is required}
VDA_MODEL_PATH=${AGZ_VDA_MODEL_PATH:-${MODEL_PATH}}
VDA_ADAPTER_PATH=${AGZ_VDA_ADAPTER_PATH:-}
TRAIN_FILE=${AGZ_TRAIN_FILE:?AGZ_TRAIN_FILE is required}
VAL_FILE=${AGZ_VAL_FILE:-${TRAIN_FILE}}
DCA_FEEDBACK_LOG=${AGZ_DCA_FEEDBACK_LOG:?AGZ_DCA_FEEDBACK_LOG is required}
RUN_NAME=${AGZ_RUN_NAME:-agentguard_dca_lora}
CHECKPOINT_DIR=${AGZ_CHECKPOINT_DIR:?AGZ_CHECKPOINT_DIR is required}
MAX_STEPS=${AGZ_MAX_STEPS:-1}
RESUME_MODE=${AGZ_RESUME_MODE:-disable}
RESUME_FROM_PATH=${AGZ_RESUME_FROM_PATH:-null}
SEED=${AGZ_SEED:-20260709}

ALLOCATED_GPUS=${AGZ_ALLOCATED_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
IFS=',' read -r -a GPU_IDS <<< "${ALLOCATED_GPUS}"
if (( ${#GPU_IDS[@]} != 4 )); then
  echo "DCA-first training requires exactly four GPUs per backbone; got ${ALLOCATED_GPUS}" >&2
  exit 70
fi
DCA_GPU_LAYOUT=${AGZ_DCA_GPU_LAYOUT:-auto}
if [[ "${DCA_GPU_LAYOUT}" == "auto" ]]; then
  DCA_GPU_LAYOUT=shared_fsdp
fi
case "${DCA_GPU_LAYOUT}" in
  dedicated)
    DCA_GPUS=${GPU_IDS[0]}
    DCA_N_GPUS=1
    DCA_PARAM_OFFLOAD=${AGZ_DCA_PARAM_OFFLOAD:-true}
    DCA_OPTIMIZER_OFFLOAD=${AGZ_DCA_OPTIMIZER_OFFLOAD:-true}
    DCA_REF_OFFLOAD=${AGZ_DCA_REF_OFFLOAD:-true}
    ;;
  shared_fsdp)
    DCA_GPUS=$(IFS=,; echo "${GPU_IDS[*]}")
    DCA_N_GPUS=${#GPU_IDS[@]}
    DCA_PARAM_OFFLOAD=${AGZ_DCA_PARAM_OFFLOAD:-false}
    DCA_OPTIMIZER_OFFLOAD=${AGZ_DCA_OPTIMIZER_OFFLOAD:-false}
    if [[ "${AGZ_BACKBONE:-}" == "qwen3.5-9b" ]]; then
      DCA_REF_OFFLOAD=${AGZ_DCA_REF_OFFLOAD:-true}
    else
      DCA_REF_OFFLOAD=${AGZ_DCA_REF_OFFLOAD:-false}
    fi
    ;;
  *)
    echo "Unsupported AGZ_DCA_GPU_LAYOUT=${DCA_GPU_LAYOUT}" >&2
    exit 64
    ;;
esac
PORT_A=${AGZ_VDA_FEEDBACK_PORT_A:-31501}
HOST=${AGZ_VDA_FEEDBACK_HOST:-127.0.0.1}
VDA_FEEDBACK_ATTN_IMPLEMENTATION=${AGZ_VDA_FEEDBACK_ATTN_IMPLEMENTATION:-sdpa}

BATCH_SIZE=${AGZ_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${AGZ_PPO_MINI_BATCH_SIZE:-${BATCH_SIZE}}
ROLLOUT_N=${AGZ_ROLLOUT_N:-2}
ROLLOUT_TEMPERATURE=${AGZ_ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_TOP_P=${AGZ_ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${AGZ_ROLLOUT_TOP_K:-0}
MAX_PROMPT_LENGTH=${AGZ_MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${AGZ_MAX_RESPONSE_LENGTH:-1024}
MAX_MODEL_LENGTH=${AGZ_MAX_MODEL_LENGTH:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ACTOR_LR=${AGZ_ACTOR_LR:-2e-5}
LR_WARMUP_STEPS=${AGZ_LR_WARMUP_STEPS:-1}
LORA_RANK=${AGZ_LORA_RANK:-16}
LORA_ALPHA=${AGZ_LORA_ALPHA:-32}
LORA_TARGET_MODULES=${AGZ_LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}
PPO_MICRO_BATCH_SIZE_PER_GPU=${AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
ENABLE_ACTIVATION_OFFLOAD=${AGZ_ENABLE_ACTIVATION_OFFLOAD:-false}
RAY_NUM_CPUS=${AGZ_RAY_NUM_CPUS:-12}
SAVE_FREQ=${AGZ_SAVE_FREQ:-1}
MAX_ACTOR_CKPT_TO_KEEP=${AGZ_MAX_ACTOR_CKPT_TO_KEEP:-1}
DYNAMIC_BSZ=${AGZ_DYNAMIC_BSZ:-false}
RESHARD_AFTER_FORWARD=${AGZ_RESHARD_AFTER_FORWARD:-false}
HF_FULL_ROLLOUT_REPLICA=${AGZ_HF_FULL_ROLLOUT_REPLICA:-true}
STOP_ON_COMPLETE_JSON=${AGZ_STOP_ON_COMPLETE_JSON:-true}
ENABLE_GRADIENT_CHECKPOINTING=${AGZ_ENABLE_GRADIENT_CHECKPOINTING:-true}
DCA_ROLLOUT_BACKEND=${AGZ_DCA_ROLLOUT_BACKEND:-hf}
DCA_MAX_NUM_SEQS=${AGZ_DCA_MAX_NUM_SEQS:-4}
DCA_GPU_MEMORY_UTILIZATION=${AGZ_DCA_GPU_MEMORY_UTILIZATION:-0.35}

mkdir -p "$(dirname "${DCA_FEEDBACK_LOG}")" "${CHECKPOINT_DIR}" "${ROOT}/logs"

ADAPTER_ARGS=()
if [[ -n "${VDA_ADAPTER_PATH}" && "${VDA_ADAPTER_PATH}" != "null" ]]; then
  ADAPTER_ARGS=(--adapter-path "${VDA_ADAPTER_PATH}")
fi
VDA_LOAD_ARGS=()
if [[ "${AGZ_VDA_FEEDBACK_LAZY_LOAD:-1}" == "1" ]]; then
  VDA_LOAD_ARGS=(--lazy-load)
fi
if [[ "${AGZ_VDA_FEEDBACK_OFFLOAD:-1}" == "1" ]]; then
  VDA_LOAD_ARGS+=(--offload-after-request)
fi

declare -a VDA_PIDS=()
declare -a VDA_URLS=()
for index in "${!GPU_IDS[@]}"; do
  gpu_id=${GPU_IDS[$index]}
  port=$((PORT_A + index))
  url="http://${HOST}:${port}"
  model_cache_key=$(basename "${VDA_MODEL_PATH}" | tr -c 'A-Za-z0-9._-' '_')
  service_cache="${AGZ_TRITON_CACHE_ROOT}/vda_feedback/${model_cache_key}/rank_${index}"
  mkdir -p "${service_cache}"
  TRITON_CACHE_DIR="${service_cache}" CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python -s "${ROOT}/scripts/vda_feedback_server.py" \
    --host "${HOST}" \
    --port "${port}" \
    --model-path "${VDA_MODEL_PATH}" \
    "${ADAPTER_ARGS[@]}" \
    "${VDA_LOAD_ARGS[@]}" \
    --seed "$((SEED + index))" \
    --max-turns "${AGZ_VDA_FEEDBACK_MAX_TURNS:-5}" \
    --max-input-tokens "${AGZ_VDA_FEEDBACK_MAX_INPUT_TOKENS:-2048}" \
    --max-new-tokens "${AGZ_VDA_FEEDBACK_MAX_NEW_TOKENS:-384}" \
    --continuation-prompt-mode "${AGZ_VDA_FEEDBACK_CONTINUATION_PROMPT_MODE:-snapshot}" \
    --history-window "${AGZ_VDA_FEEDBACK_HISTORY_WINDOW:-6}" \
    --invalid-action-patience "${AGZ_VDA_FEEDBACK_INVALID_ACTION_PATIENCE:-0}" \
    --attn-implementation "${VDA_FEEDBACK_ATTN_IMPLEMENTATION}" \
    --top-p "${AGZ_VDA_FEEDBACK_TOP_P:-1.0}" \
    --top-k "${AGZ_VDA_FEEDBACK_TOP_K:-0}" \
    > "${ROOT}/logs/${RUN_NAME}_vda_feedback_${index}.log" 2>&1 &
  VDA_PIDS+=("$!")
  VDA_URLS+=("${url}")
done

cleanup() {
  for pid in "${VDA_PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${VDA_PIDS[@]}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

wait_for_service() {
  local url=$1
  local pid=$2
  for attempt in $(seq 1 600); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "VDA feedback service exited before becoming healthy: ${url}" >&2
      return 1
    fi
    if python -s - "${url}/health" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=1) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for VDA feedback service: ${url}" >&2
  return 1
}

for index in "${!VDA_PIDS[@]}"; do
  wait_for_service "${VDA_URLS[$index]}" "${VDA_PIDS[$index]}"
done
export AGZ_VDA_FEEDBACK_URLS=$(IFS=,; echo "${VDA_URLS[*]}")
export AGZ_DCA_FEEDBACK_LOG
export AGZ_VDA_FEEDBACK_TIMEOUT=${AGZ_VDA_FEEDBACK_TIMEOUT:-1800}

echo "DCA GPU layout=${DCA_GPU_LAYOUT} GPUs=${DCA_GPUS}; VDA feedback GPUs=${DCA_GPUS}"
echo "DCA feedback services=${AGZ_VDA_FEEDBACK_URLS}"
echo "DCA feedback attention=${VDA_FEEDBACK_ATTN_IMPLEMENTATION}"
echo "DCA parent VDA adapter=${VDA_ADAPTER_PATH:-base}"
echo "DCA feedback history_window=${AGZ_VDA_FEEDBACK_HISTORY_WINDOW:-6}"
echo "DCA reward fsync_every_batches=${AGZ_DCA_REWARD_FSYNC_EVERY_BATCHES:-1}"
echo "DCA gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING}"
echo "DCA rollout backend=${DCA_ROLLOUT_BACKEND} max_num_seqs=${DCA_MAX_NUM_SEQS} gpu_utilization=${DCA_GPU_MEMORY_UTILIZATION}"
echo "DCA resume_mode=${RESUME_MODE} resume_from_path=${RESUME_FROM_PATH} target_steps=${MAX_STEPS}"

export CUDA_VISIBLE_DEVICES="${DCA_GPUS}"
export VERL_RUN_ID="${RUN_NAME}"
export AGZ_TRITON_CACHE_NAMESPACE="verl_$(basename "${MODEL_PATH}")_dca"
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_memory_usage_threshold=${AGZ_RAY_MEMORY_USAGE_THRESHOLD:-0.999}
export RAY_memory_monitor_refresh_ms=${AGZ_RAY_MEMORY_MONITOR_REFRESH_MS:-0}

cd "${ROOT}"
PYTHONUNBUFFERED=1 python -s -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=problem \
    data.train_batch_size="${BATCH_SIZE}" \
    data.val_batch_size="${BATCH_SIZE}" \
    data.gen_batch_size="${BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.truncation=right \
    data.seed="${SEED}" \
    ray_init.num_cpus="${RAY_NUM_CPUS}" \
    reward_model.reward_manager=batch \
    reward_model.launch_reward_fn_async=False \
    custom_reward_function.path="${ROOT}/curriculum/reward_function/dca_online_reward.py" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}" \
    actor_rollout_ref.model.enable_activation_offload="${ENABLE_ACTIVATION_OFFLOAD}" \
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
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${DCA_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${DCA_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward="${RESHARD_AFTER_FORWARD}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    +actor_rollout_ref.actor.fsdp_config.cpu_offload=false \
    +actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.agent.enable_agent=False \
    actor_rollout_ref.agent.tool_server_url=null \
    actor_rollout_ref.agent.max_turns=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.name="${DCA_ROLLOUT_BACKEND}" \
    +actor_rollout_ref.rollout.use_full_replica="${HF_FULL_ROLLOUT_REPLICA}" \
    +actor_rollout_ref.rollout.stop_on_complete_json="${STOP_ON_COMPLETE_JSON}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.max_num_seqs="${DCA_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${DCA_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.param_offload="${DCA_REF_OFFLOAD}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
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
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${DCA_N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.test_freq=0 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${MAX_STEPS}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.resume_from_path="${RESUME_FROM_PATH}" \
    2>&1 | tee "${ROOT}/logs/${RUN_NAME}.log"
