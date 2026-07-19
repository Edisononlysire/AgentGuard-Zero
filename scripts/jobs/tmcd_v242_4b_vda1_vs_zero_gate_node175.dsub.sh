#!/bin/bash
#DSUB -n AGZV242_V1ZERO
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=48;gpu=4;mem=210000"
#DSUB -pn cyclone001-agent-175
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_vda1_vs_zero/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_vda1_vs_zero/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
PYTHON=/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/agent0-gpu/bin/python
MODEL=${ROOT}/models/qwen3_5/Qwen3.5-4B
ADAPTER=${ROOT}/checkpoints/tmcd_v242/qwen3.5-4b/vda/round_1/adapter
DATA=${ROOT}/outputs/tmcd_v242/diagnostics/vda1_vs_zero_20260718/data/r1_dev_balanced_80.parquet
OUTPUT=${ROOT}/outputs/tmcd_v242/diagnostics/vda1_vs_zero_20260718/job_${BATCH_JOB_ID}
EXPECTED_NODE=cyclone001-agent-175

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ ! -f "${DATA}" || ! -f "${ADAPTER}/adapter_model.safetensors" ]]; then
  echo "Missing diagnostic data or VDA1 adapter." >&2
  exit 73
fi

mkdir -p "${ROOT}/logs/tmcd_v242_vda1_vs_zero" "${OUTPUT}/logs"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Expected four allocated GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 74
fi

run_shard() {
  local gpu_id=$1
  local run_name=$2
  local shard_index=$3
  local adapter_path=$4
  local cache_dir=${AGZ_TRITON_CACHE_ROOT}/vda1_vs_zero_20260718/${BATCH_JOB_ID}/${run_name}/shard_${shard_index}
  mkdir -p "${cache_dir}"
  local adapter_args=()
  if [[ -n "${adapter_path}" ]]; then
    adapter_args=(--adapter_path "${adapter_path}")
  fi
  CUDA_VISIBLE_DEVICES=${gpu_id} \
  TRITON_CACHE_DIR=${cache_dir} \
  TORCHINDUCTOR_CACHE_DIR=${cache_dir}/torchinductor \
  "${PYTHON}" -s "${ROOT}/scripts/eval_tmcd_systems.py" \
    --data "${DATA}" \
    --system agentguard_zero_train \
    --model_backend hf \
    --model_path "${MODEL}" \
    "${adapter_args[@]}" \
    --limit 80 \
    --output_dir "${OUTPUT}" \
    --run_name "${run_name}" \
    --candidate_count 1 \
    --max_turns 16 \
    --trajectory_batch_size 4 \
    --max_input_tokens 2048 \
    --max_new_tokens 320 \
    --temperature 0.7 \
    --top_p 1.0 \
    --top_k 0 \
    --dtype bf16 \
    --attn_implementation sdpa \
    --seed 20260718 \
    --num_shards 2 \
    --shard_index "${shard_index}" \
    >"${OUTPUT}/logs/${run_name}_shard_${shard_index}.log" 2>&1
}

pids=()
run_shard "${GPU_IDS[0]}" zero_same_training_prompt 0 "" & pids+=("$!")
run_shard "${GPU_IDS[1]}" zero_same_training_prompt 1 "" & pids+=("$!")
run_shard "${GPU_IDS[2]}" vda1 0 "${ADAPTER}" & pids+=("$!")
run_shard "${GPU_IDS[3]}" vda1 1 "${ADAPTER}" & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ ${failed} -ne 0 ]]; then
  echo "One or more evaluation shards failed; inspect ${OUTPUT}/logs." >&2
  exit 2
fi

"${PYTHON}" -s "${ROOT}/scripts/merge_tmcd_eval_shards.py" \
  --run-dir "${OUTPUT}/zero_same_training_prompt" \
  --expected-count 80
"${PYTHON}" -s "${ROOT}/scripts/merge_tmcd_eval_shards.py" \
  --run-dir "${OUTPUT}/vda1" \
  --expected-count 80

sha256sum \
  "${DATA}" \
  "${ROOT}/outputs/tmcd_v242/diagnostics/vda1_vs_zero_20260718/data/manifest.json" \
  "${ROOT}/checkpoints/tmcd_v242/qwen3.5-4b/vda/round_1/manifest.json" \
  "${ADAPTER}/adapter_model.safetensors" \
  >"${OUTPUT}/input_hashes.sha256"
touch "${OUTPUT}/SUCCEEDED"
echo "Completed VDA1 versus same-prompt zero-shot diagnostic: ${OUTPUT}"
