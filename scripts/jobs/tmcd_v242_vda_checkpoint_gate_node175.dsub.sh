#!/bin/bash
#DSUB -n AGZV242_VDA_GATE
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=48;gpu=4;mem=210000"
#DSUB -pn cyclone001-agent-175
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_vda_gate/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_vda_gate/%J.err

set -u

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
PYTHON=/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/agent0-gpu/bin/python
DATA=${ROOT}/data/tmcd_v242/v5c_calibration/qwen3.5-4b/calibration.parquet
CAL_MANIFEST=${ROOT}/data/tmcd_v242/v5c_calibration/qwen3.5-4b/manifest.json
OUTPUT_ROOT=${ROOT}/outputs/tmcd_v242/vda_checkpoint_contract_gate/20260718_exact_prompt_320_v3
EXPECTED_NODE=cyclone001-agent-175

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

mkdir -p "${ROOT}/logs/tmcd_v242_vda_gate" "${OUTPUT_ROOT}"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

CACHE_ROOT=${AGZ_TRITON_CACHE_ROOT}/vda_checkpoint_gate_20260718_v3
mkdir -p "${CACHE_ROOT}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Expected four allocated GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 74
fi

pids=()
rounds=(1 2 3)
for index in 0 1 2; do
  round=${rounds[$index]}
  mkdir -p "${CACHE_ROOT}/round_${round}"
  CUDA_VISIBLE_DEVICES=${GPU_IDS[$index]} \
  TRITON_CACHE_DIR="${CACHE_ROOT}/round_${round}" \
  TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/round_${round}/torchinductor" \
  "${PYTHON}" -s \
    "${ROOT}/scripts/gate_vda_checkpoint_contract.py" \
    --data "${DATA}" \
    --calibration-manifest "${CAL_MANIFEST}" \
    --vda-manifest "${ROOT}/checkpoints/tmcd_v242/qwen3.5-4b/vda/round_${round}/manifest.json" \
    --round-index "${round}" \
    --output-dir "${OUTPUT_ROOT}/round_${round}" \
    --limit-per-task 4 \
    --candidate-count 6 \
    --batch-size 2 \
    --max-input-tokens 2048 \
    --max-new-tokens 320 \
    --temperature 0.7 \
    --top-p 1.0 \
    --top-k 0 \
    --dtype bf16 \
    --attn-implementation sdpa \
    --seed 20260718 \
    >"${OUTPUT_ROOT}/round_${round}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    failed=1
  fi
done

if [[ ${failed} -ne 0 ]]; then
  echo "One or more VDA checkpoint contract gates failed; inspect ${OUTPUT_ROOT}" >&2
  exit 2
fi

touch "${OUTPUT_ROOT}/ALL_CHECKPOINT_GATES_PASSED"
