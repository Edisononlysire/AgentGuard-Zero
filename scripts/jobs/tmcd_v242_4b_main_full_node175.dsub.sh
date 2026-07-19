#!/bin/bash
#DSUB -n AGZV242_4B_FULLK6
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-175
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_main_full/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_main_full/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-175
SCOPE=tmcd_v242
BACKBONE=qwen3.5-4b
DATA=${ROOT}/data/final_heldout/${SCOPE}/${BACKBONE}/final_heldout.parquet
DATA_MANIFEST=${ROOT}/data/final_heldout/${SCOPE}/${BACKBONE}/manifest.json
VDA_MANIFEST=${ROOT}/checkpoints/${SCOPE}/${BACKBONE}/vda/round_3/manifest.json
VDA_ADAPTER=${ROOT}/checkpoints/${SCOPE}/${BACKBONE}/vda/round_3/adapter
ECRG_DIR=${ROOT}/outputs/${SCOPE}/ecrg_calibration/${BACKBONE}
ECRG_CONFIG=${ECRG_DIR}/ecrg_config.json
ECRG_MANIFEST=${ECRG_DIR}/manifest.json
OUTPUT_ROOT=${ROOT}/outputs/${SCOPE}/main_results/${BACKBONE}
RUN_NAME=agentguard_zero_full_k6_seed20260718
RUN_DIR=${OUTPUT_ROOT}/${RUN_NAME}
FORMAL_MANIFEST=${RUN_DIR}/formal_manifest.json
SOURCE_HASHES=${ROOT}/outputs/source_snapshots/20260718T1358_tmcd_main_full/deployed_source.sha256
TEST_PY=/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/qwen3/bin/python

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ -e "${FORMAL_MANIFEST}" ]]; then
  echo "Refusing to replace sealed formal Full results: ${FORMAL_MANIFEST}" >&2
  exit 73
fi

mkdir -p "${ROOT}/logs/tmcd_v242_main_full" "${OUTPUT_ROOT}"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

sha256sum -c "${SOURCE_HASHES}"
printf '%s  %s\n' \
  f89e27a71475650f089531c93c4259109ae07ce453510ab6f1de06ba5b483212 \
  "${DATA}" | sha256sum -c -
printf '%s  %s\n' \
  48dea06d55e88ad6e65e3f286a5c4e272b077f1890094289c285449952888d92 \
  "${DATA_MANIFEST}" | sha256sum -c -
printf '%s  %s\n' \
  b2bac07dc3c88e13e699ed02d21b032d930a6e48e6aa9ee4b57076d61f9365c5 \
  "${VDA_MANIFEST}" | sha256sum -c -
printf '%s  %s\n' \
  db3bb1a34ad6f5910b64a3100d932e0cdb600a3275d302569cdbd4d208d05169 \
  "${ECRG_CONFIG}" | sha256sum -c -
printf '%s  %s\n' \
  47790f256886fd94c622b55ee433160920130f4e25a9fceb256078d07da923ac \
  "${ECRG_MANIFEST}" | sha256sum -c -

python -s "${ROOT}/scripts/preflight_tmcd_main_eval.py" \
  --backbone "${BACKBONE}" \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --expected-node agent-175 \
  --vda-manifest "${VDA_MANIFEST}" \
  --data-manifest "${DATA_MANIFEST}" \
  --ecrg-config "${ECRG_CONFIG}" \
  --source-hashes "${SOURCE_HASHES}" \
  --test-python "${TEST_PY}" \
  --output "${OUTPUT_ROOT}/preflight_node175.json"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Expected four allocated GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 74
fi

python -s "${ROOT}/scripts/run_tmcd_eval_four_gpu.py" \
  --data "${DATA}" \
  --system agentguard_zero_full \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --adapter-path "${VDA_ADAPTER}" \
  --ecrg-config "${ECRG_CONFIG}" \
  --model-backend hf \
  --candidate-count 6 \
  --limit 2400 \
  --split all \
  --max-turns 16 \
  --trajectory-batch-size 24 \
  --max-input-tokens 2048 \
  --max-new-tokens 320 \
  --temperature 0.7 \
  --top-p 1.0 \
  --top-k 0 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --seed 20260718

python -s "${ROOT}/scripts/validate_tmcd_main_full.py" \
  --run-dir "${RUN_DIR}" \
  --data "${DATA}" \
  --data-manifest "${DATA_MANIFEST}" \
  --vda-manifest "${VDA_MANIFEST}" \
  --ecrg-config "${ECRG_CONFIG}" \
  --ecrg-manifest "${ECRG_MANIFEST}" \
  --source-hashes "${SOURCE_HASHES}" \
  --output "${FORMAL_MANIFEST}"

touch "${RUN_DIR}/FORMAL_EVALUATION_SUCCEEDED"
