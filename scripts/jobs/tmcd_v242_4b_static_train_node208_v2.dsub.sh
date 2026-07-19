#!/bin/bash
#DSUB -n AGZV242_4B_STV2
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-208
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_progressive/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_progressive/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-208
SCOPE=tmcd_v242
BACKBONE=qwen3.5-4b
SOURCE_HASHES=${ROOT}/outputs/source_snapshots/20260718T1300_progressive_variant_fix/deployed_source.sha256

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

mkdir -p "${ROOT}/logs/tmcd_v242_progressive" "${ROOT}/outputs/${SCOPE}/progressive/static_train"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

sha256sum -c "${SOURCE_HASHES}"
printf '%s  %s\n' ff3ad75126f115b11c635e9c4d6583e47e36012a855d817406069b39abb6c8de \
  "${ROOT}/data/${SCOPE}/${BACKBONE}/round_1/vda_train/train.parquet" | sha256sum -c -
printf '%s  %s\n' 2bc24b2c15078a5e03397523e98496dc757438e3529f7b6883deebfb286fe6c2 \
  "${ROOT}/data/${SCOPE}/${BACKBONE}/round_2/vda_train/train.parquet" | sha256sum -c -
printf '%s  %s\n' 82ee301c7b54bb28172e15399a3ff622ad3b8ca39c44c954724ec3342a32eed6 \
  "${ROOT}/data/${SCOPE}/${BACKBONE}/round_3/vda_train/train.parquet" | sha256sum -c -

python -s "${ROOT}/scripts/run_progressive_ablation.py" \
  --root "${ROOT}" \
  --formal-scope "${SCOPE}" \
  --backbone "${BACKBONE}" \
  --variant static_train \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --allocated-gpus "${CUDA_VISIBLE_DEVICES}" \
  --seed 20260709

test -f "${ROOT}/checkpoints/${SCOPE}/ablations/progressive/static_train/${BACKBONE}/vda/round_3/manifest.json"
touch "${ROOT}/outputs/${SCOPE}/progressive/static_train/node208_4b_static_train_v2.SUCCEEDED"
