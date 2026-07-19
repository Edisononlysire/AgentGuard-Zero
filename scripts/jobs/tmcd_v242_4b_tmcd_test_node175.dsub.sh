#!/bin/bash
#DSUB -n AGZV242_4B_TEST
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-175
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_test/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_test/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-175
SCOPE=tmcd_v242
BACKBONE=qwen3.5-4b
DCA_MANIFEST=${ROOT}/checkpoints/${SCOPE}/${BACKBONE}/dca/round_3/manifest.json
OUTPUT=${ROOT}/data/final_heldout/${SCOPE}/${BACKBONE}
SOURCE_HASHES=${ROOT}/outputs/source_snapshots/20260718T0430_plan2to1/deployed_source.sha256

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ -e "${OUTPUT}/manifest.json" ]]; then
  echo "Refusing to replace an existing sealed TMCD-Test: ${OUTPUT}/manifest.json" >&2
  exit 73
fi

mkdir -p "${ROOT}/logs/tmcd_v242_test" "${ROOT}/outputs/${SCOPE}/tmcd_test"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

sha256sum -c "${SOURCE_HASHES}"
printf '%s  %s\n' \
  469d3086631aec15750f55151995d69b08f78f918eaafee9e5c4d5dedadf32c3 \
  "${DCA_MANIFEST}" | sha256sum -c -

export AGZ_DCA_GENERATION_ATTN_IMPLEMENTATION=sdpa
export AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES=16
export AGZ_TRITON_CACHE_ROOT=${AGZ_TRITON_CACHE_ROOT}/tmcd_v242_test_4b

python -s "${ROOT}/scripts/generate_final_heldout.py" \
  --root "${ROOT}" \
  --artifact-scope "${SCOPE}" \
  --backbone "${BACKBONE}" \
  --checkpoint-manifest "${DCA_MANIFEST}" \
  --candidate-count 4800 \
  --per-task 600 \
  --batch-size 72 \
  --max-input-tokens 2048 \
  --max-new-tokens 1024 \
  --generation-seed 21160709 \
  --selection-seed 21160710 \
  --topup-size 500 \
  --allocated-gpus "${CUDA_VISIBLE_DEVICES}"

test -f "${OUTPUT}/manifest.json"
test -f "${OUTPUT}/final_heldout.parquet"
touch "${ROOT}/outputs/${SCOPE}/tmcd_test/node175_4b_tmcd_test.SUCCEEDED"
