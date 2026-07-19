#!/bin/bash
#DSUB -n AGZV242_4B_V5CAL800
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-175
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_calibration/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_calibration/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-175
SCOPE=tmcd_v242
BACKBONE=qwen3.5-4b
DCA_MANIFEST=${ROOT}/checkpoints/${SCOPE}/${BACKBONE}/dca/round_3/manifest.json
TEST_MANIFEST=${ROOT}/data/final_heldout/${SCOPE}/${BACKBONE}/manifest.json
TEST_PARQUET=${ROOT}/data/final_heldout/${SCOPE}/${BACKBONE}/final_heldout.parquet
OUTPUT=${ROOT}/data/${SCOPE}/v5c_calibration/${BACKBONE}
SOURCE_HASHES=${ROOT}/outputs/source_snapshots/20260718T0640_v5c_calibration_800/deployed_source.sha256

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ -e "${OUTPUT}/manifest.json" ]]; then
  echo "Refusing to replace an existing sealed ECRG-Cal: ${OUTPUT}/manifest.json" >&2
  exit 73
fi

mkdir -p "${ROOT}/logs/tmcd_v242_calibration" "${ROOT}/outputs/${SCOPE}/v5c_calibration"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

sha256sum -c "${SOURCE_HASHES}"
printf '%s  %s\n' \
  469d3086631aec15750f55151995d69b08f78f918eaafee9e5c4d5dedadf32c3 \
  "${DCA_MANIFEST}" | sha256sum -c -
printf '%s  %s\n' \
  48dea06d55e88ad6e65e3f286a5c4e272b077f1890094289c285449952888d92 \
  "${TEST_MANIFEST}" | sha256sum -c -
printf '%s  %s\n' \
  f89e27a71475650f089531c93c4259109ae07ce453510ab6f1de06ba5b483212 \
  "${TEST_PARQUET}" | sha256sum -c -

export AGZ_DCA_GENERATION_ATTN_IMPLEMENTATION=sdpa
export AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES=16
export AGZ_TRITON_CACHE_ROOT=${AGZ_TRITON_CACHE_ROOT}/tmcd_v242_v5c_calibration_4b

python -s "${ROOT}/scripts/generate_v5c_calibration.py" \
  --root "${ROOT}" \
  --artifact-scope "${SCOPE}" \
  --backbone "${BACKBONE}" \
  --checkpoint-manifest "${DCA_MANIFEST}" \
  --candidate-count 1600 \
  --per-task 200 \
  --batch-size 72 \
  --max-input-tokens 2048 \
  --max-new-tokens 1024 \
  --generation-seed 21180709 \
  --selection-seed 21180710 \
  --topup-size 500 \
  --allocated-gpus "${CUDA_VISIBLE_DEVICES}"

python -s - "${OUTPUT}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

output = Path(sys.argv[1])
manifest = json.loads((output / "manifest.json").read_text())
audit = json.loads((output / "selection_audit.json").read_text())
frame = pd.read_parquet(output / "calibration.parquet")
counts = Counter(frame["task_id"].tolist())
expected = {"T1": 200, "T2": 200, "T3": 200, "T4": 200}
assert manifest["status"] == "sealed"
assert manifest["source_dca_round"] == 3
assert manifest["selected_count"] == 800
assert manifest["task_counts"] == expected
assert len(frame) == 800 and dict(counts) == expected
assert audit["selected_formal_or_test_overlap_count"] == 0
assert audit["selected_duplicate_count"] == 0
assert audit["model_performance_filtering"] is False
assert audit["frontier_score_filtering"] is False
print(json.dumps({"rows": len(frame), "task_counts": dict(counts), "status": "sealed"}, sort_keys=True))
PY

touch "${ROOT}/outputs/${SCOPE}/v5c_calibration/node175_4b_v5c_calibration_800.SUCCEEDED"
