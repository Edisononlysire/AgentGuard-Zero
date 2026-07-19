#!/bin/bash
#DSUB -n AGZV242_4B_ECRGCAL
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-175
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_ecrg_calibration/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v242_ecrg_calibration/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-175
SCOPE=tmcd_v242
BACKBONE=qwen3.5-4b
CAL_DIR=${ROOT}/data/${SCOPE}/v5c_calibration/${BACKBONE}
CAL_DATA=${CAL_DIR}/calibration.parquet
CAL_MANIFEST=${CAL_DIR}/manifest.json
VDA_MANIFEST=${ROOT}/checkpoints/${SCOPE}/${BACKBONE}/vda/round_3/manifest.json
DCA_MANIFEST=${ROOT}/checkpoints/${SCOPE}/${BACKBONE}/dca/round_3/manifest.json
OUTPUT=${ROOT}/outputs/${SCOPE}/ecrg_calibration/${BACKBONE}
SOURCE_HASHES=${ROOT}/outputs/source_snapshots/20260718T0710_ecrg_calibration/deployed_source.sha256

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ -e "${OUTPUT}/manifest.json" ]]; then
  echo "Refusing to replace a frozen ECRG configuration: ${OUTPUT}/manifest.json" >&2
  exit 73
fi

mkdir -p "${ROOT}/logs/tmcd_v242_ecrg_calibration" "${OUTPUT}/traces"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

sha256sum -c "${SOURCE_HASHES}"
printf '%s  %s\n' \
  992de81aa46072152cc87ded47395e2e19e2c07fa4ce168f13afa821f1cf527a \
  "${CAL_MANIFEST}" | sha256sum -c -
printf '%s  %s\n' \
  647c6a1f6f9da2320813d03f17d5ec73053acf46454d5391fe7b776f190a880c \
  "${CAL_DATA}" | sha256sum -c -
printf '%s  %s\n' \
  469d3086631aec15750f55151995d69b08f78f918eaafee9e5c4d5dedadf32c3 \
  "${DCA_MANIFEST}" | sha256sum -c -
printf '%s  %s\n' \
  b2bac07dc3c88e13e699ed02d21b032d930a6e48e6aa9ee4b57076d61f9365c5 \
  "${VDA_MANIFEST}" | sha256sum -c -

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Expected four allocated GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 74
fi

TASKS=(T1 T2 T3 T4)
PIDS=()
for INDEX in 0 1 2 3; do
  TASK=${TASKS[${INDEX}]}
  TASK_OUTPUT=${OUTPUT}/traces/${TASK}
  mkdir -p "${TASK_OUTPUT}" "${AGZ_TRITON_CACHE_ROOT}/ecrg_calibration/${TASK}"
  CUDA_VISIBLE_DEVICES=${GPU_IDS[${INDEX}]} \
  TRITON_CACHE_DIR=${AGZ_TRITON_CACHE_ROOT}/ecrg_calibration/${TASK} \
  python -s "${ROOT}/scripts/generate_ecrg_calibration_traces.py" \
    --root "${ROOT}" \
    --data "${CAL_DATA}" \
    --calibration-manifest "${CAL_MANIFEST}" \
    --vda-manifest "${VDA_MANIFEST}" \
    --output-dir "${TASK_OUTPUT}" \
    --task-id "${TASK}" \
    --expected-count 200 \
    --candidate-count 6 \
    --max-turns 16 \
    --trajectory-batch-size 4 \
    --label-workers 12 \
    --seed "$((20260718 + INDEX * 1000003))" \
    --model-backend hf \
    --max-input-tokens 2048 \
    --max-new-tokens 256 \
    --temperature 0.7 \
    --top-p 1.0 \
    --top-k 0 \
    --dtype bf16 \
    --attn-implementation sdpa \
    >"${OUTPUT}/trace_${TASK}.log" 2>&1 &
  PIDS+=("$!")
done

TRACE_STATUS=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || TRACE_STATUS=1
done
if [[ ${TRACE_STATUS} -ne 0 ]]; then
  echo "At least one ECRG trace shard failed" >&2
  tail -n 120 "${OUTPUT}"/trace_T*.log >&2 || true
  exit 75
fi

python -s "${ROOT}/scripts/fit_ecrg_calibration.py" \
  --trace-manifests \
    "${OUTPUT}/traces/T1/manifest.json" \
    "${OUTPUT}/traces/T2/manifest.json" \
    "${OUTPUT}/traces/T3/manifest.json" \
    "${OUTPUT}/traces/T4/manifest.json" \
  --calibration-manifest "${CAL_MANIFEST}" \
  --vda-manifest "${VDA_MANIFEST}" \
  --dca-manifest "${DCA_MANIFEST}" \
  --output-dir "${OUTPUT}" \
  --split-seed 20260718 \
  --fit-per-task 160 \
  --select-per-task 40 \
  --fit-finalists 12

python -s - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
config = json.loads((root / "ecrg_config.json").read_text())
assert manifest["status"] == "frozen"
assert manifest["candidate_count"] == 6
assert manifest["calibration_scenario_count"] == 800
assert manifest["task_counts"] == {"T1": 200, "T2": 200, "T3": 200, "T4": 200}
assert manifest["fit_scenario_count"] == 640
assert manifest["selection_scenario_count"] == 160
assert manifest["parameter_training"] is False
assert manifest["vda_adapter_sha256_before"] == manifest["vda_adapter_sha256_after"]
assert manifest["dca_adapter_sha256_before"] == manifest["dca_adapter_sha256_after"]
assert manifest["tmcd_test_used"] is False
assert config["status"] == "frozen"
assert config["hidden_state_access"] is False
assert config["vda_parameter_update"] is False
assert config["dca_parameter_update"] is False
print(json.dumps({"status": "frozen", "candidate_count": 6, "calibration": 800, "winner": manifest["winner_profiles"]}, sort_keys=True))
PY

touch "${ROOT}/outputs/${SCOPE}/ecrg_calibration/node175_4b_ecrg_calibration.SUCCEEDED"
