#!/bin/bash
set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
source "${ROOT}/scripts/agentguard_env.sh"

export PYTHONPATH="${ROOT}:${ROOT}/executor_train:${ROOT}/executor_train/verl:${PYTHONPATH:-}"

NUM_CANDIDATES=${AGZ_LEVEL1_NUM_CANDIDATES:-500}
FRONTIER_SIZE=${AGZ_LEVEL1_FRONTIER_SIZE:-256}
SEED=${AGZ_LEVEL1_SEED:-20260706}
OUTPUT_DIR=${AGZ_LEVEL1_OUTPUT_DIR:-${ROOT}/data/level1}
PREFIX=${AGZ_LEVEL1_PREFIX:-level1_seed${SEED}_n${NUM_CANDIDATES}}

mkdir -p "${OUTPUT_DIR}"

echo "Building Level-1 frontier data"
echo "ROOT=${ROOT}"
echo "NUM_CANDIDATES=${NUM_CANDIDATES}"
echo "FRONTIER_SIZE=${FRONTIER_SIZE}"
echo "SEED=${SEED}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PREFIX=${PREFIX}"

python -s "${ROOT}/scripts/generate_level1_frontier.py" \
  --num-candidates "${NUM_CANDIDATES}" \
  --frontier-size "${FRONTIER_SIZE}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --prefix "${PREFIX}"
