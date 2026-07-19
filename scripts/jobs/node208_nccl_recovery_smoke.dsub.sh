#!/bin/bash
#DSUB -n AGZV24_NCCL208
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-208
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v24_parallel_pilot/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v24_parallel_pilot/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-208
NAME=node208_nccl_recovery_smoke
OUT=${ROOT}/outputs/tmcd_v24_parallel_pilot/${NAME}
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite recovery smoke output: ${OUT}" >&2
  exit 73
fi
mkdir -p "${OUT}" "${ROOT}/logs/tmcd_v24_parallel_pilot"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
test "${CONDA_DEFAULT_ENV:-}" = agent0-gpu
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [[ "${GPU_COUNT}" -ne 4 ]]; then
  echo "Expected four visible GPUs on ${EXPECTED_NODE}, found ${GPU_COUNT}" >&2
  exit 75
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free,ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader > "${OUT}/nvidia_smi_before.csv"
export AGZ_NCCL_SMOKE_REPORT=${OUT}/report.json
torchrun --standalone --nproc-per-node=4 \
  "${ROOT}/scripts/jobs/node208_nccl_recovery_smoke.py"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader > "${OUT}/nvidia_smi_after.csv"
touch "${OUT}/SMOKE_SUCCEEDED"
