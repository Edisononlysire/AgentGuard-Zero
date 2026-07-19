#!/bin/bash
#DSUB -n AGZTMCDCPU
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=4;mem=32000"
#DSUB -ex job
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
mkdir -p "${ROOT}/logs" "${ROOT}/outputs/tmcd_eval"
export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"

SYSTEM=${AGZ_SYSTEM:-rule_based_soc}
DATA=${AGZ_EVAL_DATA:-${ROOT}/data/level1/level1_seed20260706_n500_frontier_vda.parquet}
LIMIT=${AGZ_EVAL_LIMIT:-256}
OUT=${AGZ_OUTPUT_DIR:-${ROOT}/outputs/tmcd_eval}
RUN_NAME=${AGZ_RUN_NAME:-tmcd_${SYSTEM}_${BATCH_JOB_ID:-manual}}

cd "${ROOT}"
python -s scripts/eval_tmcd_systems.py \
  --data "${DATA}" \
  --system "${SYSTEM}" \
  --model_backend mock \
  --limit "${LIMIT}" \
  --output_dir "${OUT}" \
  --run_name "${RUN_NAME}"
