#!/bin/bash
#DSUB -n AGZQwen35Smoke
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=4;mem=32000"
#DSUB -ex job
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
mkdir -p "${ROOT}/logs"
export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"

python -s "${ROOT}/scripts/qwen35_text_smoke.py" \
  --model "${AGZ_QWEN35_4B_PATH}" \
  --model "${AGZ_QWEN35_9B_PATH}"
