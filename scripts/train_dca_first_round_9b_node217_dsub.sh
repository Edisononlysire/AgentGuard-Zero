#!/bin/bash
#DSUB -n AGZDCAFirst9B
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail
if [[ "$(hostname)" != "${AGZ_EXPECTED_NODE:-cyclone001-agent-217}" ]]; then
  echo "Refusing to run outside cyclone001-agent-217: $(hostname)" >&2
  exit 72
fi
export AGZ_BACKBONE=qwen3.5-9b
export AGZ_PILOT=${AGZ_PILOT:-0}
export AGZ_START_ROUND=${AGZ_START_ROUND:-0}
export AGZ_END_ROUND=${AGZ_END_ROUND:-2}
exec /usr/bin/bash /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/run_dca_first_three_rounds_node217.sh
