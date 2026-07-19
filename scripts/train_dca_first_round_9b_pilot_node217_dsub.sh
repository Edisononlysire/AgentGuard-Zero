#!/bin/bash
#DSUB -n AGZDCAFirst9BPilot
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
export AGZ_PILOT=1
export AGZ_SOURCE_ROUND=${AGZ_SOURCE_ROUND:-0}
AGZ_PYTHON=${AGZ_PYTHON:-/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/agent0-gpu/bin/python}
/usr/bin/bash /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/run_dca_first_round_node217.sh
"${AGZ_PYTHON}" -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/audit_dca_first_lineage.py \
  --root /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero \
  --backbone qwen3.5-9b --artifact-scope pilot --max-round 1
