#!/bin/bash
#DSUB -n AGZ4BOptimizedGate
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
if [[ "$(hostname)" != "cyclone001-agent-217" ]]; then
  echo "Refusing to run outside cyclone001-agent-217: $(hostname)" >&2
  exit 72
fi

export AGZ_ROOT="${ROOT}"
export AGZ_BACKBONE=qwen3.5-4b
export AGZ_PILOT=1
export AGZ_SOURCE_ROUND=0
export AGZ_DCA_FEEDBACK_CANDIDATES=64
export AGZ_DCA_ROLLOUT_N=2
export AGZ_DCA_BATCH_SIZE=32
export AGZ_DCA_STEPS=1
export AGZ_VDA_CANDIDATES=256
export AGZ_VDA_TRAIN_SIZE=32
export AGZ_VDA_DEV_SIZE=4
export AGZ_VDA_XPLAY_SIZE=4
export AGZ_VDA_BATCH_SIZE=32
export AGZ_VDA_STEPS=1
export AGZ_VDA_MAX_TURNS=4
export AGZ_VDA_FEEDBACK_MAX_TURNS=4
export AGZ_CANDIDATE_BATCH_SIZE=16
export AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU=4
export AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
export AGZ_SAVE_FREQ=1

/usr/bin/bash "${ROOT}/scripts/run_dca_first_round_node217.sh"
/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/agent0-gpu/bin/python -s \
  "${ROOT}/scripts/audit_dca_first_lineage.py" \
  --root "${ROOT}" --backbone qwen3.5-4b --artifact-scope pilot --max-round 1
