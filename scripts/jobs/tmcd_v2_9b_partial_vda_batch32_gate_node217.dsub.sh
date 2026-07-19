#!/bin/bash
#DSUB -n AGZV2_9B_VDA_PART
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
DATA=${ROOT}/outputs/tmcd_v2/gates/vda-full9-partial-data

export AGZ_GATE_EXPECTED_NODE=cyclone001-agent-217
export AGZ_GATE_OUTPUT_DIR="${ROOT}/outputs/tmcd_v2/gates/vda-full9-partial-b32-node217"
export AGZ_GATE_TRAIN_FILE="${DATA}/train.parquet"
export AGZ_GATE_VAL_FILE="${DATA}/train.parquet"
export AGZ_GATE_POOL_MANIFEST="${DATA}/manifest.json"
export AGZ_GATE_RUN_NAME=agz_gate_qwen3.5-9b_partial_vda_batch32_node217

exec bash "${ROOT}/scripts/jobs/tmcd_v2_9b_full_vda_batch32_gate_node217.dsub.sh"
