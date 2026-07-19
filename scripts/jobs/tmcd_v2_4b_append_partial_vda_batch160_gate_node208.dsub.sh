#!/bin/bash
#DSUB -n AGZV2_4B_APP_PART
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
DATA=${ROOT}/outputs/tmcd_v2/gates/vda-append4-partial-data

export AGZ_GATE_EXPECTED_NODE=cyclone001-agent-208
export AGZ_GATE_OUTPUT_DIR="${ROOT}/outputs/tmcd_v2/gates/vda-append4-partial-b160-node208"
export AGZ_GATE_TRAIN_FILE="${DATA}/train.parquet"
export AGZ_GATE_VAL_FILE="${DATA}/train.parquet"
export AGZ_GATE_POOL_MANIFEST="${DATA}/manifest.json"
export AGZ_GATE_RUN_NAME=agz_gate_qwen3.5-4b_append_partial_vda_batch160_node208

exec bash "${ROOT}/scripts/jobs/tmcd_v2_4b_append_vda_batch160_gate_node208.dsub.sh"
