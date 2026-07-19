#!/bin/bash
#DSUB -n AGZAPIZero16
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=4;mem=16000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero

export AGZ_SELECT_POLICY="zero_shot_vda"
export AGZ_SELECT_LIMIT="${AGZ_SELECT_LIMIT:-16}"
export AGZ_SELECT_OFFSET="${AGZ_SELECT_OFFSET:-0}"
export AGZ_SELECT_K="1"
export AGZ_SELECT_MAX_TURNS="${AGZ_SELECT_MAX_TURNS:-3}"
export AGZ_SELECT_MAX_NEW_TOKENS="${AGZ_SELECT_MAX_NEW_TOKENS:-768}"
export AGZ_API_RESPONSE_FORMAT_JSON="${AGZ_API_RESPONSE_FORMAT_JSON:-1}"
export AGZ_API_DISABLE_THINKING="${AGZ_API_DISABLE_THINKING:-1}"
export AGZ_SELECT_RUN_NAME="${AGZ_SELECT_RUN_NAME:-agentguard_api_zeroshot_l16_${BATCH_JOB_ID:-manual}}"

/usr/bin/bash "${ROOT}/scripts/eval_level1_api_select_dsub.sh"
