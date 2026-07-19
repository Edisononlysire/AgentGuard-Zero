#!/bin/bash
set -euo pipefail

source /opt/batch/cli/envs/profile.env
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero

export AGZ_SELECT_LIMIT=16
export AGZ_SELECT_OFFSET=16
export AGZ_SELECT_K=4
export AGZ_SELECTOR_MODE=mitigation_v3
export AGZ_SELECT_MAX_NEW_TOKENS=768
export AGZ_SELECT_MAX_TURNS=3
export AGZ_SELECT_RUN_NAME=agentguard_api_select_v3_l16_holdout16_$(date +%Y%m%d_%H%M%S)

dsub -s scripts/eval_level1_api_select_dsub.sh
