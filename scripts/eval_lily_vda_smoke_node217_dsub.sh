#!/bin/bash
#DSUB -n AGZLilyVDASmoke
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=12;gpu=1;mem=80000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/cyber_llm_env.sh"

export AGZ_SYSTEM=lily_cybersecurity_vda
export AGZ_MODEL_BACKEND=hf
export AGZ_MODEL_PATH="${AGZ_LILY_CYBER_MODEL_PATH}"
export AGZ_EVAL_LIMIT=${AGZ_EVAL_LIMIT:-1}
export AGZ_EVAL_OFFSET=${AGZ_EVAL_OFFSET:-0}
export AGZ_CANDIDATE_COUNT=${AGZ_CANDIDATE_COUNT:-1}
export AGZ_AGENT_MAX_TURNS=${AGZ_AGENT_MAX_TURNS:-2}
export AGZ_MAX_INPUT_TOKENS=${AGZ_MAX_INPUT_TOKENS:-4096}
export AGZ_MAX_NEW_TOKENS=${AGZ_MAX_NEW_TOKENS:-1024}
export AGZ_DTYPE=${AGZ_DTYPE:-bf16}
export AGZ_ATTN_IMPLEMENTATION=${AGZ_ATTN_IMPLEMENTATION:-sdpa}
export AGZ_OUTPUT_DIR=${AGZ_OUTPUT_DIR:-${ROOT}/outputs/tmcd_lily_smoke}
export AGZ_RUN_NAME=${AGZ_RUN_NAME:-tmcd_lily_cybersecurity_vda_functional_smoke_${BATCH_JOB_ID:-manual}}

exec /usr/bin/bash "${ROOT}/scripts/eval_tmcd_system_dsub.sh"
