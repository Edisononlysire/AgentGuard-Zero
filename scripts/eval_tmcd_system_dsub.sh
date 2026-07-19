#!/bin/bash
#DSUB -n AGZTMCD
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=12;gpu=1;mem=120000"
#DSUB -ex job
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
mkdir -p "${ROOT}/logs" "${ROOT}/outputs/tmcd_eval"
export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/cyber_llm_env.sh"

SYSTEM=${AGZ_SYSTEM:-qwen_zero_shot_vda}
DATA=${AGZ_EVAL_DATA:-${ROOT}/data/level1/level1_seed20260706_n500_frontier_vda.parquet}
LIMIT=${AGZ_EVAL_LIMIT:-64}
OFFSET=${AGZ_EVAL_OFFSET:-0}
OUT=${AGZ_OUTPUT_DIR:-${ROOT}/outputs/tmcd_eval}
RUN_NAME=${AGZ_RUN_NAME:-tmcd_${SYSTEM}_${BATCH_JOB_ID:-manual}}
BACKEND=${AGZ_MODEL_BACKEND:-hf}

case "${SYSTEM}" in
  rule_based_soc|oracle_defender|random_policy)
    BACKEND=mock
    ;;
  cyber_llm_vda)
    export AGZ_MODEL_PATH="${AGZ_MODEL_PATH:-${AGZ_CYBER_LLM_MODEL_PATH}}"
    ;;
  lily_cybersecurity_vda)
    export AGZ_MODEL_PATH="${AGZ_MODEL_PATH:-${AGZ_LILY_CYBER_MODEL_PATH}}"
    ;;
  *)
    export AGZ_MODEL_PATH="${AGZ_MODEL_PATH:-${AGZ_QWEN35_9B_PATH}}"
    ;;
esac

cd "${ROOT}"
python -s scripts/eval_tmcd_systems.py \
  --data "${DATA}" \
  --system "${SYSTEM}" \
  --model_backend "${BACKEND}" \
  --model_path "${AGZ_MODEL_PATH:-}" \
  --adapter_path "${AGZ_ADAPTER_PATH:-}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --output_dir "${OUT}" \
  --run_name "${RUN_NAME}" \
  --candidate_count "${AGZ_CANDIDATE_COUNT:-4}" \
  --max_turns "${AGZ_AGENT_MAX_TURNS:-5}" \
  --max_input_tokens "${AGZ_MAX_INPUT_TOKENS:-4096}" \
  --max_new_tokens "${AGZ_MAX_NEW_TOKENS:-256}" \
  --dtype "${AGZ_DTYPE:-bf16}" \
  --attn_implementation "${AGZ_ATTN_IMPLEMENTATION:-auto}"
