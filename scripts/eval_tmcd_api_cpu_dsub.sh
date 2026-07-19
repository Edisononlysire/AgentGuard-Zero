#!/bin/bash
#DSUB -n AGZTMCDAPI
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=4;mem=16000"
#DSUB -ex job
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
mkdir -p "${ROOT}/logs" "${ROOT}/outputs/tmcd_api_eval"
export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/agentguard_env.sh"

if [[ -n "${AGZ_API_ENV_FILE:-}" && -f "${AGZ_API_ENV_FILE}" ]]; then
  set -a
  source "${AGZ_API_ENV_FILE}"
  set +a
elif [[ -f "${HOME}/.agentguard_api_env" ]]; then
  set -a
  source "${HOME}/.agentguard_api_env"
  set +a
fi

SYSTEM=${AGZ_SYSTEM:-agentguard_zero_select}
DATA=${AGZ_EVAL_DATA:-${ROOT}/data/level1/level1_seed20260706_n500_frontier_vda.parquet}
LIMIT=${AGZ_EVAL_LIMIT:-16}
OFFSET=${AGZ_EVAL_OFFSET:-0}
OUT=${AGZ_OUTPUT_DIR:-${ROOT}/outputs/tmcd_api_eval}
RUN_NAME=${AGZ_RUN_NAME:-tmcd_api_${SYSTEM}_${AGZ_API_MODEL:-${LLM_MODEL:-model}}_${BATCH_JOB_ID:-manual}}

API_EXTRA_ARGS=()
if [[ "${AGZ_API_RESPONSE_FORMAT_JSON:-0}" == "1" || "${AGZ_API_RESPONSE_FORMAT_JSON:-0}" == "true" ]]; then
  API_EXTRA_ARGS+=(--api_response_format_json)
fi
if [[ "${AGZ_API_DISABLE_THINKING:-0}" == "1" || "${AGZ_API_DISABLE_THINKING:-0}" == "true" ]]; then
  API_EXTRA_ARGS+=(--api_disable_thinking)
fi
if [[ "${AGZ_API_MULTI_CHOICE:-0}" == "1" || "${AGZ_API_MULTI_CHOICE:-0}" == "true" ]]; then
  API_EXTRA_ARGS+=(--api_multi_choice)
fi

cd "${ROOT}"
python -s scripts/eval_tmcd_systems.py \
  --data "${DATA}" \
  --system "${SYSTEM}" \
  --model_backend api \
  --api_model "${AGZ_API_MODEL:-${LLM_MODEL:-}}" \
  --api_base_url "${AGZ_API_BASE_URL:-${LLM_BASE_URL:-}}" \
  --api_key_env "${AGZ_API_KEY_ENV:-}" \
  --candidate_count "${AGZ_CANDIDATE_COUNT:-4}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --output_dir "${OUT}" \
  --run_name "${RUN_NAME}" \
  --selector_mode "${AGZ_SELECTOR_MODE:-v5_c_frontier_minimax}" \
  --max_turns "${AGZ_AGENT_MAX_TURNS:-5}" \
  --max_new_tokens "${AGZ_MAX_NEW_TOKENS:-256}" \
  --temperature "${AGZ_TEMPERATURE:-0.7}" \
  --top_p "${AGZ_TOP_P:-0.9}" \
  --api_timeout "${AGZ_API_TIMEOUT:-90}" \
  --api_retries "${AGZ_API_RETRIES:-2}" \
  "${API_EXTRA_ARGS[@]}"
