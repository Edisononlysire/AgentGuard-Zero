#!/bin/bash
#DSUB -n AGZAPISelect
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=4;mem=16000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
mkdir -p "${ROOT}/logs" "${ROOT}/outputs/eval_api_select"

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

export AGZ_LEVEL1_FRONTIER_FILE="${AGZ_LEVEL1_FRONTIER_FILE:-${ROOT}/data/level1/level1_seed20260706_n500_frontier_vda.parquet}"
export AGZ_SELECT_POLICY="${AGZ_SELECT_POLICY:-agentguard_zero_select}"
export AGZ_SELECT_LIMIT="${AGZ_SELECT_LIMIT:-4}"
export AGZ_SELECT_OFFSET="${AGZ_SELECT_OFFSET:-0}"
export AGZ_SELECT_K="${AGZ_SELECT_K:-4}"
export AGZ_SELECT_MAX_TURNS="${AGZ_SELECT_MAX_TURNS:-3}"
export AGZ_SELECT_MAX_NEW_TOKENS="${AGZ_SELECT_MAX_NEW_TOKENS:-768}"
export AGZ_SELECT_TEMPERATURE="${AGZ_SELECT_TEMPERATURE:-0.7}"
export AGZ_SELECT_TOP_P="${AGZ_SELECT_TOP_P:-0.9}"
export AGZ_SELECTOR_MODE="${AGZ_SELECTOR_MODE:-mitigation_v4}"
export AGZ_SELECT_RUN_NAME="${AGZ_SELECT_RUN_NAME:-agentguard_api_select_${BATCH_JOB_ID:-manual}}"
export AGZ_SELECT_OUTPUT_DIR="${AGZ_SELECT_OUTPUT_DIR:-${ROOT}/outputs/eval_api_select}"
export AGZ_API_TIMEOUT="${AGZ_API_TIMEOUT:-90}"
export AGZ_API_RETRIES="${AGZ_API_RETRIES:-2}"
export AGZ_API_RESPONSE_FORMAT_JSON="${AGZ_API_RESPONSE_FORMAT_JSON:-1}"
export AGZ_API_DISABLE_THINKING="${AGZ_API_DISABLE_THINKING:-1}"
export AGZ_API_MULTI_CHOICE="${AGZ_API_MULTI_CHOICE:-0}"

cd "${ROOT}"

API_EXTRA_ARGS=()
if [[ "${AGZ_API_RESPONSE_FORMAT_JSON}" == "1" || "${AGZ_API_RESPONSE_FORMAT_JSON}" == "true" ]]; then
  API_EXTRA_ARGS+=(--api_response_format_json)
fi
if [[ "${AGZ_API_DISABLE_THINKING}" == "1" || "${AGZ_API_DISABLE_THINKING}" == "true" ]]; then
  API_EXTRA_ARGS+=(--api_disable_thinking)
fi
if [[ "${AGZ_API_MULTI_CHOICE}" == "1" || "${AGZ_API_MULTI_CHOICE}" == "true" ]]; then
  API_EXTRA_ARGS+=(--api_multi_choice)
fi

python -s "${ROOT}/scripts/eval_level1_select.py" \
  --data "${AGZ_LEVEL1_FRONTIER_FILE}" \
  --policy "${AGZ_SELECT_POLICY}" \
  --model_backend api \
  --api_model "${AGZ_API_MODEL:-${LLM_MODEL:-}}" \
  --api_base_url "${AGZ_API_BASE_URL:-${LLM_BASE_URL:-}}" \
  --api_key_env "${AGZ_API_KEY_ENV:-}" \
  --candidate_count "${AGZ_SELECT_K}" \
  --limit "${AGZ_SELECT_LIMIT}" \
  --offset "${AGZ_SELECT_OFFSET}" \
  --selector_mode "${AGZ_SELECTOR_MODE}" \
  --max_turns "${AGZ_SELECT_MAX_TURNS}" \
  --max_new_tokens "${AGZ_SELECT_MAX_NEW_TOKENS}" \
  --temperature "${AGZ_SELECT_TEMPERATURE}" \
  --top_p "${AGZ_SELECT_TOP_P}" \
  --api_timeout "${AGZ_API_TIMEOUT}" \
  --api_retries "${AGZ_API_RETRIES}" \
  --run_name "${AGZ_SELECT_RUN_NAME}" \
  --output_dir "${AGZ_SELECT_OUTPUT_DIR}" \
  "${API_EXTRA_ARGS[@]}"
