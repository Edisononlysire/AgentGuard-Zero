#!/bin/bash
#DSUB -n AGZSelectEval
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=8;gpu=1;mem=80000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
mkdir -p "${ROOT}/logs" "${ROOT}/outputs/eval_select"

source "${ROOT}/scripts/agentguard_env.sh"

export AGZ_MODEL_PATH="${AGZ_MODEL_PATH:-/home/share/huadjyin/home/s_qinhua2/02code/guozhihan/InfectModel/model/Qwen/Qwen3-8B}"
export AGZ_LEVEL1_FRONTIER_FILE="${AGZ_LEVEL1_FRONTIER_FILE:-${ROOT}/data/level1/level1_seed20260706_n500_frontier_vda.parquet}"
export AGZ_SELECT_POLICY="${AGZ_SELECT_POLICY:-agentguard_zero_select}"
export AGZ_SELECT_LIMIT="${AGZ_SELECT_LIMIT:-16}"
export AGZ_SELECT_K="${AGZ_SELECT_K:-4}"
export AGZ_SELECT_MAX_TURNS="${AGZ_SELECT_MAX_TURNS:-5}"
export AGZ_SELECT_MAX_INPUT_TOKENS="${AGZ_SELECT_MAX_INPUT_TOKENS:-4096}"
export AGZ_SELECT_MAX_NEW_TOKENS="${AGZ_SELECT_MAX_NEW_TOKENS:-256}"
export AGZ_SELECT_TEMPERATURE="${AGZ_SELECT_TEMPERATURE:-0.7}"
export AGZ_SELECT_TOP_P="${AGZ_SELECT_TOP_P:-0.9}"
export AGZ_SELECT_BACKEND="${AGZ_SELECT_BACKEND:-hf}"
export AGZ_SELECT_RUN_NAME="${AGZ_SELECT_RUN_NAME:-agentguard_select_${BATCH_JOB_ID:-manual}}"
export AGZ_SELECT_OUTPUT_DIR="${AGZ_SELECT_OUTPUT_DIR:-${ROOT}/outputs/eval_select}"

cd "${ROOT}"

python -s "${ROOT}/scripts/eval_level1_select.py" \
  --data "${AGZ_LEVEL1_FRONTIER_FILE}" \
  --model_path "${AGZ_MODEL_PATH}" \
  --adapter_path "${AGZ_ADAPTER_PATH:-}" \
  --policy "${AGZ_SELECT_POLICY}" \
  --model_backend "${AGZ_SELECT_BACKEND}" \
  --candidate_count "${AGZ_SELECT_K}" \
  --limit "${AGZ_SELECT_LIMIT}" \
  --max_turns "${AGZ_SELECT_MAX_TURNS}" \
  --max_input_tokens "${AGZ_SELECT_MAX_INPUT_TOKENS}" \
  --max_new_tokens "${AGZ_SELECT_MAX_NEW_TOKENS}" \
  --temperature "${AGZ_SELECT_TEMPERATURE}" \
  --top_p "${AGZ_SELECT_TOP_P}" \
  --run_name "${AGZ_SELECT_RUN_NAME}" \
  --output_dir "${AGZ_SELECT_OUTPUT_DIR}"
