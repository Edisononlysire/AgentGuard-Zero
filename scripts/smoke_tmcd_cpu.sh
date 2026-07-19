#!/bin/bash
set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"

DATA=${AGZ_LEVEL1_FRONTIER_FILE:-${ROOT}/data/level1/level1_seed20260706_n500_frontier_vda.parquet}
OUT=${AGZ_OUTPUT_DIR:-${ROOT}/outputs/tmcd_eval_smoke}
LIMIT=${AGZ_EVAL_LIMIT:-4}

cd "${ROOT}"
python -s scripts/level1_rollout_server.py --self-test
for SYSTEM in rule_based_soc oracle_defender random_policy react_base_tools memory_agent trust_score_agent qwen_zero_shot_vda agentguard_zero_select; do
  BACKEND=mock
  if [[ "${SYSTEM}" == "rule_based_soc" || "${SYSTEM}" == "oracle_defender" || "${SYSTEM}" == "random_policy" ]]; then
    BACKEND=mock
  fi
  python -s scripts/eval_tmcd_systems.py \
    --data "${DATA}" \
    --system "${SYSTEM}" \
    --model_backend "${BACKEND}" \
    --limit "${LIMIT}" \
    --output_dir "${OUT}" \
    --run_name "smoke_${SYSTEM}"
done
python -s scripts/export_tmcd_tables.py \
  --input_dir "${OUT}" \
  --output_dir "${ROOT}/outputs/paper_tables_smoke"
