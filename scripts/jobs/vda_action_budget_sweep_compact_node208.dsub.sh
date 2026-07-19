#!/bin/bash
#DSUB -n AGZ4BVDACompactAction
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-208
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

OUT="${ROOT}/outputs/optimization_gates/vda-action-budget-sweep-compact-v2"
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite existing sweep: ${OUT}" >&2
  exit 73
fi
mkdir -p "${OUT}"
SOURCE="/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/data/tmcd_v2/qwen3.5-4b/round_1/dca_feedback/feedback.jsonl"

pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${shard}" \
    TRITON_CACHE_DIR="/tmp/agentguard_zero_vda_budget_${USER}_${shard}" \
    python -s "${ROOT}/scripts/benchmark_vda_action_budget.py" \
      --model-path "${AGZ_QWEN35_4B_PATH}" \
      --scenario-jsonl "${SOURCE}" \
      --output "${OUT}/shard_${shard}.json" \
      --num-shards 4 \
      --shard-index "${shard}" \
      --scenarios-per-shard 4 \
      --budgets 320,384 \
      --max-input-tokens 4096 \
      --attn-implementation sdpa \
      > "${OUT}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python - "${OUT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = [json.loads(path.read_text()) for path in sorted(root.glob("shard_*.json"))]
summary = {}
for row in rows:
    for result in row["results"]:
        budget = str(result["budget"])
        value = summary.setdefault(budget, {"scenarios": 0, "parse_ok": 0, "max_elapsed_s": 0.0})
        value["scenarios"] += result["scenario_count"]
        value["parse_ok"] += result["parse_ok"]
        value["max_elapsed_s"] = max(value["max_elapsed_s"], result["elapsed_s"])
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
