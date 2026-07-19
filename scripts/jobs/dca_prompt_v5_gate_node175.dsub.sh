#!/bin/bash
#DSUB -n AGZ_DCA_V5_GATE
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-175
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

OUT="${ROOT}/outputs/tmcd_v2/gates/dca-prompt-v5-4b"
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite existing gate: ${OUT}" >&2
  exit 73
fi
mkdir -p "${OUT}" "${ROOT}/logs/gates"

MANIFEST="${ROOT}/checkpoints/tmcd_v2/qwen3.5-4b/dca/round_1/manifest.json"
pids=()
for shard in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    export TRITON_CACHE_DIR="${AGZ_TRITON_CACHE_ROOT}/dca_prompt_v5_gate/shard_${shard}"
    mkdir -p "${TRITON_CACHE_DIR}"
    python -s "${ROOT}/scripts/generate_dca_scenarios.py" \
      --checkpoint-manifest "${MANIFEST}" \
      --output "${OUT}/candidates.shard_${shard}.json" \
      --num-candidates 256 \
      --batch-size 64 \
      --num-shards 4 \
      --shard-index "${shard}" \
      --seed 20261709 \
      --max-input-tokens 2048 \
      --max-new-tokens 1024 \
      --max-attempts 3 \
      --attn-implementation sdpa \
      --partial-fsync-every-batches 1 \
      --experiment-variant full \
      > "${OUT}/shard_${shard}.log" 2>&1
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

python -s "${ROOT}/scripts/merge_dca_candidate_shards.py" \
  --shards \
  "${OUT}/candidates.shard_0.json" \
  "${OUT}/candidates.shard_1.json" \
  "${OUT}/candidates.shard_2.json" \
  "${OUT}/candidates.shard_3.json" \
  --expected-count 256 \
  --output "${OUT}/candidates.json"

python -s - "${OUT}/candidates.json" "${OUT}/report.json" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
pool = json.loads(source.read_text(encoding="utf-8"))
valid = Counter()
attempts = Counter()
for record in pool["candidates"]:
    task_id = str(record.get("task_focus", "unknown")).split()[0]
    valid[task_id] += int(bool((record.get("checks", {}) or {}).get("all_ok")))
    attempts[task_id] += len(record.get("generation_attempts", []))
report = {
    "candidate_count": len(pool["candidates"]),
    "generation_prompt_version": pool.get("generation_prompt_version"),
    "max_attempts": pool.get("max_attempts"),
    "valid_by_task": dict(valid),
    "attempts_by_task": dict(attempts),
}
if pool.get("generation_prompt_version") != 5 or pool.get("max_attempts") != 3:
    raise SystemExit(f"generation signature mismatch: {report}")
short = {task: valid.get(task, 0) for task in ("T1", "T2", "T3", "T4") if valid.get(task, 0) < 32}
if short:
    raise SystemExit(f"task-validity gate failed: {short}; report={report}")
target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
PY

touch "${OUT}/SUCCEEDED"
