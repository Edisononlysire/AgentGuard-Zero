#!/bin/bash
#DSUB -n AGZ4BCandidateLongSweep
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

OUT="${ROOT}/outputs/optimization_gates/candidate-generation-long-context-sweep"
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite existing sweep: ${OUT}" >&2
  exit 73
fi
mkdir -p "${OUT}"
export AGZ_TRITON_CACHE_ROOT="/tmp/agentguard_zero_candidate_long_sweep_${USER}"

run_case() {
  local batch_size="$1"
  local output="${OUT}/sdpa_b${batch_size}_n1024.json"
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    "${ROOT}/scripts/benchmark_qwen35_fast_generation.py" \
    --model-path "${AGZ_QWEN35_4B_PATH}" \
    --output "${output}" \
    --batch-size "${batch_size}" \
    --prompt-tokens 1344 \
    --new-tokens 1024 \
    --attn-implementation sdpa
}

run_case 48
run_case 56

python - "${OUT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
