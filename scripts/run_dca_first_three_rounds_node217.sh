#!/bin/bash
set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
AGZ_PYTHON=${AGZ_PYTHON:-/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/agent0-gpu/bin/python}
START_ROUND=${AGZ_START_ROUND:-0}
END_ROUND=${AGZ_END_ROUND:-2}

if (( START_ROUND < 0 || END_ROUND < START_ROUND || END_ROUND > 2 )); then
  echo "Expected 0 <= AGZ_START_ROUND <= AGZ_END_ROUND <= 2" >&2
  exit 64
fi

for source_round in $(seq "${START_ROUND}" "${END_ROUND}"); do
  echo "Starting ${AGZ_BACKBONE:?AGZ_BACKBONE is required} DCA-first source round ${source_round}"
  AGZ_SOURCE_ROUND="${source_round}" \
    /usr/bin/bash "${ROOT}/scripts/run_dca_first_round_node217.sh"
done

if [[ "${AGZ_PILOT:-0}" != "1" && "${END_ROUND}" == "2" ]]; then
  candidate_batch_size=4
  if [[ "${AGZ_BACKBONE}" == "qwen3.5-9b" ]]; then
    candidate_batch_size=2
  fi
  "${AGZ_PYTHON}" -s "${ROOT}/scripts/generate_final_heldout_node217.py" \
    --root "${ROOT}" \
    --backbone "${AGZ_BACKBONE}" \
    --candidate-count "${AGZ_FINAL_HELDOUT_CANDIDATES:-4000}" \
    --per-task "${AGZ_FINAL_HELDOUT_PER_TASK:-200}" \
    --batch-size "${AGZ_FINAL_HELDOUT_BATCH_SIZE:-${candidate_batch_size}}" \
    --seed "${AGZ_FINAL_HELDOUT_SEED:-21160709}" \
    --allocated-gpus "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

  "${AGZ_PYTHON}" -s "${ROOT}/scripts/audit_dca_first_lineage.py" \
    --root "${ROOT}" \
    --backbone "${AGZ_BACKBONE}" \
    --artifact-scope formal \
    --max-round 3
fi
