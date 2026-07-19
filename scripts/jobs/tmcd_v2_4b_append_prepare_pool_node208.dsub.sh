#!/bin/bash
#DSUB -n AGZV2_4B_APPEND_POOL
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-208
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

mkdir -p "${ROOT}/logs/gates" "${ROOT}/outputs/tmcd_v2/gates"
export AGZ_ROOT="${ROOT}"
export AGZ_DCA_MAX_PROMPT_LENGTH=2048
export AGZ_DCA_MAX_RESPONSE_LENGTH=1024
export AGZ_DCA_CANDIDATE_ATTN_IMPLEMENTATION=sdpa
export AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES=8
export AGZ_DCA_CANDIDATE_MAX_ATTEMPTS=3

source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"

python -s "${ROOT}/scripts/run_dca_first_round.py" \
  --root "${ROOT}" \
  --backbone qwen3.5-4b \
  --experiment-variant append_only_memory \
  --artifact-scope tmcd_v2 \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --source-round 0 \
  --allocated-gpus "${CUDA_VISIBLE_DEVICES}" \
  --seed 20260709 \
  --dca-feedback-candidates 4000 \
  --dca-rollout-n 2 \
  --dca-batch-size 80 \
  --dca-steps 25 \
  --vda-candidates 10000 \
  --vda-train-size 2400 \
  --vda-dev-size 400 \
  --vda-xplay-size 800 \
  --vda-batch-size 160 \
  --vda-steps 15 \
  --vda-rollout-n 1 \
  --vda-max-turns 16 \
  --candidate-batch-size 72 \
  --stop-after-stage build_isolated_vda_pool

if find "${ROOT}/checkpoints/tmcd_v2/ablations/append_only_memory/qwen3.5-4b/vda/round_1/trainer" \
  -maxdepth 1 -type d -name 'global_step_*' -print -quit 2>/dev/null | grep -q .; then
  echo "Pool gate unexpectedly created a VDA training checkpoint" >&2
  exit 74
fi

touch "${ROOT}/outputs/tmcd_v2/gates/4b-append-pool.SUCCEEDED"
