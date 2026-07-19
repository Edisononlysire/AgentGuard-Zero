#!/bin/bash
#DSUB -n AGZV2_4B_GATE_FIN
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/gates/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
OUT=${ROOT}/outputs/tmcd_v2/gates/vda-full4-partial-b160-node175
DATA=${ROOT}/outputs/tmcd_v2/gates/vda-full4-partial-data
RUN_NAME=agz_gate_qwen3.5-4b_partial_vda_batch160_node175

if [[ "$(hostname)" != "cyclone001-agent-175" ]]; then
  echo "Refusing to run outside cyclone001-agent-175: $(hostname)" >&2
  exit 72
fi
if [[ -e "${OUT}/SUCCEEDED" ]]; then
  echo "Gate was already finalized: ${OUT}" >&2
  exit 73
fi

export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"

python -s "${ROOT}/scripts/validate_vda_training_log.py" \
  --log "${ROOT}/logs/${RUN_NAME}.log" \
  --output "${OUT}/training_metrics.json" \
  --expected-step 1 \
  --action-budget 320 \
  --observation-budget 1280

python -s "${ROOT}/scripts/finalize_vda_training_gate.py" \
  --output-dir "${OUT}" \
  --backbone qwen3.5-4b \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --parent-manifest "${ROOT}/checkpoints/tmcd_v2/qwen3.5-4b/vda/round_0/manifest.json" \
  --pool-manifest "${DATA}/manifest.json" \
  --dca-manifest "${ROOT}/checkpoints/tmcd_v2/qwen3.5-4b/dca/round_1/manifest.json" \
  --checkpoint-root "${OUT}/checkpoints" \
  --batch-size 160 \
  --seed 20260709

CUDA_VISIBLE_DEVICES=0 python -s "${ROOT}/scripts/validate_adapter_reload.py" \
  --checkpoint-manifest "${OUT}/checkpoint_manifest.json" \
  --output "${OUT}/adapter_reload.json"

python -s "${ROOT}/scripts/prune_gate_recovery_checkpoint.py" \
  --checkpoint-manifest "${OUT}/checkpoint_manifest.json" \
  --output "${OUT}/recovery_pruned.json"

touch "${OUT}/SUCCEEDED"
