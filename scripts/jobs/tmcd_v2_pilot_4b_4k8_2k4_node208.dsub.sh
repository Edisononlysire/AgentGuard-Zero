#!/bin/bash
#DSUB -n AGZ_4K8_2K4
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-208
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2_pilot_4k8_2k4/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2_pilot_4k8_2k4/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-208
SCOPE=tmcd_v2_pilot_fast_4k8_2k4
BACKBONE=qwen3.5-4b
SEED=20260719
OUTPUT_ROOT=${ROOT}/outputs/${SCOPE}/scale_4k8_2k4
EVAL_ROOT=${OUTPUT_ROOT}/eval
GATE_ROOT=${OUTPUT_ROOT}/gates
SOURCE_HASHES=${ROOT}/outputs/source_snapshots/20260719_tmcd_4k8_2k4_prelaunch/deployed_source.sha256

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ -e "${OUTPUT_ROOT}/COEVOLUTION_SUCCEEDED" ]]; then
  echo "Refusing to replace a completed 4,800-to-2,400 lineage" >&2
  exit 73
fi

mkdir -p "${ROOT}/logs/tmcd_v2_pilot_4k8_2k4" "${EVAL_ROOT}" "${GATE_ROOT}"
export AGZ_ROOT=${ROOT}
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

sha256sum -c "${SOURCE_HASHES}"
export PYTHONUNBUFFERED=1
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Expected exactly four allocated GPUs, got ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 74
fi

# DCA uses the throughput settings already exercised by the formal 4B run:
# 80 prompts x two rollouts x 30 steps = 4,800 feedback rows per round.
export AGZ_DCA_PPO_MINI_BATCH_SIZE=40
export AGZ_DCA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_MAX_NUM_SEQS=160
export AGZ_DCA_GPU_MEMORY_UTILIZATION=0.50
export AGZ_DCA_REWARD_FSYNC_EVERY_BATCHES=8
export AGZ_DCA_CANDIDATE_ATTN_IMPLEMENTATION=sdpa
export AGZ_DCA_CANDIDATE_PARTIAL_FSYNC_EVERY_BATCHES=16
export AGZ_DCA_CANDIDATE_MAX_ATTEMPTS=3

# VDA consumes all 2,400 train scenarios once.  Each generation batch of 96
# is split into three 32-scenario PPO mini-updates: 25 generation batches and
# 75 optimizer updates per round, matching the proven formal 4B layout.
export AGZ_ROLLOUT_BACKEND=hf
export AGZ_AGENT_NUM_WORKERS=4
export AGZ_MAX_NUM_SEQS=32
export AGZ_GPU_MEMORY_UTILIZATION=0.50
export AGZ_VDA_GENERATION_BATCH_SIZE=96
export AGZ_VDA_PPO_MINI_BATCH_SIZE=32
export AGZ_VDA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_VDA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
export AGZ_ROLLOUT_SERVER_MAX_PARALLEL_TRAJECTORIES=24
export AGZ_ROLLOUT_SERVER_MAX_STATES=512
export AGZ_VDA_ACTION_TOKENS=320
export AGZ_VDA_OBSERVATION_TOKENS=512
export AGZ_MAX_ACTOR_CKPT_TO_KEEP=1
export AGZ_ENABLE_GRADIENT_CHECKPOINTING=true

export AGZ_ROLLOUT_TEMPERATURE=0.8
export AGZ_ROLLOUT_TOP_P=0.95
export AGZ_ROLLOUT_TOP_K=0
export AGZ_VDA_FEEDBACK_CONTINUATION_PROMPT_MODE=snapshot
export AGZ_VDA_FEEDBACK_HISTORY_WINDOW=6
export AGZ_VDA_FEEDBACK_MAX_TURNS=10
export AGZ_VDA_FEEDBACK_MAX_INPUT_TOKENS=2048
export AGZ_VDA_FEEDBACK_MAX_NEW_TOKENS=320
export AGZ_VDA_FEEDBACK_INVALID_ACTION_PATIENCE=2
export AGZ_VDA_FEEDBACK_ATTN_IMPLEMENTATION=sdpa

python -s - <<'PY'
from verl.workers.rollout.hf_rollout import _rollout_chunk_plan

chunks, largest = _rollout_chunk_plan(
    batch_size=40,
    sequence_tokens=3072,
    max_num_seqs=160,
    max_batch_tokens=None,
)
if (chunks, largest) != (1, 40):
    raise SystemExit(
        f"4k8 DCA rollout partition gate failed: expected (1, 40), got {(chunks, largest)}"
    )
print("4k8 DCA rollout partition: 40 sequences per replica = PASS")
PY

common_round_args=(
  --root "${ROOT}"
  --backbone "${BACKBONE}"
  --experiment-variant full
  --artifact-scope "${SCOPE}"
  --model-path "${AGZ_QWEN35_4B_PATH}"
  --allocated-gpus "${CUDA_VISIBLE_DEVICES}"
  --seed "${SEED}"
  --dca-feedback-candidates 4800
  --dca-rollout-n 2
  --dca-batch-size 80
  --dca-steps 30
  --vda-candidates 4800
  --vda-train-size 2400
  --vda-dev-size 400
  --vda-xplay-size 800
  --vda-batch-size 32
  --vda-steps 75
  --vda-rollout-n 2
  --vda-max-turns 10
  --vda-selection-policy pilot_balanced_50_40_10
  --vda-learning-rate 1e-6
  --vda-kl-coef 0.02
  --candidate-batch-size 72
)

manifest_adapter() {
  python -s - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("")
    raise SystemExit(0)
print(json.load(path.open(encoding="utf-8")).get("adapter_path") or "")
PY
}

run_eval() {
  local data_path=$1
  local system_name=$2
  local run_name=$3
  local adapter_path=$4
  local command=(
    python -s "${ROOT}/scripts/run_tmcd_eval_four_gpu.py"
    --data "${data_path}"
    --system "${system_name}"
    --run-name "${run_name}"
    --output-dir "${EVAL_ROOT}"
    --model-path "${AGZ_QWEN35_4B_PATH}"
    --model-backend hf
    --candidate-count 1
    --limit 800
    --split all
    --max-turns 10
    --trajectory-batch-size 16
    --max-input-tokens 2048
    --max-new-tokens 320
    --temperature 1.0
    --top-p 1.0
    --top-k 0
    --dtype bf16
    --attn-implementation sdpa
    --seed "${SEED}"
  )
  if [[ -n "${adapter_path}" ]]; then
    command+=(--adapter-path "${adapter_path}")
  fi
  "${command[@]}"
}

for source_round in 0 1 2; do
  target_round=$((source_round + 1))
  echo "SCALE_STAGE round=${target_round} phase=dca_update_and_fresh_pool started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  round_data=${ROOT}/data/${SCOPE}/${BACKBONE}/round_${target_round}
  pool_manifest=${round_data}/vda_pool_manifest.json
  xplay_data=${round_data}/vda_xplay/xplay.parquet
  parent_manifest=${ROOT}/checkpoints/${SCOPE}/${BACKBONE}/vda/round_${source_round}/manifest.json
  target_manifest=${ROOT}/checkpoints/${SCOPE}/${BACKBONE}/vda/round_${target_round}/manifest.json
  pre_name=scale_4k8_2k4_r${target_round}_pre
  post_name=scale_4k8_2k4_r${target_round}_post
  gate_path=${GATE_ROOT}/round_${target_round}.json

  python -s "${ROOT}/scripts/run_dca_first_round.py" \
    --source-round "${source_round}" \
    "${common_round_args[@]}" \
    --stop-after-stage build_isolated_vda_pool

  parent_adapter=$(manifest_adapter "${parent_manifest}")
  pre_system=agentguard_zero_train
  if [[ -z "${parent_adapter}" ]]; then
    pre_system=qwen_zero_shot_vda
  fi
  run_eval "${xplay_data}" "${pre_system}" "${pre_name}" "${parent_adapter}"

  echo "SCALE_STAGE round=${target_round} phase=vda_update started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python -s "${ROOT}/scripts/run_dca_first_round.py" \
    --source-round "${source_round}" \
    "${common_round_args[@]}"

  target_adapter=$(manifest_adapter "${target_manifest}")
  if [[ -z "${target_adapter}" ]]; then
    echo "Target VDA adapter is missing for round ${target_round}" >&2
    exit 75
  fi
  run_eval "${xplay_data}" agentguard_zero_train "${post_name}" "${target_adapter}"

  echo "SCALE_STAGE round=${target_round} phase=gate started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ ! -e "${gate_path}" ]]; then
    python -s "${ROOT}/scripts/audit_micro_coevolution_eval.py" \
      --pre-run-dir "${EVAL_ROOT}/${pre_name}" \
      --post-run-dir "${EVAL_ROOT}/${post_name}" \
      --pool-manifest "${pool_manifest}" \
      --vda-manifest "${target_manifest}" \
      --round "${target_round}" \
      --expected-scenarios 800 \
      --expected-candidates 4800 \
      --expected-train-size 2400 \
      --output "${gate_path}"
  fi
  python -s - "${gate_path}" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
if gate.get("accepted") is not True:
    raise SystemExit(f"4k8-to-2k4 round gate rejected: {gate.get('failures')}")
PY
  echo "SCALE_STAGE round=${target_round} phase=accepted finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

python -s "${ROOT}/scripts/audit_dca_first_lineage.py" \
  --root "${ROOT}" \
  --backbone "${BACKBONE}" \
  --artifact-scope "${SCOPE}" \
  --experiment-variant full \
  --max-round 3 \
  --expected-host "${EXPECTED_NODE}" \
  --output "${OUTPUT_ROOT}/lineage_audit.json"

touch "${OUTPUT_ROOT}/COEVOLUTION_SUCCEEDED"
