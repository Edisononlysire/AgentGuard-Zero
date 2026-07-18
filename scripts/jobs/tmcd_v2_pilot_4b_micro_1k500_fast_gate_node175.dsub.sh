#!/bin/bash
#DSUB -n AGZ_FAST_GATE
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-175
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2_pilot_micro_1k500_fast_gate/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/tmcd_v2_pilot_micro_1k500_fast_gate/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-175
SCOPE=tmcd_v2_pilot_fast_gate
BACKBONE=qwen3.5-4b
SEED=20260719
OUTPUT_ROOT=${ROOT}/outputs/${SCOPE}/micro_1k500_fast_gate
SOURCE_HASHES=${ROOT}/outputs/source_snapshots/20260719_tmcd_micro_1k500_fast_prelaunch/deployed_source.sha256

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ -e "${OUTPUT_ROOT}/FAST_RESOURCE_GATE_SUCCEEDED" ]]; then
  echo "Refusing to replace a completed fast resource gate" >&2
  exit 73
fi

mkdir -p "${ROOT}/logs/tmcd_v2_pilot_micro_1k500_fast_gate" "${OUTPUT_ROOT}"
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

# The gate executes one real DCA optimizer step over 50 prompts x two
# rollouts.  Four replicas therefore receive 25 sequences each.  This is the
# largest new concurrency setting in the fast protocol and is verified before
# the independent three-round run is submitted.
export AGZ_DCA_PPO_MINI_BATCH_SIZE=50
export AGZ_DCA_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_DCA_MAX_NUM_SEQS=25
export AGZ_DCA_GPU_MEMORY_UTILIZATION=0.50
export AGZ_DCA_REWARD_FSYNC_EVERY_BATCHES=1
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
export AGZ_MAX_ACTOR_CKPT_TO_KEEP=1

python -s - <<'PY'
from verl.workers.rollout.hf_rollout import _rollout_chunk_plan

chunks, largest = _rollout_chunk_plan(
    batch_size=25,
    sequence_tokens=3072,
    max_num_seqs=25,
    max_batch_tokens=None,
)
if (chunks, largest) != (1, 25):
    raise SystemExit(
        f"fast DCA rollout partition gate failed: expected (1, 25), got {(chunks, largest)}"
    )
print("fast DCA rollout partition gate: 25 sequences -> one chunk = PASS")
PY

python -s "${ROOT}/scripts/run_dca_first_round.py" \
  --root "${ROOT}" \
  --backbone "${BACKBONE}" \
  --experiment-variant full \
  --artifact-scope "${SCOPE}" \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --source-round 0 \
  --allocated-gpus "${CUDA_VISIBLE_DEVICES}" \
  --seed "${SEED}" \
  --dca-feedback-candidates 100 \
  --dca-rollout-n 2 \
  --dca-batch-size 50 \
  --dca-steps 1 \
  --vda-candidates 1000 \
  --vda-train-size 500 \
  --vda-dev-size 100 \
  --vda-xplay-size 200 \
  --vda-batch-size 50 \
  --vda-steps 10 \
  --vda-rollout-n 2 \
  --vda-max-turns 10 \
  --vda-selection-policy pilot_balanced_50_40_10 \
  --vda-learning-rate 1e-6 \
  --vda-kl-coef 0.02 \
  --candidate-batch-size 72 \
  --stop-after-stage update_dca

python -s - "${ROOT}" "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
manifest_path = root / "checkpoints/tmcd_v2_pilot_fast_gate/qwen3.5-4b/dca/round_1/manifest.json"
feedback_path = root / "data/tmcd_v2_pilot_fast_gate/qwen3.5-4b/round_1/dca_feedback/feedback.jsonl"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config = manifest["training_config"]
expected = {
    "feedback_candidates": 100,
    "rollout_n": 2,
    "batch_size": 50,
    "round_steps": 1,
    "rollout_max_num_seqs": 25,
}
actual = {key: config.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"fast resource gate manifest mismatch: {actual} != {expected}")
rows = sum(1 for _ in feedback_path.open(encoding="utf-8"))
if rows != 100:
    raise SystemExit(f"fast resource gate feedback mismatch: {rows} != 100")
report = {
    "schema_version": 1,
    "status": "accepted",
    "kind": "tmcd_micro_fast_resource_gate",
    "dca_manifest": str(manifest_path),
    "feedback_rows": rows,
    "training_config": actual,
}
(output / "acceptance.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

touch "${OUTPUT_ROOT}/FAST_RESOURCE_GATE_SUCCEEDED"
