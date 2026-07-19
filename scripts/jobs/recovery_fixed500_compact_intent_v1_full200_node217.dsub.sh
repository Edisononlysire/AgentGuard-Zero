#!/usr/bin/env bash
#DSUB -n AGZ_R500_INTV1E
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-217
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/recovery_fixed500/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/recovery_fixed500/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
MODEL="$ROOT/models/qwen3_5/Qwen3.5-4B"
XPLAY="$ROOT/data/tmcd_v2_pilot_fast/qwen3.5-4b/round_1/vda_xplay/xplay.parquet"
BASE="$ROOT/outputs/recovery/fixed500_threeway_20260719"
ADAPTER="$BASE/teacher_sft_compact_intent_3ep/adapter"
PARENT_MANIFEST="$BASE/teacher_sft_compact_intent_3ep/manifest.json"
EVAL="$BASE/evaluations/teacher_sft_compact_intent_3ep_full200"

export AGZ_ROOT="$ROOT"
source "$ROOT/scripts/qwen35_env.sh"
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONUNBUFFERED=1
if [[ -e "$EVAL" ]]; then
  echo "refusing to overwrite $EVAL" >&2
  exit 72
fi
python - "$PARENT_MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    payload.get("status") != "trained_pending_k1_evaluation"
    or payload.get("thinking_mode") != "disabled"
    or payload.get("prompt_target_token_prefix_exact") is not True
    or payload.get("target_serialization") != "compact_public_action_intent_v1"
    or payload.get("adapter_sha256")
    != "34e940ec13f52f311765a4a9de76742ba78ff8400d5df46961f8cbd703414192"
):
    raise SystemExit("compact-intent V1 adapter identity or invariants changed")
PY

mkdir -p "$EVAL"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then exit 71; fi
pids=()
for shard in 0 1 2 3; do
  cache="${AGZ_TRITON_CACHE_ROOT}/fixed500_intent_v1_full200/eval_shard_${shard}"
  mkdir -p "$cache"
  TRITON_CACHE_DIR="$cache" CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" python \
    scripts/eval_recovery_fixed_source.py run \
    --variant teacher_sft_compact_intent_3ep_full200 \
    --model-path "$MODEL" --adapter "$ADAPTER" --scenarios "$XPLAY" \
    --device cuda:0 --skip-teacher-diagnostics \
    --output-format compact_public_action_intent_v1 \
    --shard-index "$shard" --shard-count 4 \
    --output "$EVAL/shard_${shard}.json" > "$EVAL/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
python scripts/eval_recovery_fixed_source.py merge \
  --inputs "$EVAL"/shard_*.json --output "$EVAL/metrics.json"
echo "RECOVERY_FIXED500_COMPACT_INTENT_V1_FULL200_READY=$EVAL"
