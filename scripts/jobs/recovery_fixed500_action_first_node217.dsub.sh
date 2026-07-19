#!/usr/bin/env bash
#DSUB -n AGZ_R500_ACT1ST
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
DATA="$BASE/teacher_data_sequence_balanced_cap4"
OUT="$BASE/teacher_sft_action_first_cap4"
EVAL="$BASE/evaluations/teacher_sft_action_first_cap4_quick8_nothink"

export AGZ_ROOT="$ROOT"
source "$ROOT/scripts/qwen35_env.sh"
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONUNBUFFERED=1
for path in "$OUT" "$EVAL"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite $path" >&2; exit 72; fi
done
python - "$DATA/manifest.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("accepted") is not True:
    raise SystemExit("sequence-balanced Teacher data is not accepted")
PY

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_recovery_bootstrap_sft.py \
  --arm teacher_sft --model-path "$MODEL" \
  --train-parquet "$DATA/bootstrap_sft.parquet" \
  --data-manifest "$DATA/manifest.json" --output-dir "$OUT" \
  --seed 20260719 --per-device-batch 2 --gradient-accumulation 8 \
  --min-records 2000 --max-records 2600 --epochs 1.0 --learning-rate 1e-5

mkdir -p "$EVAL"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then exit 71; fi
pids=()
for shard in 0 1 2 3; do
  cache="${AGZ_TRITON_CACHE_ROOT}/fixed500_action_first/eval_shard_${shard}"
  mkdir -p "$cache"
  TRITON_CACHE_DIR="$cache" CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" python \
    scripts/eval_recovery_fixed_source.py run \
    --variant teacher_sft_action_first_cap4_quick8_nothink \
    --model-path "$MODEL" --adapter "$OUT/adapter" --scenarios "$XPLAY" \
    --device cuda:0 --scenario-limit 8 --skip-teacher-diagnostics \
    --shard-index "$shard" --shard-count 4 \
    --output "$EVAL/shard_${shard}.json" > "$EVAL/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
python scripts/eval_recovery_fixed_source.py merge \
  --inputs "$EVAL"/shard_*.json --output "$EVAL/metrics.json"
echo "RECOVERY_FIXED500_ACTION_FIRST_READY=$OUT"
