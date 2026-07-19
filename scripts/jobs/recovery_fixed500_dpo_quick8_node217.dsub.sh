#!/usr/bin/env bash
#DSUB -n AGZ_R500_DPO_Q8
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
ADAPTER="$ROOT/outputs/recovery/fixed500_threeway_20260719/teacher_sft_balanced_plus_dpo/adapter"
OUT="$ROOT/outputs/recovery/fixed500_threeway_20260719/evaluations/teacher_sft_balanced_plus_dpo_quick8_nothink"

export AGZ_ROOT="$ROOT"
source "$ROOT/scripts/qwen35_env.sh"
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONUNBUFFERED=1
if [[ -e "$OUT" ]]; then echo "refusing to overwrite $OUT" >&2; exit 72; fi
mkdir -p "$OUT"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then exit 71; fi

pids=()
for shard in 0 1 2 3; do
  cache="${AGZ_TRITON_CACHE_ROOT}/fixed500_dpo_quick8_nothink/shard_${shard}"
  mkdir -p "$cache"
  TRITON_CACHE_DIR="$cache" CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" python \
    scripts/eval_recovery_fixed_source.py run \
    --variant teacher_sft_balanced_plus_dpo_quick8_nothink \
    --model-path "$MODEL" --adapter "$ADAPTER" --scenarios "$XPLAY" \
    --device cuda:0 --scenario-limit 8 --skip-teacher-diagnostics \
    --shard-index "$shard" --shard-count 4 \
    --output "$OUT/shard_${shard}.json" > "$OUT/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
python scripts/eval_recovery_fixed_source.py merge \
  --inputs "$OUT"/shard_*.json --output "$OUT/metrics.json"
echo "RECOVERY_FIXED500_DPO_QUICK8_READY=$OUT"
