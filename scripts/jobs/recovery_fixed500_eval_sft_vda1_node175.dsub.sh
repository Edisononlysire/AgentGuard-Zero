#!/usr/bin/env bash
#DSUB -n AGZ_R500_EV_A_NT
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
COMMON="$ROOT/outputs/recovery/fixed500_threeway_20260719/teacher_sft_balanced"
OUT="$ROOT/outputs/recovery/fixed500_threeway_20260719/evaluations"

export AGZ_ROOT="$ROOT"
source "$ROOT/scripts/qwen35_env.sh"
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then exit 71; fi

run_eval() {
  local variant=$1
  local adapter=$2
  local eval_dir="$OUT/$variant"
  if [[ -e "$eval_dir" ]]; then
    echo "refusing to overwrite $eval_dir" >&2
    return 2
  fi
  mkdir -p "$eval_dir"
  local pids=()
  for shard in 0 1 2 3; do
    mkdir -p "${AGZ_TRITON_CACHE_ROOT}/fixed500_eval/${variant}/shard_${shard}"
    TRITON_CACHE_DIR="${AGZ_TRITON_CACHE_ROOT}/fixed500_eval/${variant}/shard_${shard}" \
      CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" python \
      scripts/eval_recovery_fixed_source.py run \
      --variant "$variant" --model-path "$MODEL" --adapter "$adapter" \
      --scenarios "$XPLAY" --device cuda:0 --shard-index "$shard" \
      --shard-count 4 --output "$eval_dir/shard_${shard}.json" \
      > "$eval_dir/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  python scripts/eval_recovery_fixed_source.py merge \
    --inputs "$eval_dir"/shard_*.json --output "$eval_dir/metrics.json"
}

run_eval "teacher_sft_balanced_nothink" "$COMMON/adapter"
echo "RECOVERY_FIXED500_EVAL_A_READY=$OUT"
