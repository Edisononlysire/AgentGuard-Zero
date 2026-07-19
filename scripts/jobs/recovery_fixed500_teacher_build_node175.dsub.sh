#!/usr/bin/env bash
#DSUB -n AGZ_R500_SFT
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-208
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/recovery_fixed500/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/recovery_fixed500/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
PY=/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/agent0-gpu/bin/python
SOURCE="$ROOT/data/tmcd_v2_pilot_fast/qwen3.5-4b/round_1/vda_train/train.parquet"
OUT="$ROOT/outputs/recovery/fixed500_threeway_20260719/teacher_data"
SFT_OUT="$ROOT/outputs/recovery/fixed500_threeway_20260719/teacher_sft"
MODEL="$ROOT/models/qwen3_5/Qwen3.5-4B"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$OUT")"
mkdir -p "${OUT}_shards"
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite $OUT" >&2
  exit 2
fi

pids=()
for shard in 0 1 2 3; do
  shard_out="${OUT}_shards/shard_${shard}"
  "$PY" scripts/build_recovery_bootstrap.py \
    --scenarios "$SOURCE" \
    --output-dir "$shard_out" \
    --expected-scenarios 500 \
    --min-records 350 \
    --max-records 850 \
    --derive-counterfactual-worlds \
    --shard-index "$shard" \
    --shard-count 4 \
    --defer-global-distribution-gates \
    > "${OUT}_shards/shard_${shard}.stdout" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more Teacher shards failed" >&2
  tail -80 "${OUT}_shards"/*.stdout >&2 || true
  exit 3
fi

"$PY" scripts/merge_recovery_teacher_data.py \
  --shards \
    "${OUT}_shards/shard_0" \
    "${OUT}_shards/shard_1" \
    "${OUT}_shards/shard_2" \
    "${OUT}_shards/shard_3" \
  --output-dir "$OUT" \
  --expected-source-scenarios 500 \
  --min-records 1800 \
  --max-records 3000

echo "RECOVERY_FIXED500_TEACHER_DATA_READY=$OUT"

export AGZ_ROOT="$ROOT"
source "$ROOT/scripts/qwen35_env.sh"
source "$ROOT/scripts/env.sh"
if [[ -e "$SFT_OUT" ]]; then
  echo "refusing to overwrite $SFT_OUT" >&2
  exit 4
fi
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_recovery_bootstrap_sft.py \
  --arm teacher_sft \
  --model-path "$MODEL" \
  --train-parquet "$OUT/bootstrap_sft.parquet" \
  --data-manifest "$OUT/manifest.json" \
  --output-dir "$SFT_OUT" \
  --seed 20260719 \
  --per-device-batch 2 \
  --gradient-accumulation 8

echo "RECOVERY_FIXED500_TEACHER_SFT_READY=$SFT_OUT"
