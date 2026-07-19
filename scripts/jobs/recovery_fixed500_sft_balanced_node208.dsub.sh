#!/usr/bin/env bash
#DSUB -n AGZ_R500_SFT_BAL
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-208
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/recovery_fixed500/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/recovery_fixed500/%J.err

set -euo pipefail

ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
DATA="$ROOT/outputs/recovery/fixed500_threeway_20260719/teacher_data_balanced"
OUT="$ROOT/outputs/recovery/fixed500_threeway_20260719/teacher_sft_balanced"
MODEL="$ROOT/models/qwen3_5/Qwen3.5-4B"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export AGZ_ROOT="$ROOT"
source "$ROOT/scripts/qwen35_env.sh"
source "$ROOT/scripts/env.sh"

if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite $OUT" >&2
  exit 2
fi

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_recovery_bootstrap_sft.py \
  --arm teacher_sft \
  --model-path "$MODEL" \
  --train-parquet "$DATA/bootstrap_sft.parquet" \
  --data-manifest "$DATA/manifest.json" \
  --output-dir "$OUT" \
  --seed 20260719 \
  --per-device-batch 2 \
  --gradient-accumulation 8 \
  --min-records 1800 \
  --max-records 3000 \
  --epochs 1 \
  --learning-rate 1e-5

echo "RECOVERY_FIXED500_TEACHER_SFT_BALANCED_READY=$OUT"
