#!/usr/bin/env bash
#DSUB -n AGZ_R500_DAG_NT
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -pn cyclone001-agent-208
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/recovery_fixed500/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/recovery_fixed500/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
MODEL="$ROOT/models/qwen3_5/Qwen3.5-4B"
SOURCE="$ROOT/data/tmcd_v2_pilot_fast/qwen3.5-4b/round_1/vda_train/train.parquet"
XPLAY="$ROOT/data/tmcd_v2_pilot_fast/qwen3.5-4b/round_1/vda_xplay/xplay.parquet"
BASE="$ROOT/outputs/recovery/fixed500_threeway_20260719"
TEACHER="$BASE/teacher_data_balanced"
COMMON="$BASE/teacher_sft_balanced"
SHARDS="$BASE/dagger_balanced_nothink_shards"
DAGGER="$BASE/dagger_data_balanced_nothink"
AGGREGATE="$BASE/dagger_aggregate_balanced_nothink"
OUT="$BASE/teacher_sft_balanced_plus_dagger_nothink"
EVAL="$BASE/evaluations/teacher_sft_balanced_plus_dagger_nothink"

export AGZ_ROOT="$ROOT"
source "$ROOT/scripts/qwen35_env.sh"
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONUNBUFFERED=1
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then exit 71; fi
for path in "$SHARDS" "$DAGGER" "$AGGREGATE" "$OUT" "$EVAL"; do
  if [[ -e "$path" ]]; then echo "refusing to overwrite $path" >&2; exit 72; fi
done
mkdir -p "$SHARDS" "$EVAL"

pids=()
for shard in 0 1 2 3; do
  mkdir -p "${AGZ_TRITON_CACHE_ROOT}/fixed500_dagger_nothink/collect_shard_${shard}"
  TRITON_CACHE_DIR="${AGZ_TRITON_CACHE_ROOT}/fixed500_dagger_nothink/collect_shard_${shard}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" python \
    scripts/collect_recovery_dagger.py \
    --adapter-manifest "$COMMON/manifest.json" \
    --model-path "$MODEL" --scenarios "$SOURCE" \
    --output-dir "$SHARDS/shard_${shard}" --device cuda:0 \
    --derive-counterfactual-worlds --expected-source-scenarios 500 \
    --shard-index "$shard" --shard-count 4 \
    --min-records 200 --max-records 900 \
    --defer-global-distribution-gates \
    > "$SHARDS/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

python scripts/merge_recovery_teacher_data.py \
  --shards "$SHARDS/shard_0" "$SHARDS/shard_1" "$SHARDS/shard_2" "$SHARDS/shard_3" \
  --output-dir "$DAGGER" --expected-source-scenarios 500 \
  --min-records 1000 --max-records 3000 \
  --kind recovery_single_dagger_dataset \
  --train-filename dagger_correction.parquet \
  --audit-filename teacher_relabel_audit.jsonl

python scripts/aggregate_recovery_training_data.py \
  --bootstrap-dir "$TEACHER" --dagger-dir "$DAGGER" --output-dir "$AGGREGATE"

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_recovery_bootstrap_sft.py \
  --arm teacher_dagger --model-path "$MODEL" \
  --init-adapter "$COMMON/adapter" \
  --train-parquet "$AGGREGATE/bootstrap_sft.parquet" \
  --data-manifest "$AGGREGATE/manifest.json" --output-dir "$OUT" \
  --seed 20260719 --per-device-batch 2 --gradient-accumulation 8 \
  --approved-stage dagger_sft --min-records 2800 --max-records 6000 \
  --epochs 0.5 --learning-rate 5e-6

pids=()
for shard in 0 1 2 3; do
  mkdir -p "${AGZ_TRITON_CACHE_ROOT}/fixed500_dagger_nothink/eval_shard_${shard}"
  TRITON_CACHE_DIR="${AGZ_TRITON_CACHE_ROOT}/fixed500_dagger_nothink/eval_shard_${shard}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" python \
    scripts/eval_recovery_fixed_source.py run \
    --variant teacher_sft_balanced_plus_dagger_nothink --model-path "$MODEL" \
    --adapter "$OUT/adapter" --scenarios "$XPLAY" --device cuda:0 \
    --shard-index "$shard" --shard-count 4 \
    --output "$EVAL/shard_${shard}.json" > "$EVAL/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
python scripts/eval_recovery_fixed_source.py merge \
  --inputs "$EVAL"/shard_*.json --output "$EVAL/metrics.json"
echo "RECOVERY_FIXED500_DAGGER_READY=$OUT"
