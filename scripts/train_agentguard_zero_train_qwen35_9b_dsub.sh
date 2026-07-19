#!/bin/bash
#DSUB -n AGZTrain9B
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=16;gpu=4;mem=200000"
#DSUB -ex job
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

# Thin paper-protocol wrapper around the existing LoRA training entry.
set -euo pipefail

export AGZ_QWEN35_SIZE=${AGZ_QWEN35_SIZE:-9B}
export AGZ_RUN_NAME=${AGZ_RUN_NAME:-agentguard_zero_train_qwen35_9b_${BATCH_JOB_ID:-manual}}
export AGZ_LEVEL1_FRONTIER_FILE=${AGZ_LEVEL1_FRONTIER_FILE:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/data/level1/level1_seed20260706_n500_frontier_vda.parquet}
export AGZ_ROLLOUT_N=${AGZ_ROLLOUT_N:-2}
export AGZ_BATCH_SIZE=${AGZ_BATCH_SIZE:-2}
export AGZ_PPO_MINI_BATCH_SIZE=${AGZ_PPO_MINI_BATCH_SIZE:-2}
export AGZ_MAX_STEPS=${AGZ_MAX_STEPS:-50}
export AGZ_LORA_RANK=${AGZ_LORA_RANK:-16}
export AGZ_LORA_ALPHA=${AGZ_LORA_ALPHA:-32}
export AGZ_GPU_MEMORY_UTILIZATION=${AGZ_GPU_MEMORY_UTILIZATION:-0.16}
export AGZ_MAX_PROMPT_LENGTH=${AGZ_MAX_PROMPT_LENGTH:-1536}
export AGZ_MAX_RESPONSE_LENGTH=${AGZ_MAX_RESPONSE_LENGTH:-128}
export AGZ_AGENT_MAX_TURNS=${AGZ_AGENT_MAX_TURNS:-5}

exec /usr/bin/bash /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/train_vda_qwen35_lora_dsub.sh
