# AgentGuard-Zero TMCD Experiment Runbook

This runbook fixes the final AAAI experiment protocol for Trust-Manipulated
Autonomous Cyber Defense (TMCD). Do not expand the baseline set unless the paper
plan changes.

## Fixed Systems

| System | Trains Parameters | Model | Runner |
|---|---:|---|---|
| Rule-based SOC | No | None | `scripts/eval_tmcd_static_cpu_dsub.sh` |
| ReAct / Base+Tools | No | Qwen3.5-9B | `scripts/eval_tmcd_system_dsub.sh` |
| Memory Agent | No | Qwen3.5-9B | `scripts/eval_tmcd_system_dsub.sh` |
| Trust-score Agent | No | Qwen3.5-9B | `scripts/eval_tmcd_system_dsub.sh` |
| Cyber LLM VDA | No | Foundation-Sec-8B-Instruct | `scripts/eval_tmcd_system_dsub.sh` |
| Qwen Zero-shot VDA | No | Qwen3.5-9B | `scripts/eval_tmcd_system_dsub.sh` |
| AgentGuard-Zero-Select | No | Qwen3.5-9B | `scripts/eval_tmcd_system_dsub.sh` |
| AgentGuard-Zero-Train | Yes, LoRA | Qwen3.5-9B + adapter | `scripts/train_agentguard_zero_train_qwen35_9b_dsub.sh` |
| Oracle Defender | No | None | `scripts/eval_tmcd_static_cpu_dsub.sh` |

## Environment

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
source scripts/qwen35_env.sh
```

Cyber LLM environment:

```bash
source scripts/cyber_llm_env.sh
```

Expected model paths:

```text
Qwen3.5-9B:
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/models/qwen3_5/Qwen3.5-9B

Foundation-Sec-8B-Instruct:
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/models/cyber_llm/Foundation-Sec-8B-Instruct
```

## CPU Smoke Before GPU Jobs

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
bash scripts/smoke_tmcd_cpu.sh
```

This runs the symbolic rollout self-test, static policies, and mock model
policies on a tiny subset, then exports smoke tables into:

```text
outputs/paper_tables_smoke
```

## Build Level 2 CAGE-style Data

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
source scripts/qwen35_env.sh
python -s scripts/build_cage_style_level2.py \
  --input data/level1/level1_seed20260706_n500_frontier_vda.parquet \
  --output data/level2/cage_style_t3_t4_vda.parquet \
  --limit_per_subset 64
```

## Table 1 / Figure 1 Main Evaluation

CPU/static systems:

```bash
AGZ_SYSTEM=rule_based_soc AGZ_EVAL_LIMIT=256 \
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/eval_tmcd_static_cpu_dsub.sh

AGZ_SYSTEM=oracle_defender AGZ_EVAL_LIMIT=256 \
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/eval_tmcd_static_cpu_dsub.sh
```

GPU/model systems:

```bash
for S in react_base_tools memory_agent trust_score_agent qwen_zero_shot_vda agentguard_zero_select cyber_llm_vda; do
  AGZ_SYSTEM=$S AGZ_EVAL_LIMIT=256 \
  dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/eval_tmcd_system_dsub.sh
done
```

After AgentGuard-Zero-Train finishes, evaluate the adapter:

```bash
AGZ_SYSTEM=agentguard_zero_train \
AGZ_ADAPTER_PATH=/path/to/lora_adapter \
AGZ_EVAL_LIMIT=256 \
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/eval_tmcd_system_dsub.sh
```

## AgentGuard-Zero-Train

```bash
AGZ_MAX_STEPS=50 \
AGZ_ROLLOUT_N=2 \
AGZ_BATCH_SIZE=2 \
AGZ_PPO_MINI_BATCH_SIZE=2 \
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/train_agentguard_zero_train_qwen35_9b_dsub.sh
```

The wrapper uses Qwen3.5-9B, LoRA rank 16, Level-1 frontier parquet, and the
trajectory reward from the Level-1 rollout server.

## Table 3 Level 2 Transfer

```bash
for S in rule_based_soc react_base_tools memory_agent trust_score_agent cyber_llm_vda agentguard_zero_select; do
  if [[ "$S" == "rule_based_soc" ]]; then
    AGZ_SYSTEM=$S \
    AGZ_EVAL_DATA=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/data/level2/cage_style_t3_t4_vda.parquet \
    AGZ_EVAL_LIMIT=128 \
    dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/eval_tmcd_static_cpu_dsub.sh
  else
    AGZ_SYSTEM=$S \
    AGZ_EVAL_DATA=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/data/level2/cage_style_t3_t4_vda.parquet \
    AGZ_EVAL_LIMIT=128 \
    dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/eval_tmcd_system_dsub.sh
  fi
done
```

Evaluate `agentguard_zero_train` on Level 2 after the adapter exists.

## Optional API Transfer Check

This is for low-cost black-box transfer debugging only. It uses the same TMCD
runner as the fixed nine-system protocol, but does not occupy GPU resources.

Put API credentials in `~/.agentguard_api_env` or pass `AGZ_API_ENV_FILE`:

```bash
export AGZ_API_MODEL=glm-5.1
export AGZ_API_BASE_URL=https://your-openai-compatible-endpoint/v1
export AGZ_API_KEY=...
```

Then run a small Select or Zero-shot check:

```bash
AGZ_SYSTEM=agentguard_zero_select AGZ_EVAL_LIMIT=16 \
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/eval_tmcd_api_cpu_dsub.sh

AGZ_SYSTEM=qwen_zero_shot_vda AGZ_EVAL_LIMIT=16 \
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/eval_tmcd_api_cpu_dsub.sh
```

## Export Paper Tables

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
source scripts/qwen35_env.sh
python -s scripts/export_tmcd_tables.py \
  --input_dir outputs/tmcd_eval \
  --output_dir outputs/paper_tables
```

Outputs:

```text
outputs/paper_tables/table1_overall_results.md
outputs/paper_tables/table1_overall_results.csv
outputs/paper_tables/figure1_task_heatmap.md
outputs/paper_tables/figure1_task_heatmap.csv
outputs/paper_tables/table3_cage_transfer.md
outputs/paper_tables/table3_cage_transfer.csv
```

## Logs

All dsub logs go to:

```text
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs
```

Use:

```bash
djob
```

to view submitted jobs.
