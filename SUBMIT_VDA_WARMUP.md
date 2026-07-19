# VDA Warmup Submission Runbook

Root directory:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
```

## 1. Pre-submit Check

Run this on the node where you prepare the job. It does not launch training.

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
bash scripts/pre_submit_vda_warmup.sh
```

Expected final line:

```text
Pre-submit checks passed.
```

This verifies:

- torch / vLLM / verl import under conda `agent0-gpu`
- AgentGuard-Zero VDA reward function
- smoke scenario JSON
- VDA warmup parquet at `data/smoke/vda_train.parquet`

## 2. Submit Warmup Training

Submit the queued job from the cluster submission environment:

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
dsub -s scripts/train_vda_warmup_dsub.sh
```

The generic script does not pin a node. If you know which node is free, use one
available scheduler placement. Do not run `train_vda_warmup_smoke.sh` directly
on a busy shared compute node unless you already have allocated GPUs.

## 3. What The Smoke Job Does

Default job:

- model: `/home/share/huadjyin/home/s_qinhua2/02code/guozhihan/InfectModel/model/Qwen/Qwen3-8B`
- env: conda `agent0-gpu`
- env path: `/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/agent0-gpu`
- GPUs requested: 4
- rollout samples: 2
- training steps: 2
- train file: `data/smoke/vda_train.parquet`
- reward: `curriculum_train/examples/reward_function/vda_reward.py`
- checkpoints: `outputs/checkpoints`
- logs: `logs/`

## 4. Useful Overrides

Set these before submission if needed:

```bash
export AGZ_MAX_STEPS=1
export AGZ_MODEL_PATH=/path/to/smaller/model
export AGZ_ROLLOUT_N=2
export AGZ_BATCH_SIZE=1
export AGZ_PPO_MINI_BATCH_SIZE=1
```

Then submit:

```bash
dsub -s scripts/train_vda_warmup_dsub.sh
```

## 5. Check Logs

After submission, check the scheduler output under:

```bash
ls -lt logs | head
tail -f logs/<JOB_ID>.out
tail -f logs/<JOB_ID>.err
```

The inner trainer log is:

```bash
tail -f logs/agentguard_vda_warmup_<JOB_ID>.log
```

## 6. Smoke Success Criteria

The warmup entry is considered runnable when the log shows:

- dataset loaded from `data/smoke/vda_train.parquet`
- custom reward function loaded from `vda_reward.py`
- model rollout begins
- at least one reward value is printed
- at least one training step finishes
- checkpoint or trainer step record is written
