# Qwen3.5 Deployment Notes

## Framework Decision

Use the existing AgentGuard-Zero training stack:

- `agent0-gpu` conda environment.
- SM80 PyTorch overlay for A100 compatibility.
- Project-local Transformers overlay for Qwen3.5 support.
- `verl` + FSDP + HF rollout.
- LoRA only; do not full-finetune 4B/9B for the first pilot.

Do not use SWIFT for the first Qwen3.5 run. The local `verl` path already supports the reward/tool-server training loop, and Qwen3.5 loads under `transformers==5.13.0`.

## Model Paths

- 4B: `/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/models/qwen3_5/Qwen3.5-4B`
- 9B: `/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/models/qwen3_5/Qwen3.5-9B`

Integrity check passed:

- 4B: 2 shards, no missing files, no incomplete files, 9,319,828,096 shard bytes.
- 9B: 4 shards, no missing files, no incomplete files, 19,306,310,880 shard bytes.

## Environment Entry

Source:

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
source scripts/qwen35_env.sh
```

Expected versions:

- `transformers==5.13.0`
- `huggingface_hub==1.5.0`
- `safetensors==0.8.0`

The Qwen3.5 overlay is project-local:

```text
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/env_overlays/transformers_qwen35_latest
```

## Smoke

Direct CPU smoke:

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
source scripts/qwen35_env.sh
python -s scripts/qwen35_text_smoke.py \
  --model "$AGZ_QWEN35_4B_PATH" \
  --model "$AGZ_QWEN35_9B_PATH"
```

dsub CPU smoke:

```bash
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/smoke_qwen35_text_dsub.sh
```

## Training Submission

Default 4B LoRA pilot:

```bash
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/train_vda_qwen35_lora_dsub.sh
```

Useful overrides:

```bash
AGZ_QWEN35_SIZE=4B \
AGZ_MAX_STEPS=20 \
AGZ_BATCH_SIZE=4 \
AGZ_PPO_MINI_BATCH_SIZE=4 \
AGZ_LORA_RANK=16 \
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/train_vda_qwen35_lora_dsub.sh
```

9B pilot, only after 4B is stable:

```bash
AGZ_QWEN35_SIZE=9B \
AGZ_MAX_STEPS=10 \
AGZ_BATCH_SIZE=2 \
AGZ_PPO_MINI_BATCH_SIZE=2 \
AGZ_GPU_MEMORY_UTILIZATION=0.16 \
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/train_vda_qwen35_lora_dsub.sh
```

## Verified Loading Behavior

Under the Qwen3.5 overlay:

- Config: `Qwen3_5Config`
- Processor: `Qwen3VLProcessor`
- Tokenizer: `Qwen2Tokenizer`
- Empty causal model: `Qwen3_5ForCausalLM`
- Empty conditional model: `Qwen3_5ForConditionalGeneration`
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

For text-only VDA training, `AutoModelForCausalLM` is acceptable under `transformers==5.13.0`. The official architecture is also available through `AutoModelForImageTextToText`.
