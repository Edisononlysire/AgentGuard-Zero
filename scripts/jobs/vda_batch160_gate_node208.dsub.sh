#!/bin/bash
#DSUB -n AGZ4BVDABatch160Gate
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail
ROOT=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
FORMAL=/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
EXPECTED_NODE=cyclone001-agent-208
if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi

export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/qwen35_env.sh"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

OUT="${ROOT}/outputs/optimization_gates/vda-delta-wire-v3-batch160-gate"
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite existing gate: ${OUT}" >&2
  exit 73
fi
mkdir -p "${OUT}"

FEEDBACK="${FORMAL}/data/tmcd_v2/qwen3.5-4b/round_1/dca_feedback/feedback.jsonl"
TRAIN="${OUT}/train.parquet"
python -s - "${FEEDBACK}" "${TRAIN}" <<'PY'
import json
import pathlib
import sys

import pandas as pd

from agentguard_zero.training.vda_dataset import scenario_to_training_row

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
scenarios = []
seen = set()
with source.open(encoding="utf-8") as handle:
    for line in handle:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        evaluation = record.get("vda_evaluation", {}) or {}
        scenario = record.get("scenario", {}) or {}
        scenario_id = str(scenario.get("scenario_id", ""))
        if (
            record.get("parse_ok")
            and evaluation.get("all_ok")
            and evaluation.get("oracle_solvable")
            and scenario.get("protocol_version") == "tmcd-v2"
            and scenario_id
            and scenario_id not in seen
        ):
            seen.add(scenario_id)
            scenarios.append(scenario)
        if len(scenarios) == 160:
            break
if len(scenarios) < 160:
    raise SystemExit(f"need 160 unique valid TMCD-v2 scenarios, found {len(scenarios)}")
rows = [scenario_to_training_row(scenario, split="train") for scenario in scenarios]
pd.DataFrame(rows).to_parquet(target, index=False)
print(json.dumps({"rows": len(rows), "output": str(target)}, indent=2))
PY

export AGZ_MODEL_PATH="${AGZ_QWEN35_4B_PATH}"
export AGZ_TRAIN_FILE="${TRAIN}"
export AGZ_VAL_FILE="${TRAIN}"
export AGZ_RUN_NAME=agz_gate_qwen3.5-4b_vda_batch160
export AGZ_CHECKPOINT_DIR="${OUT}/checkpoints"
export AGZ_MAX_STEPS=1
export AGZ_RESUME_MODE=disable
export AGZ_CUDA_VISIBLE_DEVICES=0,1,2,3
export AGZ_N_GPUS_PER_NODE=4
export AGZ_BATCH_SIZE=160
export AGZ_PPO_MINI_BATCH_SIZE=40
export AGZ_ROLLOUT_N=1
export AGZ_ADV_ESTIMATOR=reinforce_plus_plus
export AGZ_TOOL_SERVER_MODE=level1
export AGZ_BUILD_SMOKE_DATASET=0
export AGZ_AGENT_MAX_TURNS=16
export AGZ_MAX_PROMPT_LENGTH=2048
export AGZ_MAX_RESPONSE_LENGTH=11264
export AGZ_MAX_MODEL_LENGTH=15360
export AGZ_MAX_ACTION_LENGTH=320
export AGZ_MAX_OBS_LENGTH=1280
export AGZ_GPU_MEMORY_UTILIZATION=0.35
export AGZ_MAX_NUM_SEQS=40
export AGZ_LORA_RANK=16
export AGZ_LORA_ALPHA=32
export AGZ_LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
export AGZ_ACTOR_LR=2e-5
export AGZ_ACTOR_CPU_OFFLOAD=false
export AGZ_ACTOR_PARAM_OFFLOAD=false
export AGZ_ACTOR_OPTIMIZER_OFFLOAD=false
export AGZ_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export AGZ_REF_PARAM_OFFLOAD=false
export AGZ_RESHARD_AFTER_FORWARD=true
export AGZ_SEED=20260709
export AGZ_VAL_BEFORE_TRAIN=False
export AGZ_DATA_SHUFFLE=false
export AGZ_SAVE_FREQ=1
export AGZ_TEST_FREQ=0
export AGZ_STOP_ON_COMPLETE_JSON=true

bash "${ROOT}/scripts/train_vda_qwen35_lora.sh"
