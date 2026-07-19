#!/bin/bash
set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
source "${ROOT}/scripts/agentguard_env.sh"

export PYTHONPATH="${ROOT}/executor_train:${ROOT}/executor_train/verl:${ROOT}:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/home/share/huadjyin/home/s_qinhua2/.cache/huggingface}

cd "${ROOT}"
mkdir -p data/smoke logs outputs/checkpoints

echo "[1/5] Import preflight"
python -s scripts/agentguard_import_preflight.py

echo "[2/5] VDA reward smoke"
python -s scripts/agentguard_vda_reward_smoke.py

echo "[3/5] Build smoke scenario input"
python -s - <<'PY'
import json
from agentguard_zero.schemas.scenario_schema import minimal_example
with open("data/smoke/minimal_scenarios.json", "w", encoding="utf-8") as f:
    json.dump([{"scenario": minimal_example()}], f, ensure_ascii=False, indent=2)
PY

echo "[4/5] Build VDA warmup parquet"
python -s curriculum_train/scenario_evaluate/build_vda_dataset.py \
  data/smoke/minimal_scenarios.json \
  --output data/smoke/vda_train.parquet

python -s - <<'PY'
import pandas as pd
path = "data/smoke/vda_train.parquet"
df = pd.read_parquet(path)
assert len(df) >= 1, "empty VDA warmup dataset"
for col in ["data_source", "problem", "reward_model", "scenario", "extra_info"]:
    assert col in df.columns, f"missing column: {col}"
print("ready", path, "rows", len(df), "columns", list(df.columns))
PY

echo "[5/5] Hydra PPO config smoke"
python -s scripts/agentguard_hydra_config_smoke.py

echo "Pre-submit checks passed."
echo "Submit with: dsub -s ${ROOT}/scripts/train_vda_warmup_dsub.sh"
