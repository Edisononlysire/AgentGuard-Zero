#!/bin/bash
set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
mkdir -p "${ROOT}/models/cyber_llm" "${ROOT}/logs"
export AGZ_ROOT="${ROOT}"
source "${ROOT}/scripts/cyber_llm_env.sh"

python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
import json
import os

repo = os.environ.get("AGZ_CYBER_LLM_REPO", "fdtn-ai/Foundation-Sec-8B-Instruct")
local_dir = Path(os.environ["AGZ_CYBER_LLM_MODEL_PATH"]).resolve()
local_dir.mkdir(parents=True, exist_ok=True)
print(json.dumps({"repo": repo, "local_dir": str(local_dir)}, indent=2))
snapshot_download(
    repo_id=repo,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.model",
        "*.txt",
        "*.md",
        "tokenizer*",
        "special_tokens_map.json",
        "generation_config.json",
        "*.py",
    ],
)
print("download_complete")
PY

python - <<'PY'
from pathlib import Path
import json
import os

path = Path(os.environ["AGZ_CYBER_LLM_MODEL_PATH"])
files = sorted(p.name for p in path.glob("*"))
shards = sorted(path.glob("*.safetensors"))
incomplete = sorted(path.glob("*.incomplete")) + sorted(path.glob("*.lock"))
summary = {
    "path": str(path),
    "exists": path.exists(),
    "file_count": len(files),
    "safetensors_shards": len(shards),
    "safetensors_bytes": sum(p.stat().st_size for p in shards),
    "incomplete_files": [p.name for p in incomplete],
    "has_config": (path / "config.json").exists(),
}
print(json.dumps(summary, indent=2))
if not summary["has_config"] or not shards or incomplete:
    raise SystemExit("Foundation-Sec download/integrity check is incomplete.")
PY
