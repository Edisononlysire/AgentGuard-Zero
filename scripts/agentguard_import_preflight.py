from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "executor_train", ROOT / "executor_train" / "verl", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import vllm
    VLLM_STATUS = f"ok: {vllm.__version__}"
except Exception as exc:  # vLLM is optional for HF rollout.
    VLLM_STATUS = f"optional_import_failed: {type(exc).__name__}: {exc}"
import flash_attn
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

from verl import DataProto
import verl_tool.workers.reward_manager  # noqa: F401
from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY


def main() -> None:
    print("numpy", np.__version__, np.__file__)
    print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
    print("torch_cuda", torch.version.cuda, torch.cuda.get_arch_list())
    print("vllm", VLLM_STATUS)
    print("vllm_use_v1", os.environ.get("VLLM_USE_V1"))
    print("flash_attn", flash_attn.__version__)
    print("flash_attn_bert_padding_ok", all([index_first_axis, pad_input, rearrange, unpad_input]))
    print("verl_ok", DataProto is not None)
    print("reward_managers", sorted(REWARD_MANAGER_REGISTRY.keys()))


if __name__ == "__main__":
    main()
