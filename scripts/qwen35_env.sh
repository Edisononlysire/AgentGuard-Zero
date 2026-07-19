#!/bin/bash
set -euo pipefail

AGZ_ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
AGZ_RESOURCE_ROOT=${AGZ_RESOURCE_ROOT:-${AGZ_ROOT}}
source "${AGZ_ROOT}/scripts/agentguard_env.sh"

AGZ_QWEN35_TRANSFORMERS_OVERLAY=${AGZ_QWEN35_TRANSFORMERS_OVERLAY:-${AGZ_RESOURCE_ROOT}/env_overlays/transformers_qwen35_latest}
AGZ_QWEN35_FLA_OVERLAY=${AGZ_QWEN35_FLA_OVERLAY:-${AGZ_RESOURCE_ROOT}/env_overlays/fla_core_py312}
AGZ_QWEN35_CAUSAL_CONV_OVERLAY=${AGZ_QWEN35_CAUSAL_CONV_OVERLAY:-${AGZ_RESOURCE_ROOT}/env_overlays/causal_conv1d_v1_6_2_py312_torch26_sm80}
if [[ ! -d "${AGZ_QWEN35_TRANSFORMERS_OVERLAY}" ]]; then
  echo "Qwen3.5 Transformers overlay not found: ${AGZ_QWEN35_TRANSFORMERS_OVERLAY}" >&2
  exit 64
fi
if [[ ! -d "${AGZ_QWEN35_FLA_OVERLAY}" ]]; then
  echo "Qwen3.5 FLA overlay not found: ${AGZ_QWEN35_FLA_OVERLAY}" >&2
  exit 65
fi
if [[ ! -d "${AGZ_QWEN35_CAUSAL_CONV_OVERLAY}" ]]; then
  echo "Qwen3.5 causal-conv1d overlay not found: ${AGZ_QWEN35_CAUSAL_CONV_OVERLAY}" >&2
  exit 66
fi

export PYTHONPATH="${AGZ_QWEN35_CAUSAL_CONV_OVERLAY}:${AGZ_QWEN35_FLA_OVERLAY}:${AGZ_QWEN35_TRANSFORMERS_OVERLAY}:${PYTHONPATH:-}"
export AGZ_QWEN35_4B_PATH=${AGZ_QWEN35_4B_PATH:-${AGZ_RESOURCE_ROOT}/models/qwen3_5/Qwen3.5-4B}
export AGZ_QWEN35_9B_PATH=${AGZ_QWEN35_9B_PATH:-${AGZ_RESOURCE_ROOT}/models/qwen3_5/Qwen3.5-9B}
export AGZ_QWEN35_TEXT_MODEL_CLASS=${AGZ_QWEN35_TEXT_MODEL_CLASS:-auto}
export VERL_NCCL_TIMEOUT_SECONDS=${VERL_NCCL_TIMEOUT_SECONDS:-7200}
export AGZ_TRITON_CACHE_ROOT=${AGZ_TRITON_CACHE_ROOT:-${TMPDIR:-/tmp}/agentguard_zero_triton_${USER:-user}}
mkdir -p "${AGZ_TRITON_CACHE_ROOT}"
# Multi-rank entrypoints must derive a rank-local TRITON_CACHE_DIR from this
# root after LOCAL_RANK is assigned. A shared Triton metadata directory races.

hash -r
echo "Using Qwen3.5 Transformers overlay: ${AGZ_QWEN35_TRANSFORMERS_OVERLAY}"
echo "Using Qwen3.5 FLA overlay: ${AGZ_QWEN35_FLA_OVERLAY}"
echo "Using Qwen3.5 causal-conv1d overlay: ${AGZ_QWEN35_CAUSAL_CONV_OVERLAY}"
echo "Using node-local Triton cache root: ${AGZ_TRITON_CACHE_ROOT}"
echo "Using VerL NCCL timeout: ${VERL_NCCL_TIMEOUT_SECONDS}s"
echo "Qwen3.5 4B path: ${AGZ_QWEN35_4B_PATH}"
echo "Qwen3.5 9B path: ${AGZ_QWEN35_9B_PATH}"
