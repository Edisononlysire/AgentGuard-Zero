#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export AGZ_ROOT=${AGZ_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
export PYTHONPATH="${AGZ_ROOT}:${AGZ_ROOT}/third_party:${AGZ_ROOT}/third_party/verl:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export AGZ_TRITON_CACHE_ROOT=${AGZ_TRITON_CACHE_ROOT:-${TMPDIR:-/tmp}/agentguard_zero_triton_${USER:-user}}

# Optional target-directory installs for platform-specific Qwen3.5 packages.
# Prepending in this order yields causal-conv -> FLA -> Transformers precedence.
for variable_name in AGZ_TRANSFORMERS_OVERLAY AGZ_FLA_OVERLAY AGZ_CAUSAL_CONV_OVERLAY; do
  overlay_path=${!variable_name-}
  if [[ -n "${overlay_path}" ]]; then
    if [[ ! -d "${overlay_path}" ]]; then
      echo "${variable_name} does not exist: ${overlay_path}" >&2
      exit 64
    fi
    export PYTHONPATH="${overlay_path}:${PYTHONPATH}"
  fi
done

mkdir -p "${HF_HOME}" "${AGZ_TRITON_CACHE_ROOT}" "${AGZ_ROOT}/logs"
