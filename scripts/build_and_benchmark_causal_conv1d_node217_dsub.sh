#!/bin/bash
#DSUB -n AGZCausalConvBench
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -R "cpu=64;gpu=4;mem=230000"
#DSUB -oo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs/%J.err

set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
EXPECTED_NODE=${AGZ_EXPECTED_NODE:-cyclone001-agent-217}
SOURCE_DIR=${ROOT}/.build/causal-conv1d-v1.6.2.post1
OVERLAY=${ROOT}/env_overlays/causal_conv1d_v1_6_2_py312_torch26_sm80
RESULT=${ROOT}/outputs/perf/qwen35_4b_causal_conv1d_benchmark.json

if [[ "$(hostname)" != "${EXPECTED_NODE}" ]]; then
  echo "Refusing to run outside ${EXPECTED_NODE}: $(hostname)" >&2
  exit 72
fi
if [[ ! -f "${SOURCE_DIR}/setup.py" ]]; then
  echo "Missing causal-conv1d source: ${SOURCE_DIR}" >&2
  exit 66
fi

source "${ROOT}/scripts/qwen35_env.sh"
mkdir -p "${OVERLAY}" "$(dirname "${RESULT}")"

export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_CXX11_ABI=FALSE
export MAX_JOBS=${MAX_JOBS:-8}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
export TMPDIR=${TMPDIR:-/tmp}/agz_causal_conv1d_${USER}_${BASHPID}
mkdir -p "${TMPDIR}"

python -s -m pip install \
  --no-build-isolation \
  --no-deps \
  --target "${OVERLAY}" \
  "${SOURCE_DIR}"

export PYTHONPATH="${OVERLAY}:${PYTHONPATH:-}"
python -s -c 'from causal_conv1d import causal_conv1d_fn, causal_conv1d_update; print("causal_conv1d import passed")'
python -s -c 'from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available; assert is_fast_path_available; print("Qwen3.5 fast path enabled")'

torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/benchmark_qwen35_fast_generation.py" \
  --model-path "${AGZ_QWEN35_4B_PATH}" \
  --output "${RESULT}" \
  --batch-size 8 \
  --prompt-tokens 750 \
  --new-tokens 128

echo "Benchmark complete: ${RESULT}"
