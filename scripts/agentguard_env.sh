#!/bin/bash
set -euo pipefail

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
module load tools/ccache/4.10.2_gcc11.3
module load compilers/cuda/12.8.0
module load libs/cudnn/9.8.0_cuda12
module load libs/nccl/2.27.3-1_cuda12.8
module load libs/cuDSS/0.5.0
module load libs/cuSPARSELt/0.7.1
module load tools/cmake/4.1.0
module load libs/openblas/0.3.31

CONDA_HOME=${AGZ_CONDA_HOME:-/home/share/huadjyin/home/s_qinhua2/01software/miniconda3}
CONDA_ENV_NAME=${AGZ_CONDA_ENV:-agent0-gpu}
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONNOUSERSITE=1
AGZ_ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
AGZ_RESOURCE_ROOT=${AGZ_RESOURCE_ROOT:-${AGZ_ROOT}}
AGZ_ENABLE_SM80_OVERLAY=${AGZ_ENABLE_SM80_OVERLAY:-1}
AGZ_TORCH_SM80_OVERLAY=${AGZ_TORCH_SM80_OVERLAY:-${AGZ_RESOURCE_ROOT}/env_overlays/torch_sm80_py312}
AGZ_TORCH_SM80_LIB=${AGZ_TORCH_SM80_LIB:-/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/foundry-rfd3/lib}
if [[ "${AGZ_ENABLE_SM80_OVERLAY}" == "1" && -d "${AGZ_TORCH_SM80_OVERLAY}" ]]; then
  export PYTHONPATH="${AGZ_TORCH_SM80_OVERLAY}:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${AGZ_TORCH_SM80_LIB}:${LD_LIBRARY_PATH:-}"
  export AGZ_ACTIVE_TORCH_OVERLAY="${AGZ_TORCH_SM80_OVERLAY}"
fi

SKLEARN_GOMP_DIR="${CONDA_PREFIX}/lib/python3.12/site-packages/scikit_learn.libs"
SKLEARN_GOMP=""
if [[ -d "${SKLEARN_GOMP_DIR}" ]]; then
  SKLEARN_GOMP=$(find "${SKLEARN_GOMP_DIR}" -name "libgomp*.so*" | head -n 1)
fi
if [[ -n "${SKLEARN_GOMP}" ]]; then
  export LD_PRELOAD="${SKLEARN_GOMP}:${LD_PRELOAD:-}"
fi

unset PYTHONHOME

export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=60
export NCCL_IB_RETRY_CNT=10
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export VLLM_DISABLE_COMPILE_CACHE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export NO_PROXY=0.0.0.0,127.0.0.1,localhost
export RAY_memory_usage_threshold=0.99
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export AGZ_DISABLE_TORCH_COMPILE=${AGZ_DISABLE_TORCH_COMPILE:-1}

hash -r
export AGZ_ROOT AGZ_RESOURCE_ROOT
echo "Using conda env: ${CONDA_DEFAULT_ENV} (${CONDA_PREFIX})"
echo "Using project root: ${AGZ_ROOT}"
echo "Using external resource root: ${AGZ_RESOURCE_ROOT}"
if [[ -n "${AGZ_ACTIVE_TORCH_OVERLAY:-}" ]]; then
  echo "Using torch sm80 overlay: ${AGZ_ACTIVE_TORCH_OVERLAY}"
fi
