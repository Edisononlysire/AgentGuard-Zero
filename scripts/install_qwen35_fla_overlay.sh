#!/bin/bash
set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
source "${ROOT}/scripts/agentguard_env.sh"

VERSION=${AGZ_FLA_CORE_VERSION:-0.5.1}
TARGET=${AGZ_QWEN35_FLA_OVERLAY:-${ROOT}/env_overlays/fla_core_py312}
STAGING="${TARGET}.staging.$$"
trap 'rm -rf "${STAGING}"' EXIT

rm -rf "${STAGING}"
python -m pip install --no-deps --target "${STAGING}" "fla-core==${VERSION}"
PYTHONPATH="${STAGING}:${PYTHONPATH:-}" python - <<'PY'
import fla
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

assert FusedRMSNormGated is not None
assert chunk_gated_delta_rule is not None
assert fused_recurrent_gated_delta_rule is not None
print(f"validated fla-core {fla.__version__}")
PY

rm -rf "${TARGET}"
mv "${STAGING}" "${TARGET}"
trap - EXIT
echo "Installed Qwen3.5 FLA overlay at ${TARGET}"
