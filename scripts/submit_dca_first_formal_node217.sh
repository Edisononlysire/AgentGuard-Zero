#!/bin/bash
set -euo pipefail

ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
NODE=${AGZ_FORMAL_NODE:-cyclone001-agent-217}
cd "${ROOT}"

extract_job_id() {
  awk '$1 ~ /^[0-9]+$/ {print $1; exit}'
}

echo "Submitting formal 4B three-round job to ${NODE}"
four_output=$(dsub -pn "${NODE}" -s scripts/train_dca_first_round_4b_node217_dsub.sh)
printf '%s\n' "${four_output}"
four_job=$(printf '%s\n' "${four_output}" | extract_job_id)
if [[ -z "${four_job}" ]]; then
  echo "Could not parse the 4B formal job id" >&2
  exit 70
fi

echo "Submitting formal 9B three-round job with dependency ${four_job}=SUCCEEDED"
nine_output=$(dsub -D "${four_job}=SUCCEEDED" -pn "${NODE}" \
  -s scripts/train_dca_first_round_9b_node217_dsub.sh)
printf '%s\n' "${nine_output}"
nine_job=$(printf '%s\n' "${nine_output}" | extract_job_id)
if [[ -z "${nine_job}" ]]; then
  echo "Could not parse the 9B formal job id" >&2
  exit 71
fi

echo "formal_4b_job=${four_job}"
echo "formal_9b_job=${nine_job} dependency=${four_job}=SUCCEEDED"
