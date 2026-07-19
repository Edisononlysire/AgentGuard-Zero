#!/bin/bash
set -euo pipefail

AGZ_ROOT=${AGZ_ROOT:-/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero}
source "${AGZ_ROOT}/scripts/qwen35_env.sh"

export AGZ_CYBER_LLM_REPO=${AGZ_CYBER_LLM_REPO:-fdtn-ai/Foundation-Sec-8B-Instruct}
export AGZ_CYBER_LLM_MODEL_PATH=${AGZ_CYBER_LLM_MODEL_PATH:-${AGZ_ROOT}/models/cyber_llm/Foundation-Sec-8B-Instruct}
export AGZ_LILY_CYBER_REPO=${AGZ_LILY_CYBER_REPO:-segolilylabs/Lily-Cybersecurity-7B-v0.2}
export AGZ_LILY_CYBER_MODEL_PATH=${AGZ_LILY_CYBER_MODEL_PATH:-${AGZ_ROOT}/models/cyber_llm/Lily-Cybersecurity-7B-v0.2}

echo "Cyber LLM repo: ${AGZ_CYBER_LLM_REPO}"
echo "Cyber LLM path: ${AGZ_CYBER_LLM_MODEL_PATH}"
echo "Lily Cyber repo: ${AGZ_LILY_CYBER_REPO}"
echo "Lily Cyber path: ${AGZ_LILY_CYBER_MODEL_PATH}"
