# AgentGuard-Zero modifications

This repository is a framework-level modification of Agent0 for:

**Zero-human-data co-evolution for profile-poisoning-resilient LLM cyber defense agents.**

The original Agent0 mapping is:

| Agent0 | AgentGuard-Zero |
|---|---|
| Curriculum Agent generates math questions | DCA generates abstract profile-poisoning scenarios |
| Executor Agent solves math questions | VDA performs active verification and safe response |
| Python code interpreter | Abstract cyber verification tools |
| Majority-vote pseudo-label | Simulator hidden-state oracle and trajectory reward |
| Self-consistency difficulty band | Frontier safe-success-rate band |

## New core directories

```text
Agent0/agentguard_zero/
  env/       abstract cyber environment, checker, oracle
  tools/     LogQuery, CrossCheck, ProvenanceCheck, GraphQuery, BusinessImpactEstimator
  memory/    profile quarantine / confirmed / rejected memory
  schemas/   scenario and VDA action schemas
  rewards/   DCA and VDA reward helpers
  prompts/   DCA/VDA prompts
```

## New curriculum pipeline

```text
Agent0/curriculum_train/scenario_generate/scenario_generate.py
Agent0/curriculum_train/scenario_evaluate/evaluate.py
Agent0/curriculum_train/scenario_evaluate/upload.py
Agent0/curriculum_train/examples/reward_function/dca_reward.py
Agent0/curriculum_train/vllm_service_init/start_vllm_server_cyber.py
```

## Smoke test without model training

```bash
cd Agent0
export PYTHONPATH=$PWD:$PYTHONPATH
python - <<'PY'
from agentguard_zero.schemas.scenario_schema import minimal_example
from curriculum_train.scenario_evaluate.evaluate import evaluate_scenario
print(evaluate_scenario(minimal_example(), num_rollouts=1)['safe_success_rate'])
PY
```

## What is complete in this patch

This patch builds the framework interfaces: scenario schema, checker, environment, tools, memory, oracle, DCA reward, VDA reward, scenario generation/evaluation/upload scripts, and a cyber tool-server adapter.

## What still needs cluster-side wiring

The VeRL executor trainer is intentionally not deeply rewritten in this first patch. The next step is to connect `frontier_scenarios/*.parquet` to `executor_train/verl_tool` and replace math boxed-answer reward with `agentguard_zero.rewards.vda_reward`.
