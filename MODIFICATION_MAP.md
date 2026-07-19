# Modification map

1. Keep VeRL/GRPO/ADPO infrastructure unchanged at first.
2. Replace math task generation with `curriculum_train/scenario_generate/scenario_generate.py`.
3. Replace math self-consistency evaluation with `curriculum_train/scenario_evaluate/evaluate.py`.
4. Replace frontier data upload with `curriculum_train/scenario_evaluate/upload.py`.
5. Replace `curriculum_reward.py` with `examples/reward_function/dca_reward.py` for DCA training.
6. Replace Python sandbox tool server with `vllm_service_init/start_vllm_server_cyber.py`.
7. Train VDA on scenario trajectories, not boxed math answers.

Core conceptual replacements:

```text
question -> scenario_spec
answer -> oracle outcome
boxed answer accuracy -> trajectory reward
self-consistency -> safe_success_rate / rollout disagreement
tool calls -> cyber verification tools
```
