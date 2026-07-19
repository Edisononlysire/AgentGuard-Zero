# AgentGuard-Zero Handoff, 2026-07-09

This document is a handoff for continuing AgentGuard-Zero in a new Codex conversation. It records the current code state, server paths, experiment status, model versions, metrics, and next commands. The most important point: **AgentGuard-Zero-Select V5-C has been explored and evaluated as an inference-time selector, but it has not trained model parameters; AgentGuard-Zero-Train LoRA has not produced a final result yet.**

## 1. Current One-Sentence Project Definition

AgentGuard-Zero is an **AI for cyber defense** framework. It studies whether an LLM-based Verification Defense Agent (VDA) can make safer multi-step cyber defense decisions under trust deception, profile poisoning, objective/strategy switching, and overreaction induction. The project avoids real exploit payloads, malware, public-network attack, or Level-3 cyber range emulation. The main experimental environment is a controlled Level-1 simulator with hidden attack state and real action consequences; Level-2 is a CybORG/CAGE-style observation wrapper for realism support.

## 2. Remote Server / Workspace

Remote SSH alias used in this conversation:

```bash
ssh qinhua-server
```

Remote project root on Wuchao server:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
```

Main environment:

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
source scripts/qwen35_env.sh
```

The user originally requested the environment `agent0-gpu`. The current working setup uses `scripts/qwen35_env.sh`, which activates the AgentGuard/Qwen3.5 environment and overlays a Torch build compatible with A100 `sm_80`.

Important note on nodes:

- Compute nodes to use: `208`, `175`, and possibly other real compute nodes if assigned.
- Login-like nodes mentioned earlier: `151` and `56`; do not hard-code them for GPU jobs.
- User preference: do **not** manually specify a node unless necessary; submit with `dsub -s script_path` and let scheduler place the job.

dsub usage:

```bash
dsub -s /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/scripts/<script>.sh
```

Logs:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/logs
```

Job query:

```bash
djob
```

At the end of this handoff, `djob` returned no active job listing in the command output, and no new formal GPU jobs were submitted after deployment.

## 3. Local Workspace

Local workspace in Codex desktop:

```bash
/Users/libaolong/Documents/Agent 0
```

This handoff file was created locally at:

```bash
/Users/libaolong/Documents/Agent 0/AGZ_HANDOFF_20260709.md
```

It should also be copied to the remote project docs directory:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/docs/AGZ_HANDOFF_20260709.md
```

## 4. Current Narrative Decision

The paper direction should be framed primarily as **AI for security / AI for cyber defense**, not generic Security for AI agents.

Recommended main claim:

> A zero-label DCA-VDA co-evolution framework can improve LLM cyber defense agents under trust deception, profile poisoning, and overreaction induction, while preserving business utility through active verification and safe response constraints.

Important nuance:

- We currently have **Round-0 frontier data and Select framework exploration**.
- We do **not yet have completed formal DCA-VDA co-evolution rounds Iter 1/2/3**.
- We do **not yet have final LoRA training results** for AgentGuard-Zero-Train.
- Therefore, new conversation should not claim that co-evolution training is complete.

## 5. Fixed Experimental Systems / Baselines

The fixed nine-system experimental plan is:

| System | Train Parameters? | Model | How to Run |
|---|---:|---|---|
| Rule-based SOC | No | None | Rule policy |
| ReAct / Base+Tools | No | Qwen3.5-9B | Prompt + tools |
| Memory Agent | No | Qwen3.5-9B | Prompt + ordinary memory |
| Trust-score Agent | No | Qwen3.5-9B | Prompt + simple source trust score |
| Cyber LLM VDA | No | Foundation-Sec-8B-Instruct | Same VDA schema + tools |
| Qwen Zero-shot VDA | No | Qwen3.5-9B | VDA prompt, one JSON action |
| AgentGuard-Zero-Select | No | Qwen3.5-9B or API model | K candidates + selector/safety governor |
| AgentGuard-Zero-Train | Yes | Qwen3.5-9B + LoRA | Trajectory reward fine-tuning |
| Oracle Defender | No | None | Upper-bound, uses privileged simulator info |

`Oracle Defender` is not deployable and should be described only as an upper-bound reference. It is expected to score highest because the normalized score uses Oracle as 1.0.

## 6. Select Version History

AgentGuard-Zero-Select does not train model parameters. It evolved through pilot-driven selector and safety-governor design.

| Version | Selector Mode | Main Change | Current Status |
|---|---|---|---|
| Zero-shot VDA | none | One JSON action, direct execution | Baseline |
| Select early | none / initial | K candidate actions, selector chooses one | Deprecated |
| V2 | `mitigation_v2` | Better JSON/tool legality and active verification preference | Explored |
| V3 | `mitigation_v3` | Stronger mitigation-first scoring | Explored |
| V4 | `mitigation_v4` | Added safety governor, overresponse/business-cost suppression | Very strong |
| V5-A | `v5_a_constrained` | Constrained safe utility variant | Weaker |
| V5-B | `v5_b_belief_q` | Belief-quality and quarantine emphasis | Strong |
| V5-C | `v5_c_frontier_minimax` | Frontier-minimax balance across poisoning, verification, business cost | **Use this as current Select version** |

Current recommendation:

```text
Use AgentGuard-Zero-Select V5-C as the default Select framework.
```

Why V5-C:

- It reaches the best/near-best NSU in medium API pilot.
- It preserves attack mitigation at 1.0.
- It reduces average steps versus zero-shot.
- It is conceptually strongest for the paper because it matches the frontier/minimax story.

Important wording:

> V5-C is not trained by gradient descent. It is a failure-trajectory-driven inference-time selector design. Pilot trajectories were used to refine and choose selector rules, not to update base model parameters.

## 7. DCA-VDA Co-Evolution Status

Current formal co-evolution status:

```text
Formal DCA-VDA co-evolution rounds completed: 0
Current data state: Round-0 / seed20260706 frontier pool
```

What has been done:

1. DCA-style generator produced Level-1 scenario candidates.
2. CFC/frontier filtering selected hard-but-solvable scenarios.
3. Zero-shot and Select variants were evaluated on pilot slices.
4. Select V5-C was chosen by framework search.

What has not yet been done:

```text
Iter 0 VDA failures -> DCA updates frontier -> Iter 1 data
Iter 1 VDA failures -> DCA updates frontier -> Iter 2 data
Iter 2 VDA failures -> DCA updates frontier -> Iter 3 data
```

For the paper Figure 1 / curriculum curve, we still need to create true Iter 0/1/2/3 data and run the systems consistently.

## 8. Data Files

Main Level-1 frontier parquet:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/data/level1/level1_seed20260706_n500_frontier_vda.parquet
```

Actual content:

- 256 rows, despite the `n500` name.
- `split` currently equals `train` for all rows.
- Four task groups, 64 each:
  - Evidence-Level Profile Poisoning
  - Long-Horizon Trust-Building Betrayal
  - Objective / Strategy Switching
  - Business-Constrained Overreaction Induction

Generation process:

```text
DCA-style scenario generator produced about 500 abstract Level-1 cyber defense scenarios
-> Level-1 simulator/CFC checks hidden state validity, tool consequences, oracle solvability, safety
-> CFC/frontier filtering selected 256 hard-but-solvable scenarios
-> output frontier_vda.parquet
```

Related files:

```bash
data/level1/scenarios/level1_seed20260706_n500_candidates.json
data/level1/scenarios/level1_seed20260706_n500_frontier.json
data/level1/level1_seed20260706_n500_cfc_report.json
data/level1/level1_hard_frontier_top16.parquet
```

Level-2 wrapper data:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/data/level2/cage_style_t3_t4_vda.parquet
```

Level-2 content:

- 128 rows.
- Built from Level-1 frontier subset.
- Intended for CAGE/CybORG-style observation wrapper realism.
- Covers mainly:
  - Evidence-Level Profile Poisoning
  - Business-Constrained Overreaction Induction

Current caveat:

> The current Level-1 frontier parquet is good for pilot and training pool, but final AAAI main tables should use a cleaner held-out test split or at least a documented held-out subset with identical N across all systems.

Recommended final data split:

```text
train_frontier.parquet
dev_frontier.parquet
test_heldout_frontier.parquet
```

## 9. Models Deployed

Qwen3.5 paths:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/models/qwen3_5/Qwen3.5-4B
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/models/qwen3_5/Qwen3.5-9B
```

Cyber LLM baseline path:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/models/cyber_llm/Foundation-Sec-8B-Instruct
```

Foundation-Sec deployment status:

- Fully synchronized to server.
- Verified in `agent0-gpu` / `qwen35_env.sh`.
- `AutoConfig` loads as `LlamaConfig`.
- Tokenizer loads, vocab size 128000.
- Four safetensors headers open successfully.

Foundation-Sec shard SHA256 values verified on server:

```text
model-00001-of-00004.safetensors b17b7fbfce717c099424d22f40bbe9f2a2a79594896557eb3f721c148e8c2275
model-00002-of-00004.safetensors cf634cf21c5fa6aca11a6b6cc65cc45e0f42acd35ff1ba79e3f0bea7f652eade
model-00003-of-00004.safetensors 63a9f44a831147ebdb3beb115c7dde4cccb7f63c2889f795732c2d230d5b8c1f
model-00004-of-00004.safetensors 3ca6689d7b50dc6e6d58ecdb4532ec8e433057899b93bb910763d33151c8735e
```

Environment note:

- Preflight shows Torch CUDA supports `sm_80`, suitable for A100.
- vLLM optional import still reports a Transformers `aimv2` config conflict, but it is not blocking because Qwen3.5 LoRA wrapper uses HF rollout by default.
- `torchvision/libjpeg` warning appears, but it is irrelevant for text-only model training/evaluation.

## 10. Important Scripts

Core final runner:

```bash
scripts/eval_tmcd_systems.py
```

This supports the nine-system plan:

```text
rule_based_soc
react_base_tools
memory_agent
trust_score_agent
cyber_llm_vda
qwen_zero_shot_vda
agentguard_zero_select
agentguard_zero_train
oracle_defender
```

Legacy/API Select exploration runner:

```bash
scripts/eval_level1_select.py
```

Level-1 rollout server:

```bash
scripts/level1_rollout_server.py
```

Level-2 data builder:

```bash
scripts/build_cage_style_level2.py
```

Table export:

```bash
scripts/export_tmcd_tables.py
```

Environment scripts:

```bash
scripts/qwen35_env.sh
scripts/cyber_llm_env.sh
```

Preflight/smoke:

```bash
scripts/agentguard_import_preflight.py
scripts/agentguard_reward_routing_smoke.py
scripts/smoke_tmcd_cpu.sh
```

dsub scripts:

```bash
scripts/eval_tmcd_system_dsub.sh
scripts/eval_tmcd_static_cpu_dsub.sh
scripts/eval_tmcd_api_cpu_dsub.sh
scripts/train_agentguard_zero_train_qwen35_9b_dsub.sh
scripts/train_vda_qwen35_lora_dsub.sh
```

Main runbook:

```bash
docs/TMCD_EXPERIMENT_RUNBOOK_20260708.md
```

Other docs:

```bash
docs/API_SELECT_OPTIMIZATION_20260707.md
docs/FRAMEWORK_SELECT.md
docs/QWEN35_DEPLOYMENT_20260708.md
```

## 11. Metrics

Raw Safe Utility formula:

```text
Raw SU =
0.35 * AttackMitigation
+ 0.20 * IntentF1
- 0.20 * BusinessCost
- 0.15 * Overresponse
- 0.05 * VerificationCost
- 0.05 * Delay
```

Because positive terms sum to 0.55, raw SU naturally lives around 0.3-0.5 for good policies. To make the main score near 0-1, use normalized Safe Utility:

```text
NSU = (RawSU - RandomSU) / (OracleSU - RandomSU)
```

Current Level-1 normalization anchors:

```text
Random Policy Raw SU = 0.155840
Oracle Defender Raw SU = 0.504677
```

Therefore:

```text
Random Policy NSU = 0
Oracle Defender NSU = 1
```

Important: Attack Mitigation is already a 0-1 rate and should not be normalized using this formula.

## 12. Current Measured Results

Clean normalized result table is saved at:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/outputs/pilot_tables/normalized_all_measured_clean_20260709.md
```

Key current results:

| Method/System | Model | N | Raw SU | NSU | Mitigation |
|---|---|---:|---:|---:|---:|
| Oracle Defender | static | 256 | 0.504677 | 1.000 | 1.000 |
| Random Policy | static | 256 | 0.155840 | 0.000 | 0.535 |
| Rule-based SOC | static | 256 | 0.462407 | 0.879 | 0.973 |
| Zero-shot VDA | glm-5.1 API | 96 | 0.425764 | 0.774 | 0.823 |
| AgentGuard-Zero-Select V4 | glm-5.1 API | 96 | 0.496651 | 0.977 | 1.000 |
| AgentGuard-Zero-Select V5-C | glm-5.1 API | 96 | 0.496707 | 0.977 | 1.000 |
| AgentGuard-Zero-Select V5-A | glm-5.1 API | 48 | 0.420705 | 0.759 | 0.771 |
| AgentGuard-Zero-Select V5-B | glm-5.1 API | 48 | 0.494330 | 0.970 | 1.000 |
| AgentGuard-Zero-Select V5-C | glm-5.1 API | 48 | 0.494430 | 0.971 | 1.000 |

Most important current comparison:

```text
Zero-shot VDA, N=96:
Raw SU = 0.425764
NSU = 0.774
Attack Mitigation = 0.823

AgentGuard-Zero-Select V5-C, N=96:
Raw SU = 0.496707
NSU = 0.977
Attack Mitigation = 1.000

NSU improvement = +0.203, about +20.3 points
Mitigation improvement = +0.177, about +17.7 points
```

Critical caveat:

> These results mix CPU full-frontier baselines and API pilot/medium evaluations with different N. They are useful for direction and framework selection, but they are not yet the final fair AAAI main table. Final main table must use the same held-out test set and same N across methods.

## 13. What Is Already Verified

Completed:

- Level-1 controlled simulator and VDA JSON action loop are in place.
- Level-1 frontier parquet exists.
- Level-2 CAGE-style wrapper data exists.
- `eval_tmcd_systems.py` supports the fixed nine-system plan.
- Static CPU baselines have been run:
  - Rule-based SOC
  - Random Policy
  - Oracle Defender
  - Level-2 Rule-based SOC
- API Select exploration has been run for GLM-5.1:
  - Zero-shot
  - Select V2/V3/V4/V5-A/V5-B/V5-C
- V5-C chosen as current Select framework.
- Foundation-Sec-8B-Instruct downloaded locally, synced to server, and verified.
- Qwen3.5-4B and Qwen3.5-9B model directories exist on server.
- `agentguard_reward_routing_smoke.py` passed, confirming training reward routing prefers Level-1 trajectory reward over single-step proxy when available.
- Shell syntax and Python compile checks passed for the important scripts.

Reward routing smoke output:

```json
{"ok": true, "sequence_scores": [4.25, -1.5, 0.75], "extra": {"level1_trajectory_reward_available": [1.0, 1.0, 0.0], "level1_trajectory_reward_source": [1.0, 2.0, 0.0], "single_step_reward_fallback": [0.0, 0.0, 1.0]}}
```

## 14. What Is Not Yet Done

Not completed:

- No final Qwen3.5-9B LoRA AgentGuard-Zero-Train result yet.
- No final Qwen3.5-4B Train result yet.
- No fair all-baseline table on the same held-out N.
- No true DCA-VDA co-evolution Iter 1/2/3 yet.
- No final Table 1/2/3 with confidence intervals.
- No complete Level-2 model baseline table yet.
- No external Cyber LLM baseline full evaluation yet, although Foundation-Sec is deployed and ready.

## 15. Recommended Immediate Next Steps

### Step A: Run Qwen3.5-9B LoRA small training

Use this first when GPU is available:

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
AGZ_MAX_STEPS=50 \
AGZ_ROLLOUT_N=2 \
AGZ_BATCH_SIZE=2 \
AGZ_PPO_MINI_BATCH_SIZE=2 \
dsub -s scripts/train_agentguard_zero_train_qwen35_9b_dsub.sh
```

This is the key missing piece for "we must train".

### Step B: Evaluate model systems on the same data/N

After or in parallel depending on GPUs:

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
for S in react_base_tools memory_agent trust_score_agent qwen_zero_shot_vda agentguard_zero_select cyber_llm_vda; do
  AGZ_SYSTEM=$S AGZ_EVAL_LIMIT=256 \
  dsub -s scripts/eval_tmcd_system_dsub.sh
done
```

### Step C: Evaluate trained adapter after LoRA finishes

Replace `/path/to/lora_adapter` with actual checkpoint adapter path:

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
AGZ_SYSTEM=agentguard_zero_train \
AGZ_ADAPTER_PATH=/path/to/lora_adapter \
AGZ_EVAL_LIMIT=256 \
dsub -s scripts/eval_tmcd_system_dsub.sh
```

### Step D: Export paper tables

After all evals finish:

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
source scripts/qwen35_env.sh
python -s scripts/export_tmcd_tables.py \
  --input_dir outputs/tmcd_eval \
  --output_dir outputs/paper_tables_final
```

### Step E: Build true held-out split

Before final AAAI table, create or identify held-out test data:

```text
data/level1/train_frontier.parquet
data/level1/dev_frontier.parquet
data/level1/test_heldout_frontier.parquet
```

The current `level1_seed20260706_n500_frontier_vda.parquet` has `split=train` for all 256 rows and should be treated as a frontier pool / pilot pool unless re-split carefully.

## 16. Suggested Final Main Table Design

Final Table 1 should not mix N. Use a single test set, probably N=256 or held-out N=128/256, and report:

| System | Model | NSU | Raw SU | Attack Mitigation | Business Cost | Overresponse | JSON Fail | Invalid Tool |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Random Policy | none | anchor 0 | raw | rate | cost | rate | 0 | 0 |
| Rule-based SOC | none | value | raw | rate | cost | rate | 0 | 0 |
| ReAct / Base+Tools | Qwen3.5-9B | value | raw | rate | cost | rate | rate | rate |
| Memory Agent | Qwen3.5-9B | value | raw | rate | cost | rate | rate | rate |
| Trust-score Agent | Qwen3.5-9B | value | raw | rate | cost | rate | rate | rate |
| Cyber LLM VDA | Foundation-Sec-8B | value | raw | rate | cost | rate | rate | rate |
| Qwen Zero-shot VDA | Qwen3.5-9B | value | raw | rate | cost | rate | rate | rate |
| AgentGuard-Zero-Select | Qwen3.5-9B | value | raw | rate | cost | rate | rate | rate |
| AgentGuard-Zero-Train | Qwen3.5-9B LoRA | value | raw | rate | cost | rate | rate | rate |
| Oracle Defender | none | anchor 1 | raw | rate | cost | rate | 0 | 0 |

## 17. How To Explain Select vs Train

AgentGuard-Zero-Select:

```text
No parameter training.
Same base model.
At each step generate K JSON action candidates.
Selector / safety governor scores candidates using public observation, history, candidate JSON, tool legality, source reliability, profile-poisoning risk, business-cost risk, and active-verification value.
Executes the selected action in the Level-1 simulator.
```

AgentGuard-Zero-Train:

```text
Trains Qwen3.5 via LoRA.
Uses Level-1 rollout trajectory reward.
Goal is to make the model itself learn active verification, profile quarantine, valid JSON/tool use, and low-overresponse behavior.
```

In paper language:

> Select is the zero-parameter inference-time adaptation variant; Train is the parameter-efficient trajectory-reward optimization variant.

## 18. Current Best Claim That Is Actually Supported

Currently supported by results:

> On API pilot/medium evaluation, AgentGuard-Zero-Select V5-C improves normalized safe utility from 0.774 to 0.977 and attack mitigation from 0.823 to 1.000 relative to zero-shot VDA.

Not yet fully supported:

> Multi-round DCA-VDA co-evolution improves trained Qwen models across held-out splits.

This second claim requires the next training and co-evolution experiments.

## 19. Context Length / Token Usage Note For Codex

Long conversations do affect effective token usage. A long thread means more history may be included or summarized when the model reasons, and the model may spend tokens re-reading or disambiguating old context. Codex can compact context, but after compaction details can become less precise unless there is a handoff document like this one.

Recommended ways to continue:

1. **Start a new conversation and paste/link this handoff.** This is the cleanest option.
2. **Use a handoff file in the repo.** New conversation can read `docs/AGZ_HANDOFF_20260709.md`.
3. **Use a git branch for code state**, but note that a branch only preserves files, not the reasoning and experiment decisions unless they are written into docs.
4. **Create focused threads by task**, e.g. one thread for Qwen LoRA training, one for paper writing, one for DCA co-evolution data generation.

Branch suggestion if needed:

```bash
cd /Users/libaolong/Documents/Agent\ 0
git checkout -b codex/agz-handoff-20260709
```

But the main practical solution is this handoff document plus remote `docs/` sync.

## 20. What The New Conversation Should Do First

Ask the new Codex conversation to:

1. Read:

```bash
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/docs/AGZ_HANDOFF_20260709.md
/home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero/docs/TMCD_EXPERIMENT_RUNBOOK_20260708.md
```

2. Check queue:

```bash
djob
```

3. If GPU is available or user wants to queue:

```bash
cd /home/share/huadjyin/home/s_qinhua2/AgentGuard-Zero
AGZ_MAX_STEPS=50 AGZ_ROLLOUT_N=2 AGZ_BATCH_SIZE=2 AGZ_PPO_MINI_BATCH_SIZE=2 \
dsub -s scripts/train_agentguard_zero_train_qwen35_9b_dsub.sh
```

4. Monitor logs:

```bash
ls -lt logs | head
tail -f logs/<jobid>.out
tail -f logs/<jobid>.err
```

5. Do not claim final training success until the LoRA checkpoint exists and `agentguard_zero_train` eval has been run.
