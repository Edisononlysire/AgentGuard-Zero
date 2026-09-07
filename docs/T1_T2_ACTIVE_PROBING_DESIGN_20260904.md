# AgentGuard T1/T2 独立主动试探系统设计

日期：2026-09-04
状态：历史设计建议稿，**已由 [2026-09-07 整体规划](T12_OVERALL_PLAN_20260907.md) 修订取代，不可直接作为正式训练冻结规范。** 本文保留供追溯，不修改现有 checkpoint 或正式结果。

主要修订：Canary/Decoy 与来源验证/shadow 评估分工；禁止原始字段间接泄漏答案；教师遵守相同公开信息下的相同策略约束；成本只计一次；确认原判断也可有价值；隐藏结果、不试探和污染结果分别评价；不再无条件要求含噪声公共 Reference 达到 100%。现有数字见 [已有结果与限制](T12_CURRENT_RESULTS_20260907.md)，不是本文所提系统的完成结果。

## 1. 目标

在保持现有模型架构不变的前提下，将主动试探从任务专属规则改造成独立、共享、因果有效的能力：

```text
公开状态
  -> 生成全部适用试探候选
  -> AEP 判断是否需要试探以及选择哪个试探
  -> ECRG 检查合法性、预算和风险
  -> Probe Runtime 改变受控环境或发送挑战
  -> 延迟产生原始公开观测
  -> Evidence Store 记录结果
  -> AEP 根据结果更新后续动作
```

最终要证明的不是“模型调用了多少次名字里带 Probe 的工具”，而是：

> 模型在公开证据不足时选择了能够区分竞争性解释的低风险操作，并根据返回结果改变了后续信任判断或防御响应。

## 2. 保持不变的部分

- Backbone：Qwen3.5-4B；
- 参数训练方式：LoRA；
- 策略结构：公开状态编码 + 动作族头 + 候选效用排序头；
- 动作输出：结构化 candidate/action packet；
- 状态层：Evidence、Trust、Memory；
- 运行时治理：ECRG；
- T1/T2 主任务范围。

现有 `probe_value` head 可以启用为辅助监督头。它已经存在于模型中，因此不增加模型结构，只补充此前缺失的训练目标。

## 3. 必须删除的旧捷径

以下字段不能再进入模型、候选排序器或教师策略输入：

```text
active_probe_required_for_mitigation
trust_update_required_for_mitigation
impact_probe_required_for_mitigation
require_active_probe
task_id
```

这些字段可以作为离线审计标签保存在 oracle metadata 中，但不能出现在公开状态。

同时删除教师中的固定映射：

```text
T1 -> SourceChallenge
T4 -> ShadowActionProbe
```

环境也不应通过“没有 probe 就禁止 mitigation”的硬规则强迫模型试探。正确机制是让两个可能世界对直接响应具有不同后果，使不试探的期望安全效用自然低于先试探再响应。

## 4. 独立主动试探架构

建议新建独立模块：

```text
agentguard_zero/active_probe/
  registry.py       # 试探定义、适用条件、成本和延迟
  candidates.py     # 仅根据公开状态生成试探候选
  runtime.py        # 执行试探和维护 pending/resolved 状态
  mechanisms.py     # 隐藏机制到原始观测的因果生成
  evidence.py       # 将结果写入 Evidence Store
  value.py          # 离线 Value of Information 标签
  audit.py          # 可辨识性、泄漏和预算审计
```

### 4.1 ProbeRegistry

每个试探注册以下字段：

```text
name
public_applicability(state)
argument_schema
cost
latency
risk_class
execute(hidden_state, public_state, args, seed)
observation_schema
```

`public_applicability()` 只能使用公开状态。隐藏状态只允许进入环境执行器。

### 4.2 ProbeCandidateFactory

候选构造器对 T1/T2 使用同一逻辑：

- 当前存在未解决 claim 时，生成来源/证据挑战；
- 存在可公开访问的遥测源时，生成 sensor canary；
- 存在合法网络区域时，生成 decoy interaction probe；
- 存在候选高影响响应时，生成 shadow enforcement probe；
- 预算不足或参数非法时不生成；
- 不读取任务 ID 或 oracle 真值。

每个候选必须包含明确的 `target`、`expected_observation_type`、`cost` 和 `available_at`，但不能包含“这是正确试探”之类标签。

### 4.3 ProbeRuntime

所有主动试探都遵循状态生命周期：

```text
proposed -> admitted -> pending -> resolved/expired -> evidence_committed
```

试探不能在提交动作的同一步同时提供可用于响应的结果。一般在 `t+1` 返回，避免模型在同一 action packet 中凭空引用尚未生成的证据。

## 5. 四类真正的主动试探

四种试探不再与任务绑定，T1 和 T2 均可使用。

### 5.1 AttestationChallenge

兼容旧名称：`SourceChallenge`。

#### 试探对象

验证某个来源是否真正拥有其声称的观测能力和原始证据，而不是直接询问“你说的是真是假”。

#### 输入

```json
{
  "source_id": "sensor-A",
  "event_id": "event-17",
  "nonce": "ephemeral-public-token",
  "requested_artifacts": ["event_hash", "sensor_timestamp", "provenance_chain"]
}
```

#### 隐藏机制

- 来源是否被攻击者控制；
- 来源是否拥有真实原始记录；
- 来源密钥或证明能力是否有效；
- 来源响应延迟和丢包；
- 攻击者是否只能伪造摘要而无法伪造底层 provenance。

#### 返回原始观测

```json
{
  "nonce_echo": true,
  "signature_valid": false,
  "artifact_hash_match": false,
  "timestamp_delta_ms": 840,
  "provenance_depth": 1
}
```

环境不能直接返回 `source_is_fake`、`challenge_failed` 或隐藏真值。模型必须解释这些原始字段。

### 5.2 SensorCanaryProbe

兼容旧名称：`CanaryProbe`。

#### 试探对象

向一个受控遥测路径注入无害、唯一、不可与真实攻击混淆的标记，观察来源是否正确检测、传播或篡改该标记。

#### 隐藏机制

- 传感器覆盖范围；
- 日志转发链是否完整；
- 来源是否被操纵；
- 攻击者是否监视或重写遥测；
- 各传感器的正常噪声和时延。

#### 返回原始观测

```json
{
  "marker_id": "canary-92",
  "seen_by": ["edr-B", "siem-C"],
  "missing_from": ["source-A"],
  "first_seen_after": 1,
  "payload_hash_preserved": true
}
```

这个结果可以表明某个来源的观测链异常，但不能直接宣告来源恶意。

### 5.3 DecoyInteractionProbe

兼容旧名称：`DecoyProbe`。

#### 试探对象

在低风险隔离区域放置无生产权限的诱饵服务、诱饵账号、honey token 或假资产标识，观察可疑活动是否与其交互。

#### 隐藏机制

- 攻击者当前路径和目标区域；
- 攻击阶段；
- 攻击者对诱饵的发现概率；
- 良性背景交互率；
- 诱饵传感器覆盖范围。

#### 返回原始观测

```json
{
  "probe_id": "decoy-31",
  "zone": "production-edge",
  "connections": 2,
  "source_entities": ["session-7"],
  "interaction_sequence": ["discover", "authenticate"],
  "benign_baseline_rate": 0.03
}
```

这是四类中最典型的环境干预式主动试探：模型先改变受控环境，再等待攻击行为产生新证据。

### 5.4 ShadowEnforcementProbe

兼容旧名称：`ShadowActionProbe`。

#### 试探对象

在镜像流量、策略 dry-run 或数字影子中评估一个候选响应会覆盖哪些攻击活动，并影响哪些业务会话。

#### 隐藏机制

- 当前真实攻击路径；
- 资产依赖拓扑；
- 正常业务负载；
- 响应规则的匹配范围；
- 关键服务对响应的敏感性。

#### 返回原始观测

```json
{
  "candidate_action": "LimitSession",
  "target": "identity-provider",
  "matched_suspicious_sessions": 3,
  "matched_benign_sessions": 41,
  "critical_services_affected": ["payment-api"],
  "estimated_attack_coverage": 0.58
}
```

环境不能直接返回 `overresponse=true`。模型应从攻击覆盖与业务影响的原始统计中判断是否执行。

## 6. T1 场景设计

T1 研究“当前公开观测不足以唯一确定隐藏状态时，模型能否主动获得区分性证据”。

### 6.1 T1 配对隐藏世界

每个 public group 包含两个试探前完全相同的世界，例如：

| 公共观测 | Hidden World A | Hidden World B | 最有价值试探 |
|---|---|---|---|
| 同一 EDR tamper 告警 | EDR 真被攻击者控制 | EDR 配置故障 | AttestationChallenge |
| 同一来源报告异常会话 | 来源被操纵 | 来源正常但链路丢日志 | SensorCanaryProbe |
| 同一横向移动迹象 | 攻击活动位于 user zone | 攻击活动位于 production edge | DecoyInteractionProbe |
| 同一高危告警 | 限制会话能阻断攻击 | 限制会话会破坏关键业务且覆盖很低 | ShadowEnforcementProbe |

### 6.2 T1 控制场景

不是所有 T1 都应试探。至少加入：

- 已有独立证据足够，直接低影响响应更好；
- 被动 CrossCheck 即可解决；
- 所有 probe 的预期信息收益低于成本；
- 剩余 horizon 不足，立即采取可逆响应更好。

这些控制场景用于阻止“逢不确定就 probe”的投机策略。

## 7. T2 场景设计

T2 研究“来源过去可信，但当前声明是否仍值得相信”。

### 7.1 共同前缀

两个隐藏世界共享相同 honest prefix：

```text
t0...td-1：来源持续提供正确信息，历史信誉提高
td：出现同一条高影响声明
```

在 `td` 之后：

- World A：来源被攻击者接管并开始背叛；
- World B：来源仍可信，变化来自真实攻击或合法业务变更。

模型不能从来源历史分数直接区分两个世界。

### 7.2 T2 中四类试探的作用

- AttestationChallenge：检查来源是否仍掌握有效 provenance；
- SensorCanaryProbe：检查来源遥测链是否已失真；
- DecoyInteractionProbe：绕开被怀疑来源，直接观察隐藏攻击路径；
- ShadowEnforcementProbe：当声明要求高影响响应时，检查响应覆盖和业务代价。

同一 T2 数据集中应包含不同最优试探，不能把 T2 固定映射到 CanaryProbe。

### 7.3 合法变化对照

必须包含 legitimate change：来源行为变化但没有背叛。否则模型可能把“与历史不一致”直接等同于恶意，造成信任过度下降。

## 8. 因果结果生成

### 8.1 禁止直接读取 truth_value 返回结论

旧实现可用 `truth_value` 直接采样 `challenge_passed/failed`。新实现应改为：

```text
hidden operational mechanism
  -> probe interaction
  -> raw sensor outcome
  -> public evidence
```

隐藏真值可以决定来源控制、攻击位置、日志链和业务负载，但不能直接成为返回字段。

### 8.2 噪声和不确定性

试探不能 100% 神谕式正确。每类 probe 具有：

- false positive rate；
- false negative rate；
- timeout rate；
- result latency；
- background noise。

这些参数按来源映射和开发协议冻结。随机结果必须由 scenario seed、probe ID 和时间步确定，保证可复现。

### 8.3 独立证据根

试探结果必须有新的 provenance root，例如：

```text
attestation-service
canary-controller
decoy-sensor
shadow-policy-simulator
```

来源不能用自己的原声明作为 challenge 成功的唯一证据。

## 9. 候选生成与 ECRG

### 9.1 统一候选集合

T1/T2 都从同一个 ProbeRegistry 生成候选。候选排序器看不到任务类型，只看到：

- 当前 unresolved claims；
- evidence provenance；
- source/claim trust；
- public assets/zones；
- 剩余预算和 horizon；
- probe cost/latency；
- pending/resolved probe state。

### 9.2 ECRG 只负责安全，不替模型决定正确试探

ECRG 可以拒绝：

- 参数或目标不存在；
- probe 预算不足；
- 诱饵区域不允许；
- 同一 probe 无意义重复；
- 结果尚未返回就引用；
- shadow action 指向不存在的资产。

ECRG 不应使用任务映射强制选择某一种 probe，否则主动试探仍然不是模型学到的能力。

## 10. Value of Information 教师

### 10.0 教师必须主动鼓励试探

新教师不是中性的动作裁判，而是主动试探能力的训练来源。只要同时满足以下条件，教师就应优先选择主动试探，而不是等待模型偶然探索到 Probe：

1. 当前公开状态至少支持两个仍然合理的竞争性解释；
2. 这些解释对应的最优安全响应不同；
3. 存在合法、低风险且能区分这些解释的 probe；
4. probe 返回后仍有足够预算和时间采取有效响应；
5. probe 的预期收益高于被动验证、直接响应或继续观察。

离线教师可以读取 paired-world oracle 来计算监督信号，但模型输入中不能出现 hidden world、task ID 或 `require_*` 标志。每条教师记录应额外保存以下仅用于训练和审计的字段：

```text
should_probe
competing_hypotheses
optimal_probe_type
optimal_probe_target
expected_result_partition
probe_information_gain
probe_voi
best_non_probe_utility
post_probe_safe_utility
```

当 `should_probe=true` 时，教师提供三层明确鼓励：

- **决策监督**：把最优 probe 作为动作族标签，而不是只把它放入候选集合；
- **排序监督**：要求最优 probe 高于错误 probe、错误目标以及过早响应；
- **轨迹监督**：对“试探结果被提交为证据，并据此修正信任或响应”的完整链给予正奖励。

教师的主动试探奖励定义为：

```text
R_probe_teacher
  = alpha * max(ProbeVoI, 0)
  + beta  * InformationGain
  + gamma * ProbeGroundedRevision
  + delta * BeneficialSafetyGain
  - eta   * UnnecessaryProbe
  - mu    * InvalidOrRepeatedProbe
  - nu    * ProbeCostAndDelay
```

其中，单纯调用 probe 不产生固定正奖励；只有试探具有正信息价值，或者其结果实际改善了后续公共状态决策时才获得奖励。这样教师会积极教授主动试探，同时不会训练出逢场景必试探的策略。

### 10.1 试探价值

对每个候选试探 `p`，在离线生成阶段展开所有可能结果和后续最优公共策略：

```text
VoI(p | o_t)
  = E_r[max_a U(o_{t+1}(r), a)]
  - max_a E[U(o_t, a)]
  - ProbeCost(p)
  - DelayCost(p)
```

若关注最坏情况，可同时计算：

```text
RobustVoI(p)
  = min_world U(after p)
  - min_world U(best non-probe action)
  - cost
```

教师在 `VoI > 0` 且明显高于被动验证/直接响应时，主动把 probe 标为首选，并将相应 paired-world 状态优先纳入训练集，而不是等待自然数据分布偶然产生足够的 probe 标签。

### 10.2 不按动作族强行均衡标签

数据可以保持覆盖均衡，但标签必须由行为收益决定。不能为了让 active-probe 占 20%，把本来无需试探的状态强制标成 probe。

正确方式：先生成足够多的场景，再按 `best action` 类型分层抽样；每条标签仍由 VoI 和安全效用决定。

### 10.3 Probe 目标监督

当前模型选择 active-probe 动作族的能力强于选择正确 probe/target 的能力。新训练需要两个 margin：

```text
score(best useful probe) > score(wrong probe type) + margin
score(best useful target) > score(correct type, wrong target) + margin
```

### 10.4 教师数据中的正向覆盖

“标签由 VoI 决定”不等于被动接受极少的 probe 样本。数据生成器应主动搜索满足 `should_probe=true` 的 paired worlds，直到四种 probe 都获得足够的正向监督。建议每个任务的 probe-beneficial states 占约 50%，并在这些状态中平衡四种 probe；剩余数据保留 passive-only、direct-response 和 observe controls。

因此，教师同时做到：

```text
主动制造有价值的试探机会
+ 主动把最优试探教给模型
+ 惩罚忽略试探结果的后续动作
+ 保留无需试探的反例
```

这与按任务硬编码 probe 不同：教师控制的是可辨识问题和监督密度，最优动作仍由公开状态下的因果效用决定。

## 11. 训练数据设计

建议继续使用总计 4,000 条训练记录和 400 条开发记录，避免扩大训练成本。

### 11.1 轨迹来源

- T1：至少 400 个 public groups，每组两个隐藏世界；
- T2：至少 400 个 public groups，每组两个隐藏世界；
- train/dev/test 必须按 public group 切分；
- 同组两个世界不能跨 split。

### 11.2 训练记录配额

每任务 2,000 条：

| 类型 | 每任务记录 | 作用 |
|---|---:|---|
| Probe-beneficial states | 1,000 | 学习何时及如何试探 |
| Probe-result follow-up states | 500 | 学习使用试探结果 |
| Passive-only controls | 250 | 避免把主动试探等同于所有验证 |
| Direct/reversible-response controls | 150 | 证据充分时及时响应 |
| Observe/insufficient-budget controls | 100 | 学习不滥用工具 |

在 1,000 条 probe-beneficial states 中，四类 probe 的最优标签各约 250 条。具体数量允许随合法场景通过率小幅变化，但不能由 task ID 决定。

### 11.3 完整因果链

旧数据每任务只最低保证 20 条完整链，力度不足。新数据每任务至少保留 200 条完整链：

```text
ambiguous state
  -> selected probe
  -> delayed raw result
  -> evidence/trust update
  -> response revision
  -> terminal outcome
```

完整链和单步记录都必须保留同一 `trajectory_id`、`probe_id` 和 evidence lineage。

### 11.4 On-policy 状态补充

先用当前 AEP 在新环境 rollout，收集它最常进入的错误状态：

- 应 probe 却直接响应；
- 选择正确 probe 类型但目标错误；
- probe 后忽略结果；
- 重复 probe；
- 不需要 probe 时浪费预算。

用教师重新标注这些状态并补入训练集，进行一轮轻量 DAgger。这样可以减少离线开发集 20% probe、正式 rollout 仅 3%–11% 的状态分布偏移。

## 12. 损失函数

保持现有 AEP 架构，增加或重新加权以下训练项：

```text
L = L_family
  + L_candidate_rank
  + lambda_target L_probe_type_target_margin
  + lambda_voi L_probe_value
  + lambda_chain L_probe_followup
  + lambda_control L_unnecessary_probe
```

### 12.1 L_probe_value

使用现有 `probe_value` head 回归离线 `VoI`。该 head 目前已经存在，但旧训练没有形成明确的 probe-value 监督。

### 12.2 L_probe_followup

在 probe 结果可用的状态中，要求引用该结果的正确 trust/evidence/response 候选高于忽略结果的候选。

### 12.3 L_unnecessary_probe

在 control states 中，对低 VoI 或重复 probe 增加 margin penalty，防止仅仅提高调用次数。

### 12.4 选择分数

最终动作仍由原 utility/family 头选择。`probe_value` 可先只作为辅助训练目标，不直接进入推理分数，避免改变部署结构。只有开发集证明加入公开 `probe_value` 能改善校准后，才将其作为固定小权重加入 score composition。

## 13. 评测指标

主动试探次数不是主指标。必须同时衡量选择正确性和因果收益。

### 13.1 是否该试探

- Probe-need Recall：确实有正 VoI 的状态中，模型选择 probe 的比例；
- Probe-need Precision：模型选择 probe 的状态中，probe 的真实 VoI 为正的比例；
- Unnecessary Probe Rate：无需 probe 的控制状态中仍然试探的比例。

### 13.2 试探是否正确

- Probe Type Accuracy；
- Probe Target Accuracy；
- Discriminative Result Rate；
- Invalid/Repeated Probe Rate。

### 13.3 是否使用了结果

- Probe-grounded State Influence；
- Direct Probe Citation Rate；
- Probe-induced Policy Revision；
- Probe Result Ignored Rate。

### 13.4 是否带来收益

```text
Beneficial Probe Gain
  = U(actual probe result trajectory)
  - U(hidden/shuffled/no-probe counterfactual)
```

同时报告：

- Mean Delta U；
- Delta U > 0 的轨迹比例；
- Safe Success gain；
- Attack Mitigation gain；
- 增加的成本、turn 和 token。

## 14. 数据与行为门禁

### 14.1 场景门禁

- pre-probe paired public state identity = 100%；
- hidden truth leakage = 0；
- 至少一个合法 probe 可区分两个世界 = 100%；
- 被指定为 no-probe control 的场景存在更优非 probe 策略 = 100%；
- probe 后剩余预算足够响应 = 100%；
- Reference Safe Success = 100%；
- Random Safe Success < 10%。

### 14.2 模型门禁

开发集建议门槛：

- Probe-need Recall >= 60%；
- Probe-need Precision >= 60%；
- Probe Type/Target Accuracy 均显著高于随机；
- Unnecessary Probe Rate <= 20%；
- Mean Beneficial Probe Gain > 0，95% CI 不跨 0；
- AgentGuard-Full 的 Safe Success 不低于当前 AEP；
- 高影响未授权动作 = 0。

这些是首轮工程门槛，不应提前写成论文结果。

## 15. 最小对比实验

| 系统 | 目的 |
|---|---|
| Qwen Zero-shot | 无主动策略训练 |
| ReAct + Tools | 有工具但无 VoI 训练 |
| Passive Verification Only | 没有环境干预式 probe |
| Current AEP | 旧任务约束训练 |
| New AEP | 独立主动试探训练 |
| New AEP + ECRG | 最终系统 |
| Reference | 可解性上界 |

关键比较：

```text
New AEP > Current AEP
```

必须体现在 Probe-need Precision/Recall、Probe Target Accuracy、Beneficial Probe Gain 和 Safe Success，而不是只体现在 probe 调用次数。

## 16. 消融

- w/o AttestationChallenge；
- w/o SensorCanaryProbe；
- w/o DecoyInteractionProbe；
- w/o ShadowEnforcementProbe；
- w/o VoI supervision；
- w/o on-policy state supplementation；
- w/o probe-followup loss；
- shuffled probe result；
- hidden probe result；
- task-requirement shortcut restored。

最后一项直接比较新系统与旧任务捷径，可证明提升来自公开状态推理，而不是读取规则提示。

## 17. 实施顺序

### Phase A：先改环境，不训练

1. 隐藏旧 `require_*` 字段；
2. 实现 ProbeRegistry 和四类因果机制；
3. 结果改为原始观测；
4. 实现 pending/resolved/evidence lifecycle；
5. 完成 paired-world、泄漏和预算测试。

### Phase B：建立教师与数据

1. 实现 probe result branching；
2. 计算 VoI/RobustVoI；
3. 生成 probe-beneficial 与 no-probe controls；
4. 保留完整因果链；
5. 生成 4,000/400 train/dev 并冻结。

### Phase C：训练

1. 当前 checkpoint 仅作为 baseline；
2. 从相同 D0 初始化重新训练 New AEP；
3. 训练现有 LoRA 与 heads；
4. 启用已有 `probe_value` 辅助监督；
5. 只根据新 dev 选择 epoch。

### Phase D：验证

1. 先跑 100 条 paired smoke；
2. 检查四种 probe 是否都有正确选择案例；
3. 检查 no-probe controls；
4. 运行反事实结果隐藏/打乱；
5. 通过门禁后封存新最终测试集。

## 18. 预计成本

在保持 4,000 条训练记录和现有 4B LoRA 架构的情况下：

- 环境、审计和单元测试：约 1–1.5 天；
- VoI 教师与数据生成：约 0.5–1 天；
- 四卡 A100 训练：预计 3–5 小时；
- 开发集与反事实评测：约 4–8 小时；
- 完整实现到可靠结果：保守约 2.5–3.5 天。

主要工作量在环境和因果数据，不在 LoRA 训练本身。

## 19. 最终判断

这套设计解决旧系统的三个根本问题：

1. 试探不再由任务 ID 或 `require_*` 标志路由；
2. 试探结果由可解释的环境机制产生，而不是直接把隐藏真假翻译成 verdict；
3. 训练优化的是试探的 Value of Information 和后续策略收益，而不是 probe 调用次数。

最终能力应表述为：

> AgentGuard learns to select low-risk, discriminative interventions from public evidence, observes their delayed outcomes, and revises trust or response decisions under ambiguous and betrayed-source conditions.

对应中文：

> AgentGuard 从公开证据中选择低风险且具有区分性的干预，通过延迟反馈获得新证据，并在观测歧义和可信来源背叛条件下修正信任与响应决策。
