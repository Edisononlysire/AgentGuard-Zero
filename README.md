# AgentGuard-Zero

**T1/T2：对抗性信任条件下的成本敏感主动证据获取。**

本仓库只维护当前 T12 源码、最新整体规划和最近核验的结果摘要。
旧作业、旧版设计、无关实验与运行日志已从默认分支移除。

> **状态：已有 T12 实现与结果可以检查；新版主动试探仍待实现和验证。**
> 当前结果来自冻结场景生成器与 Qwen3.5-4B LoRA 候选策略，ECRG 未启用。
> 不把它写成新版因果试探、ECRG 增益或 DCA 参数共进化的结果。

这里的“最新”指当前发布整理与核验状态，不表示规划中的全部方法已经实现。
已有实现、历史结果和待实施规划分别由下面的文档说明。

## 从这里阅读

| 文档 | 内容 |
|---|---|
| **[整体规划](docs/PLAN.md)** | 唯一有效的研究方案：T1/T2、主动证据策略、ECRG、教师、实验和迁移 |
| **[模型架构](docs/ARCHITECTURE.md)** | 当前真正产生结果的模型、LoRA、排序头、训练与场景生成路径 |
| **[现有结果](docs/RESULTS.md)** | 最新核验的历史结果、原始分母、失败门禁及来源哈希 |
| **[当前状态与问题](docs/STATUS.md)** | 已验证内容、确定性反例、待修正项和清理边界 |

## 已有结果

| 当前 T12，ECRG 关闭 | Safe Success | Attack Mitigation |
|---|---:|---:|
| T1 观测歧义与主动调查 | 50/150（33.33%） | 65/150（43.33%） |
| T2 信任建立后背叛／合法变化 | 100/150（66.67%） | 132/150（88.00%） |

这是旧协议的 development/retention 证据，不是新协议未接触的最终测试。
原始 retention gate **未通过**，包括过度响应与部分意图识别退化。
详见[结果解释](docs/RESULTS.md)及[可机读汇总](results/t12_legacy_retention_20260907.json)。

## 当前代码路径

```text
T1/T2 配对隐藏世界 -> 公开观测与合法候选
                   -> 公开信息教师与流程约束监督
                   -> 4B LoRA：动作族选择 + 候选效用排序
                   -> 环境执行、证据更新、轨迹评价
```

| 路径 | 用途 |
|---|---|
| `agentguard_zero/candidate/` | 候选生成、模型编码、排序头与策略 |
| `agentguard_zero/recovery/canonical_scenarios.py` | 受控场景生成器 |
| `agentguard_zero/candidate/cyber_grounding.py` | 网络安全语义与遥测映射 |
| `agentguard_zero/recovery/public_teacher.py` | 公开信息教师 |
| `agentguard_zero/env/`、`world/`、`defender_state/` | 环境、隐藏状态、证据、信任与记忆 |
| `agentguard_zero/governance/` | 公开证据授权与 ECRG 相关实现 |
| `scripts/build_v11_t12_native_v2.py` | 当前 4,000/400 条候选状态数据构建 |
| `scripts/train_candidate_ranker.py` | LoRA 候选策略训练 |
| `scripts/eval_v11_single_expert_policy.py` | 单一 T12 专家轨迹评测 |
| `scripts/select_v11_t12_epoch.py` | 开发轨迹上的 checkpoint 选择 |
| `configs/t12/t12_ranker_public.json` | 已有模型的公开训练配置摘要 |

仍被上述入口导入的共享兼容模块保留原文件名，避免破坏依赖。
它们不表示恢复四任务、三专家路由或旧共进化实验作为当前主线。
当前模型仍可看到 `require_*` 流程要求；这是新版要消除的已知限制，
不是已经完成无捷径验证的实现。

## CPU 检查

参考实验使用 Python 3.12。无需下载模型即可执行环境与数据路径检查：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/smoke_t12_scenarios.py --groups-per-task 2
python -m pytest -q
python scripts/check_public_release.py
```

参数训练另需本地 Qwen3.5-4B 权重与匹配 CUDA 的运行环境，依赖见
`requirements-training.txt`。本发布不包含模型、adapter、原始训练／测试数据
或机器专用作业脚本。汇总计数支持重算表格，但不替代完整实验材料。

## 发布边界

- 不发布 checkpoint、原始轨迹、日志、凭据或服务器路径。
- 保留结果中的负面证据；不通过清理仓库隐藏失败。
- 不在本次整理中改变训练协议或启动训练；待修复内容明确列在状态页。
- 旧版可由 Git 历史及 `pre-cleanup-20260907` 标签追溯，不在默认目录重复摆放。
- 代码仅执行受控符号化防御，不执行真实攻击、恶意载荷或漏洞利用。

## License

项目代码采用 Apache-2.0。第三方出处与许可材料见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
