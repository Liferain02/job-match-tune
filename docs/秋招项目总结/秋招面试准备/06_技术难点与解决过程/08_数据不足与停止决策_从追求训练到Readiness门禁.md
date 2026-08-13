# 数据不足与停止决策：从追求训练到 Readiness 门禁

## 1. 为什么“停止训练”也是项目成果

后训练项目很容易形成惯性：继续抓数据、继续合成、继续调参、继续跑 DPO。只要脚本能运行，就会产生新 checkpoint 和新指标。

但当前真正缺的是独立真实 Resume、真实 Match Pair 和业务 outcome。继续在同一合成分布训练，计算成本增加，证据强度不增加。

## 2. 大数据量如何掩盖问题

仓库原始 JD 可达到数十万行，简历 SFT 有15,470行，Match有4,798对。表面规模足够训练。

审计后发现：

```text
Resume：15,470行只有2,557个source group
Resume：2,525个group来自bootstrap，占98.75%
Match：4,798个Pair全部规则合成
Match：1,944条明确年限要求样本全部是不满足
```

行数不能代表独立实体、多样来源或标签平衡。

## 3. 最初继续 synthetic 的边际收益为什么下降

Synthetic 数据适合：

- 验证格式和训练脚本；
- 覆盖明确边界；
- 构造 smoke preference；
- 防止已知回归。

但继续扩大只会强化规则自己的分布。规则生成标签、模型学习标签、再用相似规则评测，会形成闭环自证。

## 4. Readiness 将主观判断变成门禁

Readiness 报告不只检查文件存在，而是汇总：

- 数据量；
- source group多样性；
- bootstrap比例；
- 跨集合泄漏；
- 标签完整性；
- privacy/license；
- Gold人工状态；
- pipeline freshness；
- 真实Pair质量证据；
- preference质量和新鲜度。

训练脚本启动前调用 `assert_training_readiness.sh`，未通过时直接失败。

## 5. 为什么有多个 ready 字段

历史兼容字段可能显示：

```text
ready_for_dpo=true
ready_for_product_dpo=true
```

但执行结论还要看：

```text
dpo_paused_by_quality_goal=true
dpo_execution_ready=false
dpo_pipeline_fresh=false
```

前者表示某些文件/旧门槛满足，后者才表示当前是否应该执行。文档必须解释全部状态，不能挑一个 true 展示。

## 6. DPO 为什么需要第二道显式 pause gate

`_dpo_pause_gate.sh` 默认退出，只有设置：

```text
JOBMATCH_ALLOW_DPO=1
```

才允许继续。

提示明确要求先有新的人工 preference，并由人决定恢复实验。它防止使用者因为看到脚本就误启动昂贵且无证据的 DPO。

## 7. 通用训练 gate 也有 escape hatch

`SKIP_TRAINING_READINESS_GATE=1` 可以绕过通用门禁。这是调试/紧急场景的逃生口，不是常规启动方式。

生产或正式实验若使用跳过变量，必须记录理由，并且结果不能进入正式对比。否则门禁只存在于文档里，没有治理效果。

## 8. Pipeline freshness 为什么重要

上游数据、转换代码或配置更新后，旧 preference 和旧 SFT 产物可能仍然存在。如果只检查文件存在，就会混用不同版本。

freshness 比较：

- 输入与输出修改时间；
- 原始数据状态；
- 输出hash；
- transform代码hash；
- manifest记录。

当前 SFT pipeline fresh，但 DPO pipeline stale，所以即使旧 preference 格式合法，也不能直接训练。

## 9. Engineering Freeze 的准确含义

Freeze 不是项目完美，也不是永远不再开发。它表示：

```text
核心在线链路可运行
历史回归和测试已闭环
已知局限已记录
继续本地合成/调参不再产生可信增量
恢复条件依赖新的外部证据
```

在新数据到来前，优先准备面试讲解和可复现实验，而不是增加 Redis、Kafka、微服务或更大 taxonomy。

## 10. 解除 Freeze 的最小条件

### Resume

获得合法、脱敏、来源独立的真实简历，并通过 privacy/license/source admission。

### Match

获得真实或至少人工构造且独立复核的 JD—Resume Pair，补足年限正负样本。

### Gold

建立未被开发过程查看的 blind holdout，最好双人标注。

### DPO

新的人工 preference、实体级切分、fresh pipeline，以及 SFT vs DPO 的预注册 A/B。

### 产品效果

用户采纳、简历修改和投递结果闭环，才能评估建议有效性。

## 11. 为什么这不是“保守过头”

门禁没有阻止：

- 单元测试；
- 数据审计；
- 推理服务演示；
- 小规模无结论 smoke；
- 文档和复现改进。

它阻止的是把证据不足的训练当正式改进。对求职项目来说，能解释为什么不训练，比多一个无法证明价值的 checkpoint 更能体现判断力。

## 12. 面试回答结构

> 审计前数据看起来很多：15,470条简历样本、4,798个匹配对。按source group和来源分析后，98.75%的简历组来自bootstrap，所有match pair都是规则合成，而且明确年限样本全部是负例。所以我没有继续堆synthetic或跑DPO，而是把source多样性、泄漏、人工状态、privacy和pipeline freshness写进readiness gate。当前SFT管线新鲜但resume/match/multitask未就绪，DPO管线还stale，因此项目进入Engineering Freeze；新真实数据和blind holdout到来才恢复。

## 13. 对应代码和报告

```text
src/jobmatch_tune/eval/report_data_readiness.py
src/jobmatch_tune/eval/assert_training_readiness.py
src/jobmatch_tune/dataset/pipeline_freshness.py
scripts/train/_training_readiness_gate.sh
scripts/train/_dpo_pause_gate.sh
outputs/eval_reports/data_readiness_report.json
outputs/eval_reports/resume_sft_profile.json
```
