# Gold 指标从单样本到三层报告完整计算

本章回答“0.99 是怎么算出来的”“为什么当前 1.0 不能当新泛化”“三层评测到底差在哪里”。

## 1. 当前 Gold 的事实边界

历史 Match Gold V1：

- 25 个样本；
- 25 个标记为 `human_verified`；
- 覆盖 AI 算法/后端交叉、单项硬门槛、项目技能、同义词、OCR、方向相近、学历和年限边界；
- exact normalized hash 审计与训练集无重叠；
- 但全部由同一审核者完成，且已经被开发过程查看；
- provenance 中仍有 repository seed/AI draft 历史来源，独立性不能只看 status。

因此它现在是高价值冻结回归集，不是新的 blind generalization 证据。

## 2. 评测字段

规则层只评以下字段：

```text
列表：命中技能、缺失技能
文本：匹配等级
布尔：岗位方向匹配、学历匹配、经验匹配
```

总分本身没有与 Gold 直接做误差指标；解释则另做 structural consistency 和 evidence grounding。

## 3. 列表字段的单样本计算

先把字符串去首尾空白并转小写，再转集合。

例：

```text
Gold = {Python, PyTorch, Docker}
Pred = {python, Docker, FastAPI}
```

规范化后：

```text
TP = {python, docker} = 2
|Pred| = 3
|Gold| = 3

Precision = 2/3
Recall    = 2/3
F1        = 2PR/(P+R) = 2/3
```

若 Pred 和 Gold 都为空，三项都记 1；若 Pred 为空而 Gold 非空，三项都记 0。

## 4. 为什么用 set 而不是 list 顺序

命中技能和缺失技能语义是集合，顺序不应影响正确性。重复项也不应获得额外权重。

但只做 lower/strip，不做 alias canonicalization；评测假设产品规则已经输出 canonical names。若 Gold 写 `postgres`、预测写 `PostgreSQL`，这里会判错，这能暴露 Gold 和 schema 未统一的问题。

## 5. 文本和布尔字段

匹配等级只压缩连续空白、去首尾空白、转小写，然后 exact match：

```text
“较匹配” == “较匹配” → 1
“较匹配” != “基本匹配” → 0
```

布尔字段通过 Python `bool()` 后比较。Gold schema 应保证是真布尔；若错误存成字符串 `"false"`，`bool("false")` 实际为 True。这是为何数据 schema 验证比评测函数兜底更重要。

## 6. 多样本如何汇总

对每个字段，先计算每一行指标，再做宏平均：

```text
Macro F1(field) = (1/N) Σ_i F1_i(field)
```

不是先把所有技能 TP/FP/FN 累加的 micro average。宏平均让技能少的边界样本与技能多的样本权重相同。

报告还统计 `num_mismatch_samples`：任一评测字段低于近似 1，就把该行算为 mismatch；解析不可用也计 mismatch。

## 7. parse success 与 analysis valid

```text
jd_resume_parse_success_rate
```

表示 JD 和简历都有可用结构，因此规则层可计算。

```text
analysis_json_valid_rate
```

表示最后解释生成能得到有效 JSON。它们是两件事：规则结果可能有效，但解释 JSON 失败。

## 8. 三层评测的输入差异

### raw_model_derived

只在 JD 和 resume 原始输出都能被 strict JSON 解析且为 dict 时，用 `raw_data` 直接算规则。这里不使用 JSON repair 和业务字段归一化。

它回答：模型原生结构输出够不够好。

### normalized

使用 `parse_json_output` 修复和归一化后的 JD/resume，再计算规则。

它回答：确定性后处理纠错后，结构结果多好。

### product_final

使用实际产品规则结果和解释，增加结构一致性与证据 grounding。

它回答：用户最终看到的结果多好。

## 9. 一个三层变化例子

模型原始 JD：

```json
{"必备技能": ["Python", "Kubernetes"], "加分项": ["Kubernetes"]}
```

原文明确 Kubernetes 只在加分项。简历只有 Python。

```text
raw_model_derived:
  命中技能 [Python]
  缺失技能 [Kubernetes]   ← 错

normalized:
  命中技能 [Python]
  缺失技能 []             ← 证据规则修正

product_final:
  同 normalized，并检查解释不能把 Kubernetes 说成硬短板
```

若只报告 product 指标，会看不到模型本身的错误；若只报告 raw，又会忽略系统真实纠错能力。

## 10. 解释结构一致性如何判断

规则检查包括：

- 方向 false，优势却说“方向一致”；
- 学历 false，优势却说“学历背景满足”；
- 经验 false，优势却说“经验背景满足”；
- 存在缺失技能，短板却说“暂无明显硬性短板”；
- 高匹配结论写成低匹配；
- 低匹配结论写成高度匹配。

它是有限模板规则，不是自然语言事实核查器。没有命中这些短语不代表解释一定正确。

## 11. Evidence grounding 如何判断

对解释中的“匹配优势、主要短板、简历优化建议”提取已知技能，并与：

```text
known_evidence = matched_skills ∪ missing_skills
```

比较。以下会报 issue：

- 解释提到规则证据中不存在的已知技能；
- 优势把 missing skill 当成已掌握；
- 短板把 matched skill 说成缺失；
- 学历/经验陈述与规则布尔相反。

当前冻结回归中 structural consistency 可到 1.0，而 evidence grounding 是 0.92。这恰好说明“逻辑不自相矛盾”比“每个技术陈述都有证据”更容易。

## 12. 为什么 advice validity 明确是 not evaluated

当前没有：

- 用户是否采纳建议；
- 修改前后简历 A/B；
- 投递、面试或 offer outcome；
- 招聘方反馈。

因此无法证明“建议提高求职成功率”。报告把它写为 `unsupported_by_current_data`，而不是用语言流畅度代替业务价值。

## 13. 历史 0.99 和当前 1.0 如何同时解释

历史模型产物在 25 条 Gold 上，技能指标约 0.99、等级/方向约 0.96，说明当时仍有少数明确错误。后来根据错例修改规则和后处理，冻结回归可达到结构字段 1.0、grounding 0.92。

因为开发者已经查看过这 25 条并据此修复，当前 1.0 证明的是：

```text
这些历史错误没有回归
```

它不能证明：

```text
对未来未见真实分布也能达到 100%
```

报告因此标记 `historical_gold_v1_regression` 和 `REGRESSION AFTER INSPECTION`。

## 14. `human_verified` 也不自动等于高质量 Gold

还要审计：

- 谁标的；
- 是否多人一致；
- 标注时间和版本；
- 是否从 AI draft 修改；
- 是否与训练数据同源；
- 是否被用于调规则；
- 边界标签覆盖是否平衡。

当前 25 条达到最小数量和结构门槛，但单审核者、历史 provenance 和已查看状态限制了泛化证据强度。

## 15. 正式新 holdout 应如何建立

最小方案不是追求几千条，而是 30～50 条真正独立、困难且冻结的 pair：

1. 从未进入训练的来源抽取；
2. 按 JD entity 和 resume entity 去重；
3. 预先定义技能 required/bonus、方向兼容、学历、年限口径；
4. 两人独立标注，争议由第三人裁决或保留 disagreement；
5. 标注时不显示当前预测；
6. 冻结 hash 和时间；
7. 在最终对比前不打开逐条答案；
8. 打开后降级为下一轮回归集，再建立新 holdout。

## 16. 指标报告的正确句式

不准确：

> 匹配准确率达到 100%。

准确：

> 在 25 条已人工复核、但已被开发过程检查过的 Gold V1 冻结回归集上，当前产品最终规则字段宏指标为 1.0，解释结构一致性为 1.0，证据 grounding 为 0.92；该结果用于历史回归，不代表新的盲测泛化或录用成功率。

## 17. 面试口述模板

> 我没有只报一个 accuracy。命中和缺失技能按单样本集合 precision/recall/F1 再宏平均，等级与三个硬条件做 exact match；同时分 raw model、normalized、product final 三层，能看出后处理贡献。解释又拆结构一致性和证据 grounding，建议有效性因为没有投递 outcome 明确不评。25 条 Gold 已被用来修复系统，所以当前结构 1.0 只叫 inspection 后冻结回归，不能叫新的 blind 泛化。

