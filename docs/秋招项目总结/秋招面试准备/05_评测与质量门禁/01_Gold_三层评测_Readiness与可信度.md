# Gold、三层评测、Readiness 与可信度

## 一、为什么评测比继续训练更重要

模型项目最危险的状态不是指标低，而是指标高但不知道测到了什么。JobMatchTune 先后遇到模板变体、训练重叠、规则与模型贡献混淆，因此最终把评测拆成数据独立性、三层输出质量和训练 Readiness。

## 二、Gold V1 是什么

当前 Match Gold V1 有25条，全部 metadata 标记 `human_verified`，包含方向交叉、硬门槛、技能同义词、OCR、相近方向、学历和年限等难例。

审计检查：

- ID、source group、Pair 不重复；
- label 字段合法；
- difficulty tags 覆盖；
- 与 Match 训练 Pair 的 normalized exact hash 重叠为0；
- JD 侧、Resume 侧与 Match 训练池重叠为0；
- JD 与 Resume 单任务训练池 exact hash 重叠为0。

### 必须说明的限制

这是项目内单人复核冻结集，不是外部多标注者 benchmark。审计主要证明规范化精确哈希隔离，不等于证明所有语义近重复都不存在。16条原始 provenance 是 `repository_seed_without_reviewer_record`，后来统一补了 annotator、reviewed_at 和 rationale；另外9条来自 AI draft 后人工复核。

因此推荐用词是“项目内正式冻结回归集”，不说“大规模真实招聘独立测试集”。

## 三、三层评测

`src/jobmatch_tune/eval/run_match_eval.py` 对同一条样本保留三层结果。

### Raw Model Derived

读取模型原始 JSON，不经过业务 normalization，直接基于 raw fields 计算规则。它回答：模型本身是否把关键字段抽出来。

历史 Gold V1：命中技能 F1 0.6243、缺失技能 F1 0.736、等级 EM 0.56、方向 EM 0.60。

### Normalized

模型输出经过 JSON 修复、字段规范、技能 canonicalization、JD requirement evidence 和当前规则。它回答：确定性产品约束能把多少可恢复错误变成稳定结构。

历史指标：命中/缺失 F1 0.990649/0.986667，等级和方向 EM 0.96，学历和经验1.0。

### Product Final

Normalized Rule Result 加结构化 Explanation。结构指标与 Normalized 相同，另计算 JSON valid 和解释质量。

这三层差值不能简单说成“微调提升”。Raw→Normalized 的大幅改善主要来自工程 normalization 和规则，恰恰说明系统不是纯模型产品。

## 四、正式历史结果

| 指标 | Gold V1 Product Final |
|---|---:|
| 样本数 | 25 |
| JD/Resume parse success | 1.0 |
| Analysis JSON valid | 1.0 |
| 命中技能 F1 | 0.990649 |
| 缺失技能 F1 | 0.986667 |
| 匹配等级 EM | 0.96 |
| 方向 EM | 0.96 |
| 学历 EM | 1.0 |
| 经验 EM | 1.0 |
| Legacy explanation consistency | 1.0 |

错误样本共3条：`match_eval_001` 技能误识别、`match_eval_008` 技能漏召回、`match_gold_candidate_007` 方向和等级错误。

## 五、为什么当前回归1.0不是新泛化

本轮针对已知错误实现通用 OCR、requirement evidence 和 direction compatibility 后，`replay_match_regression.py` 读取冻结 normalized parse，不重新生成模型输出，只重算当前规则。

结果六项结构指标均为1.0。但开发者已经看过 Gold 错误并据此设计修复，因此报告明确标记：

> REGRESSION AFTER INSPECTION

它证明“当前代码没有保留这三类已知退化”，不能证明在未见分布上也达到100%。真正的新泛化结论需要独立 Blind Gold V2。

## 六、解释评测为何拆三类

`src/jobmatch_tune/eval/explanation_grounding.py::evaluate_explanation` 输出：

### Structural Consistency

解释是否与 Rule Result 自相矛盾。例如经验不满足，却把“经验满足”写成优势；缺失 Kafka，却说“已覆盖 Kafka”。

### Evidence Grounding

解释中可确定的技能和硬条件陈述，能否在解析/规则证据中找到支持。当前保存解释的 grounding rate 是0.92：两条历史解释仍存在 Agent 无支持和把 Pytest 作为缺失的旧陈述。

### Advice Validity

建议是否真的改善简历、投递或录用结果。当前返回 `not_evaluated/unsupported_by_current_data`，因为没有用户 A/B、投递反馈和招聘 outcome。

所以 Explanation Consistency=1.0 只能说结构没有矛盾，不能说建议有效率100%。

## 七、Readiness 是什么

Readiness 是训练前门禁，不是一个总样本数阈值。它汇总：

- 数据数量和格式；
- source group 多样性；
- 来源许可和准入；
- 隐私；
- 跨 split 重复；
- Gold/holdout 重叠；
- Match Pair provenance；
- 条件分布；
- pipeline freshness；
- preference 独立性。

当前总报告：

```text
all_ready_for_training = false
all_ready_for_sft = false
not_ready_tasks = [resume, match, multitask]
sft_pipeline_fresh = true
dpo_pipeline_fresh = false
match_real_pair_quality_evidence_ready = false
dpo_execution_ready = false
```

报告里旧的 `ready_for_dpo=true` 只表示部分数量/格式条件曾通过，不能越过 `dpo_paused_by_quality_goal=true` 和 `dpo_execution_ready=false`。

## 八、为何各任务状态不同

### JD

JD 当前数量、格式、拆分和质量审计相对成熟。它仍然主要是结构化弱监督任务，但能作为项目内 SFT 数据线。

### Resume

15,470行中大部分是模板扩展，2,557个 group 中2,525个来自 bootstrap，来源组占比98.75%，外部真实简历准入不足，因此 false。

### Match

4,798 Pair 全为规则合成，没有 human training pair 或 real observed pair；年限显式样本全为不满足，分布单侧，因此 false。

### Multitask

多任务包含不 ready 的 Resume 和 Match，所以整体不能因为 JD ready 就通过。

## 九、Candidate V2

Semantic Boundary Candidate V2 有18条，覆盖 OCR 断词、OCR 假阳性、职责/要求边界、加分项、项目证据和跨岗位方向。

审计状态：

- `needs_human_review=18`；
- `training_eligible_rows=0`；
- 与 V1 comparison pair overlap=0；
- `gold_ready=false`。

它是下一版人工审核候选，不可以提前报产品指标，也不能用于训练后再作为 Gold。

## 十、指标使用原则

### 列表字段

技能使用 precision、recall、F1；集合顺序不应影响正确性。F1 需要逐样本计算再平均，不能只报全局命中数而掩盖小样本失败。

### 类别/布尔字段

等级、方向、学历和经验使用 exact match。25条中0.96意味着错1条，必须同时报告样本数，避免“96%”显得过度精确。

### JSON Valid

只说明输出可解析，不说明语义正确。一个格式完美但技能全错的 JSON 仍然无用。

### Training Loss/Reward Accuracy

只说明优化目标变化，不是产品准确率。DPO reward accuracy 高可能只是模型学会识别 synthetic corruption。

## 十一、一次正确的实验决策

候选 adapter 只有同时满足以下条件才晋级：

1. 训练数据和 pipeline ready；
2. 训练 run manifest 可追溯；
3. Raw 层至少没有结构退化；
4. Product Final 绝对阈值通过；
5. 相对 baseline 的关键指标不回退；
6. 错误类型没有出现高风险新模式；
7. API/文件/UI 主流程可运行。

若 Gold 已被用来调规则，只能作为 regression；必须使用新 blind set 决定泛化。

## 十二、可信度边界的标准回答

> 当前项目证明了：在项目内冻结的25条人工复核样本上，混合流水线能稳定产生较高质量结构结果；审计未发现与当前训练池的五类精确哈希重叠。它没有证明真实招聘决策质量、跨行业泛化、建议有效性或录用概率校准。正因为这些证据缺失，训练门禁保持关闭。

这个回答比单报0.96更可信，也更能体现评测意识。
