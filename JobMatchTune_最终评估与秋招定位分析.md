# JobMatchTune 最终评估与秋招定位分析

更新时间：2026-08-13  
最新审查基线提交：`897c2cc72198099f38778f6fc7d491281b3f9f13`

## 一、最终结论

JobMatchTune 已经达到 **Engineering Freeze（工程冻结）** 状态。

这里的“冻结”不是指项目已经不存在任何问题，而是指：在当前数据条件下，能够通过代码、规则、工程和评测方法合理解决的问题已经基本收口；剩余最关键的问题主要依赖新的合法真实数据、独立人工标注和真实业务反馈，继续单纯增加代码、规则或训练轮次，边际收益已经很低，反而可能降低项目可信度。

因此，当前不建议继续主动开发 JobMatchTune。后续只在获得新的合法独立 Resume、真实/人工 Match Pair、人工 preference 或新的独立 Blind Gold 后重新开启训练和核心逻辑迭代。

## 二、当前项目准确定位

JobMatchTune 不应再被描述为“一个 Qwen3-14B 简历匹配模型”，更准确的定位是：

> **面向中文招聘场景的模型后训练、结构化抽取、可解释人岗匹配与可信评测工作台。**

当前应区分离线训练治理和在线产品请求两条链，避免把 SFT/DPO 误解为用户每次请求都会执行：

```text
离线：招聘/Resume 数据 → License/Provenance/Privacy
      → 清洗/去重/Source Group → Entity-level Split
      → Readiness → SFT/QLoRA → 可选 Preference/DPO
      → 三层评测/Gold/泄漏审计 → Adapter 晋级或拒绝

在线：文本或文件 → JD Parse + Resume Parse
      → Skill/OCR Canonicalization + JD Requirement Evidence
      → Direction Compatibility + Deterministic Match Policy
      → LLM Explanation → API/Web Response
```

项目最有价值的部分已经不是“用了哪些模型技术”，而是形成了一条 **Evaluation-driven Post-training Engineering** 闭环。

## 三、核心语义边界已经工程收口到什么程度

这里的“工程收口”不等于产品外部有效性已经成熟：当前仍没有真实招聘 outcome、规模化外部 Gold 或建议有效性验证。

### 1. Skill / OCR Canonicalization

最初技能归一化主要依赖有限 alias，例如 `Py thon → Python`、`My SOL → MySQL`。最终 Gold 暴露出 `Py test → Pytest` 未恢复的问题。

现在没有通过单条特判解决，而是实现了基于 **Known Skill Vocabulary 的有界 OCR Canonicalization**：

```text
已知 canonical skill vocabulary
        ↓
安全归一化
        ↓
OCR 空格 / 连字符 / 符号异常压缩
        ↓
仅在已知技能词表内部匹配
```

这避免了对全文直接删除空格带来的误识别。

当前已覆盖类似：

```text
Py test      → Pytest
Py thon      → Python
Kubernet es  → Kubernetes
My SOL       → MySQL
C + +        → C++
Node . js    → Node.js
Spring  Boot → Spring Boot
```

同时也加入了反向假阳性测试，避免 `Postman`、普通英文短语、短技能字符间隔等被错误归一。

这一轮全量测试还进一步发现并修复了短技能 `C` 从 `C++ / C语言 / C#` 中被重复抽取的确定性 Bug。

### 2. JD Requirement / Responsibility Boundary

之前最大的语义风险之一是：

```text
JD 中提到某个技术
≠
该技术一定是硬性必备技能
```

例如：

```text
岗位职责：负责 Agent 应用开发
```

不等价于：

```text
任职要求：必须掌握 Agent
```

现在已经将技能 Evidence 明确拆分为：

```text
requirement
responsibility
bonus
other
```

只有 `requirement evidence` 能进入最终“必备技能”。

这意味着模型输出的 `必备技能` 字段不再直接被当作事实，而会继续回到原始 JD 文本中寻找证据。

### 3. Direction Compatibility

旧逻辑主要是：

```text
same
或 substring
```

无法表示 AI 后端、模型服务、算法平台、AI Infra 等交叉岗位之间的“高度兼容但并不相同”。

当前内部方向判断已经升级为：

```text
exact
compatible
mismatch
```

`compatible` 并不是简单的岗位名称 alias，而需要结合：

- 最小岗位 taxonomy；
- JD 职责上下文；
- Resume 项目/职责上下文；
- 至少一定数量的共享技能；
- AI 平台类岗位还需要 AI 技能 + 工程技能的组合证据。

因此没有做 `算法工程 == 后端开发` 这种危险的宽泛映射。

### 4. Match Scoring Policy

当前评分仍然是：

```text
方向   20
技能   45
学历   10
经验   15
项目   10
```

以及：

```text
>= 85  高匹配
>= 65  较匹配
>= 45  基本匹配
```

但现在代码已经明确把这些数字建模为：

> **heuristic compatibility score**

而不是录用概率、投递成功率或真实招聘 outcome 的统计估计。

当前没有真实观察 Pair 和真实招聘 outcome，因此正确做法不是用 25 条 Gold 或 4,798 条 synthetic Pair 重新拟合权重，而是明确它是可审计的产品启发式。

### 5. Explanation Evaluation

原来的 `explanation_consistency_rate = 1.0` 很容易被误解成“建议有效率 100%”。

当前已经拆成三个概念：

```text
Structural Consistency
Evidence Grounding
Advice Validity
```

其中 Advice Validity 当前明确是：

```text
not_evaluated
```

因为项目没有真实投递反馈和 downstream outcome。

## 四、正式 Match 结果应该怎么解释

历史正式 Match Gold V1：

- 25 条；
- 25/25 human verified；
- 与当前 JD、Resume、Match 训练池完成五类规范化精确哈希隔离审计。

它是项目内单人复核的冻结回归集，不是外部多标注者 benchmark；精确哈希为零也不能证明所有语义近重复都不存在。

正式 Product Final：

| 指标 | 结果 |
|---|---:|
| 命中技能 F1 | 0.990649 |
| 缺失技能 F1 | 0.986667 |
| 匹配等级 Exact Match | 0.96 |
| 岗位方向 Exact Match | 0.96 |
| 学历 Exact Match | 1.0 |
| 经验 Exact Match | 1.0 |
| Explanation Consistency | 1.0 |

语义边界修复后的保存解释评测还应同时报告：Structural Consistency 为 1.0，Evidence Grounding 为 0.92，Advice Validity 为 `not_evaluated`。因此 Explanation Consistency 不能解释成“建议准确率”。

最重要的口径是：

> 这不是“Qwen3-14B 模型自身准确率 96%”。

Product Final 实际是：

```text
Domain Model Parse
+
Deterministic Normalization
+
Deterministic Match Rule
+
LLM Explanation
```

## 五、为什么现在 Regression 1.0 不能替代历史 0.96

本轮语义边界修复后，保存预测重放的六项结构指标已经达到 1.0。

但是这些错误样本已经被查看并用于分析，因此新的结果必须标记：

> **REGRESSION AFTER INSPECTION**

不能重新称为 Blind Generalization。

因此对外最稳妥的口径仍然是：

```text
历史 Formal Gold V1：
0.991 / 0.987 / 0.96 / 0.96 / 1.0 / 1.0

当前回归：
六项结构指标 1.0
但仅用于 regression after inspection
```

## 六、数据治理是项目最强的部分之一

JobMatchTune 最有面试价值的一条演进不是“模型越训越好”，而是：

```text
最初 Pair Random Split
        ↓
发现 JD / Resume Entity Leakage
        ↓
修复为 Entity-level Split
```

同时还发现：

```text
15k+ Resume SFT rows
≠
15k+ 独立 Resume
```

最终重新按 source_group 统计后：

```text
Resume SFT rows = 15,470
source groups = 2,557
bootstrap source groups = 2,525
bootstrap ratio = 98.75%
```

此外：

```text
Match Pair = 4,798
synthetic = 4,798
human training pair = 0
real observed pair = 0
```

显式经验要求样本：

```text
matched = 0
unmatched = 1,944
```

这些数字并不好看，但正因为项目没有隐藏它们，训练 readiness 才继续保持：

```text
Resume = false
Match = false
Multitask = false
```

这不是失败，而是数据门禁正确工作。

## 七、当前工程完整度

本轮已经进一步补齐：

- 上传文件内容级校验；
- PDF signature；
- DOCX OOXML container 校验；
- 图片真实解码；
- UTF-8 / 控制字符检测；
- 文件大小在 OCR/模型调用前拦截；
- Vue/Vite 本地构建；
- `package-lock.json` 锁定前端；
- CPU / GPU inference / GPU training 分环境 constraints；
- vLLM benchmark harness；
- 517 个本地测试；
- Ruff；
- compileall；
- Shell syntax；
- frontend build / smoke。

vLLM 当前环境没有安装，因此真实 Transformers/vLLM 同硬件 A/B Benchmark 被明确保留为 pending，而不是伪造性能数字。

## 八、实现文档与秋招准备状态

当前已经新增：

```text
docs/秋招项目总结/实现深挖/
```

共 16 篇实现文档和索引，覆盖请求链路、JD Parse、Resume/OCR/Privacy、Normalization、Match/Direction、Skill Canonicalization、数据构建、SourceGroup/EntitySplit、QLoRA/DFT、DPO、Transformers/vLLM、Gold Eval、Readiness、FastAPI、完整 Case、踩坑映射和面试多层追问。

这已经足够作为秋招复习材料，不建议继续为了“更详细”扩成大而不可读的百科文档。

## 九、当前剩余技术债应该如何分类

### 外部条件阻塞

- Resume 独立合法来源不足；
- Match 100% synthetic；
- 年限满足正例缺失；
- 人工 preference 不足；
- 新独立 Gold V2 未完成；
- Score 没有真实招聘结果校准；
- Advice Validity 没有真实投递反馈。

### 部分解决

- OCR Canonicalization 仍然是有限词表；
- Direction taxonomy 覆盖范围有限；
- 非标准 JD section 仍可能导致 required-skill 少召回；
- 前端暂无真实浏览器 E2E；
- 上传校验不是恶意文件沙箱；
- vLLM 真实 A/B 性能报告待环境具备。

### 刻意不解决

当前不建议增加：

- 微服务；
- Redis / Kafka / Celery；
- 账号系统；
- 多租户；
- CI；
- ATS 管理后台；
- 新模型；
- Embedding Matcher；
- 新 DPO；
- 更多 synthetic 数据。

## 十、秋招项目价值

JobMatchTune 最强的项目叙事已经不是：

> “我会 QLoRA、DPO、Qwen。”

而是：

> **我做了一套中文招聘后训练和匹配系统，在不断追求模型效果的过程中发现历史评测污染、实体级数据泄漏、模板膨胀、隐私风险、来源集中和条件分布问题，因此重构了数据划分、Gold、Readiness 和评测体系；最终 Match Product Final 在 25 条项目内人工复核冻结 Gold 上达到 0.96 级结构匹配效果，但由于真实 Resume / Pair 数据仍不足，训练门禁主动阻止了下一轮 SFT/DPO。**

这个故事很适合 AI 应用工程、后训练工程、算法工程、后端 + AI 工程岗位。

## 十一、最终状态

### 当前状态

> **ENGINEERING FREEZE**

### 重新开启开发的条件

只有在获得以下任一新证据时才建议重新打开：

1. 合法、独立、去标识化的 Resume；
2. 人工或真实观察 Match Pair；
3. 真实 human preference；
4. 新独立人工 Gold；
5. 真实招聘 outcome / 用户投递反馈。

否则继续增加代码的价值已经低于学习项目和准备面试的价值。

## 十二、当前最优下一步

JobMatchTune 从现在开始不再是“待开发项目”，而应该变成“待掌握项目”。

秋招准备应该重点训练自己能够脱离文档回答：

- 为什么不让 LLM 直接给 Match？
- 为什么 Normalization 不算作弊？
- 为什么 Pair Random Split 会造成 Entity Leakage？
- 为什么 15,470 条 Resume 不等于 15,470 个独立样本？
- 为什么当前 1.0 regression 不能替代历史 0.96？
- 为什么 85 分不是录用概率？
- 为什么 DPO 暂停？
- 为什么 Readiness=false 是正确结论？
- 为什么职责技术不能自动当 required skill？
- 为什么 Direction 需要 exact / compatible / mismatch？

当这些问题能够自然回答时，这个项目才真正从“Codex 写出来的工程”变成自己的项目。
