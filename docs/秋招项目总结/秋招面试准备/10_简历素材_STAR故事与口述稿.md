# 简历素材、STAR 故事与口述稿

## 一、项目标题

推荐主标题：

> JobMatchTune｜中文招聘场景 LLM 后训练与可解释匹配系统

偏不同岗位可调整副标题：

- 后训练/评测：基于 Qwen3-14B、QLoRA 与可信 Gold 的多任务后训练工作台。
- AI 应用：LLM 结构化抽取、确定性规则和证据约束的人岗匹配系统。
- 数据工程：面向 JD/Resume/Match 的来源治理、实体切分与训练准入流水线。
- AI 后端：支持 Transformers/vLLM、文件 OCR、批量 API 和可观测延迟的推理服务。

## 二、一句话描述

> 设计并实现中文 JD/简历结构化与可解释匹配系统，采用 Qwen3-14B QLoRA、证据约束 normalization 和确定性评分，并通过 SourceGroup、Entity Split、Gold 泄漏审计与 Readiness 门禁治理后训练数据和模型晋级。

## 三、推荐的三条简历 Bullet

### 通用版本

- 构建 JD Parse、Resume Parse 与 Match 三任务流水线，以 Qwen3-14B + QLoRA 完成结构化抽取，用确定性规则处理方向、技能、学历、经验和项目条件，再约束 LLM 生成可审计解释。
- 针对 OCR 断词、职责/要求混淆和交叉岗位误判，实现已知词表有界 canonicalization、四类技能 evidence provenance 与 exact/compatible/mismatch 方向策略；项目内25条冻结 Gold 的历史匹配等级/方向 EM 均为0.96。
- 审计并修复 Pair 随机切分导致的 JD/Resume 实体泄漏，建立 SourceGroup、Entity-level Split、隐私/许可和训练 Readiness 门禁；识别15,470条 Resume 的98.75%来源组为 bootstrap、4,798个 Match Pair 全合成，并主动暂停证据不足的新 SFT/DPO。

### AI 应用/后端补充 Bullet

- 使用 FastAPI 编排文本、PDF、DOCX、图片/OCR、单条和批量请求，按 Transformers/vLLM 后端选择串行锁或受控并发；补充文件内容签名校验、统一启停、Vue/Vite 锁定构建及五类 workload benchmark harness。

### 后训练/评测补充 Bullet

- 基于 TRL/PEFT 实现4-bit NF4 QLoRA、assistant-only loss、DFT/NLL 对照与 DPO，使用 run manifest 固化 Git/配置/数据 hash，并将 Raw Model、Normalized、Product Final 分层评测，避免把工程后处理收益错误归因给模型。

## 四、数字使用说明

简历空间有限，可以写0.96，但必须在面试中补全：

> 项目内25条单人复核冻结 Match Gold，Product Final 的匹配等级和方向 exact match 为0.96；不是裸模型准确率，也不是外部业务 benchmark。

不建议把当前规则回归1.0写进简历，因为它是 `REGRESSION AFTER INSPECTION`。不建议写“15k真实简历”或“4,798真实人岗 Pair”。

## 五、STAR 故事一：主动废弃漂亮但污染的评测

### Situation

项目已有64条 Match 评测和不错的最终指标，准备作为主要效果结论。

### Task

确认这组结果是否真正独立，能否支持 adapter 晋级和秋招表述。

### Action

我从样本 identity 而不是文件名入手，追踪基础样例、格式变体和训练来源，发现64条主要来自16个基础 Pair，且基础样例进入过训练池。我没有修饰历史结论，而是将该结果标记失效；重新建立25条人工复核集合，增加 Pair/JD/Resume/单任务五类 exact hash 隔离、difficulty tags 和冻结 hash。

### Result

形成可追溯 Gold V1，历史 Product Final 等级和方向 EM 0.96；后续修复使用已看错例时，报告明确降级为 regression after inspection。

### Reflection

评测独立性必须进入数据结构和自动审计，不能依赖“holdout”文件名。项目可信度比保留一个高分更重要。

## 六、STAR 故事二：从行数幻觉到 SourceGroup/Entity Split

### Situation

Resume SFT 有上万行、Match 也有数千 Pair，表面满足训练规模。

### Task

判断数据是否真正覆盖独立简历和人岗关系，避免多任务训练过拟合模板和实体。

### Action

我为样本引入 `source_group`，统计每个原始实体的模板扩展；多任务采样改为 group round-robin。针对 Match Pair，把随机边切分改成 JD/Resume 两侧实体先切分、split 内配对，并保存两侧 entity hash。

### Result

审计发现15,470条 Resume 仅2,557个 group，98.75%的 group 来自 bootstrap；4,798个 Match Pair 全为规则合成，年限样本又只有负例。Readiness 因此保持 false 并阻止新训练。

### Reflection

数据量应同时从 row、source/entity 和 label provenance 三层度量。门禁能阻止一场可运行但无新增证据的实验。

## 七、STAR 故事三：在召回和误报间设计 OCR 规则

### Situation

Gold 暴露 `Py test` 未恢复，直接影响技能命中和匹配等级。

### Task

提高 OCR 技能召回，同时避免全局 fuzzy match 把普通文本或短词识别成技能。

### Action

我没有添加 `Py test` 单条 alias，而是在 known vocabulary 内构建 exact 和唯一 compact lookup；原文查找用带 ASCII 边界的 OCR separator pattern，短技能禁用字符间 fuzzy。补充 Py thon、Kubernet es、C + +、Node . js 正例和普通英文、pytest-like、短词反例。全量测试又发现 `C` 从 C++/C#/C语言重复抽取，我补了 standalone C 通用边界。

### Result

已知 OCR 断词得到通用恢复，假阳性契约保留；全量517个测试通过。

### Reflection

招聘硬条件更看重 precision 和可解释性，规则扩展必须同时有正例与反例，不能围绕 Gold ID 特判。

## 八、STAR 故事四：并行不是开两个线程

### Situation

一次 Match 需要先解析 JD 和 Resume，两者逻辑独立，但默认延迟接近两次生成之和。

### Task

在不破坏单模型显存安全的情况下支持并行和批量。

### Action

我把任务 DAG 与执行后端分开。Transformers 共享14B实例，用 Lock 串行生成；vLLM 路径通过 OpenAI-compatible server 和 BoundedSemaphore 受控并发，只有并发槽至少2时并行两侧解析。健康接口暴露实际 parse mode，benchmark harness 对同一输入运行五类 workload。

### Result

架构支持后端特定并发，同时默认路径保持安全。当前环境没安装 vLLM，因此没有制造性能数字，真实 A/B 保持 pending。

### Reflection

逻辑可并行不代表资源可并行；优化必须同时考虑共享状态、显存和可测量证据。

## 九、1分钟口述稿

> JobMatchTune 是我做的中文招聘 LLM 后训练与可解释匹配系统。用户输入 JD 和简历后，Qwen3-14B 分别抽取结构字段；中间层做技能 OCR 归一、JD 要求证据和岗位方向兼容；规则引擎再计算方向、技能、学历、经验、项目五个分项；最后模型只根据规则事实生成解释。项目后期我发现真正难点不是模型，而是数据和评测：同一简历的模板被当成独立数据，Pair 随机切分导致实体泄漏，旧 Match 集还进入过训练。因此我重构了 SourceGroup、Entity Split、Gold 审计和 Readiness。历史25条项目内冻结 Gold 上最终等级和方向 EM 是0.96，但当前 Resume/Match 数据仍主要是 bootstrap 和 synthetic，所以我让门禁阻止了继续训练。这段经历让我从追求模型分数转向构建可验证的 AI 系统。

## 十、3分钟口述稿提词卡

不要背全文，只记五个词：

1. **分层**：Parse → Normalize → Rule → Explain。
2. **证据**：Requirement provenance、bounded OCR、direction compatibility。
3. **泄漏**：SourceGroup、Entity Split、旧 holdout 失效。
4. **评测**：Raw/Normalized/Product、0.96历史、1.0 after inspection、grounding 0.92。
5. **停止**：Resume/Match readiness=false，不再训练。

## 十一、按岗位改写项目重点

### AI 应用工程

多讲模型和规则的职责边界、结构化输出、文件链路、解释约束；少讲所有 crawler 细节。

### 后训练/算法

多讲 messages、assistant-only loss、QLoRA、DFT/NLL、DPO 数据、三层评测和 Gold；主动承认算法创新有限。

### 数据工程

多讲 registry、provenance、source group、entity graph、freshness、readiness 和隐私；把训练当数据消费者。

### 后端/推理

多讲 ModelService 生命周期、Lock、vLLM semaphore、批量、上传验证、timings、启动/停止和故障边界。

## 十二、面试结尾

> 如果继续做，我不会先换更大模型或增加中间件，而会优先获取合法独立 Resume、人工 Match Pair 和 Blind Gold V2。只有这些外部证据改变，新的训练和更复杂 scorer 才有意义。当前版本选择工程冻结，是因为核心问题已经从代码能力转成数据可得性。

这句话能把项目局限转化为工程判断，但前提是前面确实讲清了现有实现。
