# Project Roadmap

## 1. 项目定位

这个项目不应该包装成通用问答系统。

更准确的定位是：

**JobMatchTune：面向招聘场景的结构化抽取与人岗匹配大模型系统**

最终目标不是“回答一个问题”，而是做成一套完整的招聘智能分析系统，覆盖：

1. `JD 解析`
2. `简历解析`
3. `JD-简历匹配度分析`
4. `结构化评估`
5. `服务化 API`
6. `前端工作台`

---

## 2. 最终产品形态

### 2.1 JD 解析

输入一段 JD，输出结构化结果：

- 岗位方向
- 核心职责
- 必备技能
- 加分项
- 经验要求
- 学历要求

### 2.2 简历解析

输入一份简历，输出候选人画像：

- 目标岗位
- 教育背景
- 核心技能
- 实习经历
- 项目经历
- 优势标签

### 2.3 人岗匹配

输入一份 JD 和一份简历，输出：

- 匹配分数
- 匹配等级
- 匹配优势
- 主要短板
- 技能缺口
- 简历优化建议

### 2.4 服务化与工作台

支持：

- Web 前端使用
- API 调用
- 模型预热与状态查看
- 版本评估与对比

---

## 3. 当前状态

### 已成型

- 公开 JD 抓取
- JD 清洗、去重、规则弱标注
- JD 主链路 SFT
- `Qwen3-14B + 4-bit QLoRA`
- DPO 链路
- FastAPI / vLLM / 前端 demo

### 半成型

- `resume_parse`
  - prompt 已有
  - schema 已有
  - API 入口已有
  - 已补人工评估集种子和通用人工评估脚本
  - 但还没有成熟训练数据链路

### 初步成型

- `match` task
- rule-based 匹配引擎
- `POST /api/match`
- 前端匹配工作台
- `match` 人工评估入口

### 未成型

- `resume_parse` 专用训练集
- 更大规模的 `match` 评估集
- `match` 偏好训练数据

---

## 4. 总体技术栈

### 模型与训练

- `Qwen3-14B`
- `Transformers`
- `PEFT`
- `TRL`
- `Accelerate`
- `4-bit QLoRA`
- `DPO`

### 推理与服务

- `FastAPI`
- `vLLM`
- `OpenAI-compatible API`
- `Pydantic`
- `JSON Schema structured outputs`

### 数据工程

- `requests`
- `BeautifulSoup`
- `SQLite`
- `PyYAML`
- `datasketch`
- 自研 crawler / preprocess / dataset pipeline

### 工程验证

- `pytest`
- `compileall`

---

## 5. 分阶段路线

---

## 阶段 0：收口现状

### 目标

- 冻结主线
- 明确哪些模块已经稳定，哪些模块还只是接口骨架

### 主要工作

1. 明确项目不是问答系统，而是招聘智能解析与匹配系统
2. 冻结核心任务：
   - `jd_parse`
   - `resume_parse`
   - `match`
3. 明确当前默认模型与默认数据口径

### 交付物

- 项目边界文档
- 当前默认方案说明

---

## 阶段 1：按任务拆数据和接口

### 目标

- 不再只维护一个 `data/sft/`
- 让训练、评估和 API 与任务边界一致

### 主要工作

1. 拆数据目录：
   - `data/sft_jd/`
   - `data/sft_resume/`
   - `data/sft_match/`
   - `data/sft_tasks/skill_extract/`
2. 拆 schema：
   - JD schema
   - Resume schema
   - Match schema
3. 拆 API task：
   - `jd_parse`
   - `resume_parse`
   - `match`

### 技术栈

- `Pydantic`
- `FastAPI`
- 现有 `templates.py`
- 现有 `structured_output.py`

### 交付物

- 清晰的任务边界
- 更可维护的训练与服务结构

---

## 阶段 2：做强 JD 解析主链路

### 目标

- 让 JD 解析持续作为最稳定的基础能力
- 提升默认高质量集质量，而不是盲目追大样本量

### 主要工作

1. 继续扩高信任中文官网源
   - 腾讯
   - 百度
   - 京东
   - Moka
2. 持续扩合理的岗位方向类
3. 持续优化：
   - 标题准入规则
   - 方向规则
   - 字段完整性
   - 近重复去重
4. 扩 JD 人工评估集

### 技术栈

- 自研 crawler
- SQLite
- 规则抽取
- 近重复去重
- `Qwen3-14B + QLoRA + DPO`

### 交付物

- 更稳定的 `data/sft_jd/`
- 更大的 JD gold eval

---

## 阶段 3：补 resume_parse 数据链路

### 目标

- 把 `resume_parse` 从“有接口”变成“有训练和评估”

### 主要工作

1. 定义 resume schema：
   - 目标岗位
   - 教育背景
   - 核心技能
   - 实习经历
   - 项目经历
   - 优势标签
2. 建立 `resume_ingest -> resume_clean` 数据链路
   - 先支持 `txt / docx / pdf`
   - 图片和扫描件先标记 `needs_ocr`
3. 构建 `data/sft_resume/`
4. 构建 resume 人工评估集

### 数据来源原则

- 优先人工构造或合法脱敏样本
- 不直接引入来路不明的真实简历 dump
- 可参考 Resume NER / 简历解析 benchmark

### 技术栈

- `Pydantic`
- 规则抽取
- 人工标注
- 多任务 SFT

### 交付物

- `resume_parse` 可训练数据
- `resume_parse` 评估集

---

## 阶段 4：做 rule-based 匹配基线

### 目标

- 先做一个可解释、可评估的人岗匹配 baseline

### 主要工作

1. 输入：
   - JD 解析结果
   - 简历解析结果
2. 输出：
   - 匹配分数
   - 匹配等级
   - 命中技能
   - 缺失技能
   - 命中项目
   - 学历匹配
   - 经验匹配
3. 评分规则：
   - 技能 overlap
   - 岗位方向一致性
   - 学历门槛
   - 经验门槛
   - 项目关键词命中

### 技术栈

- Python rule engine
- `schemas.py` 中的 `MatchRuleResult`
- 技能 taxonomy / alias 词表

### 交付物

- 可解释的规则匹配引擎
- 匹配基线输出 JSON

---

## 阶段 5：实现 match task

### 目标

- 在规则基线之上，加入模型生成的匹配分析层

### 主要工作

1. 扩 `match` task：
   - prompt
   - schema
   - structured output
   - predict
   - API
2. 输入：
   - JD 文本
   - 简历文本
   - rule result
3. 输出：
   - 匹配结论
   - 匹配优势
   - 主要短板
   - 简历优化建议
   - 推荐岗位方向

### 技术栈

- `FastAPI`
- `vLLM`
- `Transformers`
- `Pydantic`
- `JSON Schema outputs`

### 交付物

- `/api/match`
- 单条匹配分析能力

---

## 阶段 6：做多任务训练

### 目标

- 让模型同时支持 JD、简历、匹配三类任务

### 主要工作

1. 训练任务组成：
   - `jd_parse`
   - `resume_parse`
   - `match`
   - `skill_extract`
2. 训练顺序：
   - 先把 `jd_parse` 和 `resume_parse` 单任务做稳
   - 再联合 SFT
   - 再对 `match` 做 DPO

### 技术栈

- `Qwen3-14B`
- `4-bit QLoRA`
- `TRL SFTTrainer`
- `DPOTrainer`

### 交付物

- 面向产品的多任务模型

---

## 阶段 7：建立评估体系

### 目标

- 让版本迭代不只看 loss，而是看产品质量

### 评估拆分

#### JD 解析评估

- 岗位方向 exact match
- 职责/技能/加分项 F1
- 学历/经验 exact match

#### 简历解析评估

- 目标岗位
- 技能抽取
- 项目/实习抽取

#### 匹配评估

- 分数合理性
- 技能缺口正确率
- 建议可用性

#### 系统评估

- JSON 合法率
- 推理延迟
- 稳定性

### 技术栈

- 自定义 metrics
- 人工 gold eval
- source holdout
- boundary eval

### 交付物

- 多模块评估报告
- 版本对比面板

---

## 阶段 8：做产品前端

### 目标

- 从 demo 页升级成真正的工作台

### 页面结构

1. JD 解析
2. 简历解析
3. 人岗匹配
4. 模型评估 / 版本对比

### 每页要有的能力

- 输入区
- 结构化结果
- 原始 JSON
- 证据/命中项展示
- 复制/导出

### 技术栈

- 当前静态前端可继续迭代
- 如有必要可升级到：
  - `React + Vite`
  - 更完整的组件体系

### 交付物

- 更产品化的前端体验

---

## 阶段 9：服务化和批处理

### 目标

- 不只支持单次页面交互，也支持系统集成

### 主要工作

1. 扩 API：
   - `/api/parse`
   - `/api/match`
   - `/api/batch_parse`
   - `/api/batch_match`
2. 支持批量处理
3. 支持 JSON / CSV 导出
4. 增加更完整的错误处理和日志

### 技术栈

- `FastAPI`
- `vLLM`
- `OpenAI-compatible API`

### 交付物

- 面向开发者可调用的服务化接口

---

## 阶段 10：项目包装与展示

### 目标

- 让项目不仅能跑，还能被清晰讲出来

### 推荐项目标题

**JobMatchTune：面向招聘场景的结构化抽取与人岗匹配大模型系统**

### 推荐描述

基于 `Qwen3-14B` 构建招聘 JD / 简历结构化解析与人岗匹配系统，完成公开 JD 数据采集、清洗弱标注、近重复去重、4-bit QLoRA 微调、DPO 偏好优化、字段级人工评估、规则后处理和 `FastAPI / vLLM` 服务化部署，支持 JD 解析、简历解析、匹配评分、缺口分析与简历优化建议。

### 交付物

- 简历项目描述
- 演示稿
- 指标看板

---

## 6. 最小可行落地顺序

如果目标是最短路径把项目做成，建议顺序如下：

1. 补 `resume_parse` 数据链路
2. 做 rule-based 匹配基线
3. 增加 `match` task 的 schema / API / prompt
4. 做 `resume_parse` 和 `match` 的人工评估集
5. 再做多任务训练
6. 最后补完整前端

---

## 7. 当前下一步建议

基于当前仓库状态，下一步最应该做的不是继续只扩 JD 数据，而是：

1. **实现 `match` task**
2. **实现 rule-based 匹配引擎**
3. **开始 `resume_parse` 数据链路**

这三步做完，项目才会从“JD 微调项目”真正升级成“招聘智能解析与人岗匹配系统”。
