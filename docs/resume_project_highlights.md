# Resume Project Highlights

## 建议写法

不要把项目写成“做了一个微调 demo”。更好的写法是：

- 目标明确
- 数据链路完整
- 训练与部署分开描述
- 评估指标可量化
- 只写实际做过的技术

## 当前项目可以写的技术点

### 训练侧

- 基于 `Qwen3-14B` 做 `4-bit QLoRA` 微调，适配低显存多卡环境
- 使用 `TRL SFTTrainer`，支持 `assistant-only loss`、`DFT loss`、`Liger Kernel` 对照实验
- 在 `3 x NVIDIA L20` 集群上完成 smoke、全量训练与人工评估闭环

### 数据与评估

- 构建公开招聘 JD 抓取、清洗、去重、结构标注、SFT 样本生成链路
- 建立 `50` 条人工 gold holdout，并按 `岗位方向 / 职责 / 技能 / 经验 / 学历` 做字段级评估
- 通过 hard case 和规则后处理修正方向边界与技能误报

### 推理与服务

- 同时支持 `Transformers` 本地推理和 `vLLM` OpenAI-compatible 推理后端
- 使用 `JSON Schema structured outputs` 约束结构化抽取结果
- 提供 `FastAPI + 静态前端` 的前后端分离 demo

## 一段适合简历的项目描述

### 版本一：偏训练平台

基于 Qwen3-14B 构建招聘 JD / 简历结构化抽取系统，在 3 x L20 集群上完成 4-bit QLoRA 微调、字段级人工评估和推理服务部署；实现公开 JD 抓取、清洗去重、SFT 数据构造、50 条人工 gold 评测集、FastAPI / vLLM 双推理后端和 JSON Schema 结构化输出，最终在 holdout 集上将岗位方向、职责、技能、经验、学历等核心字段评估指标稳定到 1.0。

### 版本二：偏应用工程

设计并落地一个面向招聘文本解析的 LLM 应用工程项目，基于 Qwen3-14B + QLoRA 完成低显存微调，结合规则后处理、人工 hard case、字段级评估和 vLLM 结构化部署，交付可直接演示的前后端分离系统。

## 子弹点写法

- 基于 `Qwen3-14B` 完成 `4-bit QLoRA` 微调，在 `3 x L20` 集群上跑通 smoke、全量训练和人工评估闭环
- 构建 JD 抓取、清洗、去重、SFT 数据生成和 `50` 条人工 gold 评测集，建立字段级评估体系
- 实现 `Transformers` 与 `vLLM` 双推理后端，支持 `OpenAI-compatible API` 与 `JSON Schema structured outputs`
- 通过规则后处理和 hard case 修正岗位方向与技能字段边界，显著降低结构化抽取误报

## 适合补进项目里的新技术

这些是业界常见、适合写到简历上的方向，但要区分“已实现”和“下一步计划”。

### 已实现

- QLoRA
- DFT loss
- Liger Kernel 对照实验
- FastAPI
- vLLM 结构化推理
- JSON Schema structured outputs
- 人工 gold holdout

### 下一步更值得继续做

- 偏好优化：`DPO / ORPO`
- 更进一步的在线策略优化：`GRPO`
- 更高吞吐部署：`speculative decoding / prefix caching / multi-LoRA serving`
- 更正式的训练栈：`FSDP / DeepSpeed ZeRO`
