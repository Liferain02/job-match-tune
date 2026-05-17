# Optimization Against Open Source (2026-05-15)

## 背景

这次优化的目标不是简单“追新”，而是对照当前主流开源项目和官方训练文档，筛选出适合这个仓库当前阶段的技术。

评估标准：

1. 能否在当前 `3 x L20` 资源下落地
2. 能否和现有 `Qwen3-14B + LoRA + FastAPI/vLLM` 链路兼容
3. 是否真的能提升项目的工程完整度或简历表达
4. 是否已经有稳定的官方 / 主流开源实现

## 参考的开源项目与官方文档

### 1. vLLM

参考：

- vLLM Structured Outputs: <https://docs.vllm.ai/en/latest/features/structured_outputs/>
- vLLM LoRA Adapters: <https://docs.vllm.ai/en/stable/features/lora.html>

启发点：

- 支持 `structured_outputs`
- 支持 `LoRA Adapters`
- 支持 OpenAI-compatible serving

已经吸收：

- 新增 `vLLM + OpenAI-compatible API` 后端
- 新增 `JSON Schema structured outputs`
- 保留 `LoRA` 适配器服务形式

### 2. TRL

参考：

- TRL index: <https://huggingface.co/docs/trl/index>
- DPOTrainer: <https://huggingface.co/docs/trl/dpo_trainer>
- ORPOTrainer: <https://huggingface.co/docs/trl/orpo_trainer>

启发点：

- 当前主流后训练不只做 SFT
- 偏好优化是更完整的项目叙事
- `DPO / ORPO / GRPO` 都是主流路径，但资源要求和成熟度不同

已经吸收：

- 增加 `DPO` 训练入口
- 增加 preference dataset 构造入口

当前没有直接上：

- `ORPO`
- `GRPO`

原因：

- 当前环境里的 `trl==1.4.0` 直接可用的是 `DPOTrainer` 和 `GRPOTrainer`
- `ORPOTrainer` 当前环境不可直接使用
- `GRPO` 对数据、奖励设计和算力要求更高，不是当前第一优先级

### 3. Qwen 官方训练文档

参考：

- Qwen + Unsloth: <https://qwen.readthedocs.io/en/latest/training/unsloth.html>
- Qwen + Axolotl: <https://qwen.readthedocs.io/en/latest/training/axolotl.html>

启发点：

- Qwen 官方已经把 `Unsloth / Axolotl` 作为可行训练栈纳入文档
- 长上下文、LoRA / QLoRA、RLHF、部署一体化已经是官方支持叙事的一部分

当前选择：

- 不直接切换训练栈

原因：

- 当前仓库已经稳定在 `Transformers + PEFT + TRL`
- 直接迁移到 Unsloth / Axolotl 会引入较大结构变化
- 在现阶段，先把 `DPO + vLLM + 14B 主链路` 做稳，收益更高

### 4. LLaMA-Factory

参考：

- <https://github.com/hiyouga/LLaMA-Factory>

启发点：

- 训练、评估、推理、部署一体化
- 高频使用 `DPO / ORPO / DoRA / PiSSA / FSDP+QLoRA / vLLM`
- 对历史实验与当前主链路分层更清楚

已经吸收：

- 把 1.7B 时代实验脚本归档到 `scripts/legacy/`
- 主目录只保留当前主链路脚本
- 补上 `DPO` 偏好优化入口

当前没直接吸收：

- `DoRA`
- `PiSSA`
- `FSDP+QLoRA`

原因：

- 当前项目规模还没逼到这些优化是第一瓶颈
- 先把结构化抽取质量、偏好优化和服务链路做好更重要

## 这次具体做了什么

### 1. 主链路聚焦到 14B

新增：

- `configs/train_qwen3_14b_qlora.yaml`

作用：

- 明确当前默认训练配置
- 不再让 14B 主链路继续依赖早期 1.7B 试验配置

### 2. 归档历史实验脚本

归档到：

- `scripts/legacy/`

包括：

- `run_modern_training_experiments.sh`
- `run_three_gpu_experiments.sh`
- `train_direction_incremental.sh`
- `train_qlora_full.sh`
- `train_qlora_smoke.sh`

作用：

- 当前 `scripts/` 目录更干净
- 新用户不会把 1.7B 历史实验误当当前推荐入口

### 3. 新增 DPO 偏好优化链路

新增：

- `configs/train_qwen3_14b_dpo.yaml`
- `src/jobmatch_tune/dataset/build_preference_dataset.py`
- `src/jobmatch_tune/train/train_dpo.py`
- `scripts/data/build_preference_dataset.sh`
- `scripts/train/train_qwen3_14b_dpo.sh`

作用：

- 项目从 “SFT-only” 升级到 “SFT + preference tuning”
- 更符合现在主流开源项目的后训练实践

## 当前推荐技术路线

### 默认路线

1. 公开 JD 抓取
2. 清洗和弱标注
3. `Qwen3-14B` QLoRA SFT
4. 人工 holdout 评估
5. 规则后处理收敛
6. `vLLM + structured outputs` 服务部署

### 下一条优先升级路线

1. 用人工评估与错例构造 preference dataset
2. 跑 `DPO`
3. 比较：
   - 字段级准确率
   - 延迟
   - 误报类型

## 不建议现在做的事情

- 直接迁移到另一个大训练框架
- 直接上 `GRPO`
- 为了“简历更花哨”就接入 `DoRA / PiSSA / FSDP+QLoRA`

这些方向不是没价值，而是当前收益不如：

- 把 14B 主链路稳定下来
- 把 preference tuning 跑通
- 把服务侧性能和可维护性再提升一层
