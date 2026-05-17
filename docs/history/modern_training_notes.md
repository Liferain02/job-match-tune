# 新训练技术落地说明

更新时间：2026-05-10

## 已接入

1. Packing

TRL `SFTTrainer` 支持将多个短样本打包进同一上下文窗口，减少 padding 浪费。当前 JD 抽取样本普遍较短，packing 能提高 token 利用率。

当前环境只验证到 SDPA attention。TRL 在 SDPA + packing 下会提示潜在样本串扰风险，因此三卡默认实验不启用 packing。后续安装并验证 FlashAttention 2/3 后再打开。

2. Assistant-only loss

训练只对 assistant 输出计算 loss，避免模型学习 system/user prompt。本项目默认开启，适合结构化抽取任务。

3. DFT loss

TRL `SFTConfig.loss_type` 支持 `dft`。这是可实验项，不建议直接替代主基线，需要通过验证集和人工评估比较。

4. Liger Kernel

Liger Kernel 通过 Triton kernel 优化训练吞吐和显存占用。TRL 已支持 `use_liger_kernel=True`，本项目作为三卡实验中的一个对照组接入。

5. SDPA attention

默认配置 `attn_implementation: sdpa`，使用 PyTorch 内置 scaled dot-product attention。FlashAttention 需要额外安装和兼容性验证，暂不默认启用。

## 三卡实验

```bash
ssh gpu03
source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo
cd /share/home/lifr/workspace/code/job-match-tune
bash scripts/legacy/run_modern_training_experiments.sh
```

三组实验：

1. `qwen3-1.7b-sdpa-lr1e-4`：SDPA + NLL 基线。
2. `qwen3-1.7b-dft-lr1e-4`：SDPA + DFT。
3. `qwen3-1.7b-liger-lr1e-4`：SDPA + Liger Kernel。

## 判断标准

不要只看训练 loss。优先看：

1. JSON 合法率。
2. `核心职责`、`任职要求`、`必备技能` 是否分离正确。
3. 推理是否稳定，不出现重复、漏字段或把整段文本塞进一个字段。
4. 验证集 loss 是否低但人工抽查质量下降，若出现说明数据模板或标签质量有问题。

## 下一阶段

优先补人工评估集，而不是继续扩大训练轮数。建议从 `data/sft/test.jsonl` 抽 50 条，人工修正输出 JSON，用于比较不同 checkpoint。
