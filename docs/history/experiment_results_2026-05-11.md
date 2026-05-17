# 实验结果 2026-05-11

## 环境

训练节点：`gpu03`

GPU：3 x NVIDIA L20, 46068 MiB

基础模型：`models/Qwen3-1.7B`

训练数据：

1. `data/sft/train.jsonl`：709 条。
2. `data/sft/valid.jsonl`：88 条。
3. `data/sft/test.jsonl`：90 条。

## 三卡并行实验

运行命令：

```bash
bash scripts/legacy/run_modern_training_experiments.sh
```

| 实验 | 训练方式 | train_loss | eval_loss | train_runtime | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `qwen3-1.7b-sdpa-lr1e-4` | SDPA + NLL | 0.1157 | 0.01907 | 482.6s | loss 正常，但抽查字段分离较差 |
| `qwen3-1.7b-dft-lr1e-4` | SDPA + DFT | 0.001222 | 0.000629 | 494.1s | 抽查结果最好，作为下一轮候选 |
| `qwen3-1.7b-liger-lr1e-4` | SDPA + Liger Kernel | 0.1175 | 0.01944 | 610.4s | 本环境下更慢，抽查字段分离较差 |

## 推理抽查

输入：`examples/jd_ai_app.txt`

### SDPA + NLL

结果能输出合法 JSON，但把 `任职要求` 和 `加分项` 原文混入 `核心职责`，`必备技能` 为空。

### SDPA + DFT

结果能输出合法 JSON，且能把 `核心职责`、`必备技能`、`加分项` 分开。

已观察到的问题：

1. `学历要求` 仍为空，没有从文本中抽出“本科及以上”。

已修复的后处理问题：

1. `岗位方向` 输出为 `AI开发` 时，归一化为 `AI应用开发`。
2. 列表字段自动去重，避免 `加分项` 重复句子。

### SDPA + Liger Kernel

结果与普通 SDPA + NLL 类似，仍把要求和加分项混入 `核心职责`。

## 当前推荐

短期推荐 checkpoint：

```text
outputs/checkpoints/qwen3-1.7b-dft-lr1e-4
```

原因：单条人工抽查中，DFT 对结构边界更敏感。注意 DFT 的 loss 数值不能和 NLL 直接横向比较，后续必须用人工评估集确认。

## 下一步

1. 建 30-50 条人工评估集，覆盖 JD 解析的常见格式。
2. 增强学历、经验等规则后处理。
3. 扩展评估脚本，从 JSON 合法率升级为字段级准确率。
4. 暂不默认启用 packing。当前 SDPA + packing 会触发 TRL 的样本串扰风险提示，等 FlashAttention 2/3 验证后再启用。
