# QLoRA、SFT 与 DFT 从张量到配置逐项推导

这一章回答四个问题：模型究竟训练了哪些参数，显存为什么能降下来，一个样本如何变成 loss，以及配置里的 `dft` 到底是什么。

## 1. 训练入口实际执行顺序

`train_lora.py` 的主线是：

```text
读取 YAML + CLI 覆盖
  → 写 run manifest
  → 加载 tokenizer
  → 构造 4-bit quantization config
  → 加载 base causal LM
  → 新建 LoRA 或载入已有 adapter
  → 加载 train/validation JSONL
  → chat template 格式化
  → SFTConfig
  → SFTTrainer.train
  → evaluate + 保存 adapter/state/metrics
```

重依赖放在 `main()` 内部导入，使数据脚本和单元测试不必安装完整 GPU 训练栈。

## 2. SFT 样本如何进入语言模型

原始样本是 messages：

```text
system: 你是结构化抽取助手
user:   解析以下 JD……
assistant: {标准 JSON}
```

`formatting_func` 调用 Qwen chat template，把 role、控制 token 和文本串成一个序列。设 token 为：

```text
x1, x2, ..., xT
```

因果语言模型在位置 `t` 根据前缀预测下一个 token：

```text
pθ(x_t | x_<t)
```

普通 NLL 是：

```text
L_NLL = - (1/N) Σ_t m_t log pθ(x_t | x_<t)
```

`m_t` 是 loss mask。项目设置 `assistant_only_loss=True`，目标是只让 assistant JSON token 贡献梯度，system/user token 只作为条件上下文。

## 3. assistant-only loss 为什么重要

若对整个序列计算 loss，模型还会学习复述固定 system prompt 和用户输入，浪费容量；结构化任务真正需要优化的是给定输入后的 JSON 输出。

但该开关依赖 chat template 能正确标记 generation 区域。Qwen3 在新版 TRL 中有支持；训练启动前仍应做一个 token mask smoke test，确认非 assistant token 的 label 是 `-100`。否则配置写了 true 也不代表掩码一定正确。

## 4. LoRA 的矩阵含义

对一个冻结线性层权重：

```text
W ∈ R^(d_out × d_in)
```

不直接更新 W，而学习两个低秩矩阵：

```text
A ∈ R^(r × d_in)
B ∈ R^(d_out × r)
ΔW = (α/r) BA
W' = W + ΔW
```

项目配置 `r=8, alpha=16`，典型缩放因子是 2。实际缩放细节由 PEFT 版本实现决定。

单个线性层新增参数约为：

```text
r(d_in + d_out)
```

而全量微调参数是 `d_in × d_out`。当 r 远小于维度时，训练参数和优化器状态显著下降。

## 5. 为什么 target modules 不只选 attention

项目目标层：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

前四项覆盖 attention 投影，后三项覆盖 MLP。结构化抽取既需要注意输入字段，也需要调整生成分布；覆盖 MLP 通常比只改 q/v 表达能力更强，代价是 adapter 参数更多。

当前没有通过消融证明“七层目标优于只 q/v”，所以面试中应说这是常见工程配置，不是项目独有结论。

## 6. QLoRA 中 4-bit 到底量化了谁

```text
基础模型权重：4-bit NF4 存储
矩阵计算：bf16 compute dtype
LoRA 参数：可训练的较高精度参数
优化器状态：只为可训练参数维护
```

NF4 适合近似正态分布的模型权重；double quantization 继续压缩量化尺度等元数据。关键点是“4-bit 存储”不等于所有算术都用 4-bit，前向计算会反量化到设定的计算精度。

QLoRA 的主要收益是把 14B 基座放进有限显存，并只训练 adapter；它不能保证质量等同全参训练，需要通过同一 Gold 比较。

## 7. 当前关键配置逐项解释

| 配置 | 当前值 | 实际作用 |
|---|---:|---|
| `max_seq_length` | 768 | 超长输入会截断；可能丢掉 JD 尾部要求或简历后部项目 |
| micro batch | 1 | 每设备每步一条，降低峰值显存 |
| gradient accumulation | 16 | 累积 16 个微步再更新 |
| epoch | 1 | 降低小数据重复拟合风险 |
| learning rate | 1e-4 | LoRA SFT 常见量级，但需实验验证 |
| warmup | 20 steps | 初期逐渐升高学习率 |
| scheduler | cosine | warmup 后余弦衰减 |
| bf16 | true | 比 fp16 动态范围大，需要硬件支持 |
| checkpointing | true | 用重算换激活显存 |
| packing | false | 不把多个短样本塞进同一序列 |
| eval max | 320 | 限制验证计算成本 |

单进程单设备时有效更新 batch 约为 `1 × 16 = 16` 条。若使用多进程 DDP，还要乘数据并行进程数；当前脚本使用 `device_map=auto`，不能仅凭 GPU 数量宣称全局 batch。

## 8. 为什么 packing 默认关闭

Packing 能减少短样本 padding，提高 token 利用率。但当前是对话结构化输出，并启用 assistant-only mask；在未验证模板边界与 mask 正确前，关闭 packing 更保守。

只有在长度分布显示大量短样本、单元测试确认跨样本边界和 assistant mask 正确，并有吞吐基准时再开启。不能把所有显存优化开关同时打开后再猜是哪项导致质量变化。

## 9. gradient checkpointing 的代价

普通反向传播保存中间激活；checkpointing 只保留部分节点，反向时重算前向片段：

```text
显存下降
计算量和墙钟时间上升
```

它不减少模型权重内存，和 4-bit、LoRA 解决的是不同部分。

## 10. DFT 不是 DPO，也不是项目自创缩写

配置 `loss_type: dft` 指 TRL 的 Dynamic Fine-Tuning，是 SFT 阶段的替代 loss；它不是 Direct Preference Optimization。

按当前 TRL 官方实现，单 token 的形式可写为：

```text
p_t = pθ(y_t | x, y_<t)
L_DFT,t = - stopgrad(p_t) · log p_t
```

代码中通过 `logprobs.exp().detach()` 得到不传梯度的 token 概率权重。直观上，它改变不同 token 对 loss 的动态权重。完整理论应以论文为准，项目只是在 Trainer 配置中启用实现。

参考：TRL 的 [SFT 配置与 DFT 实现](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py)和[官方 SFT 文档](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md)。

## 11. 一个必须主动说明的版本风险

仓库当前训练约束是 `trl>=0.10,<2`，范围很宽；当前 CPU 环境甚至未安装 TRL，无法在这里验证实际训练节点版本。`loss_type="dft"` 是较新的接口，旧版 TRL 可能不接受该参数或语义不同。

因此可信表达是：

> 配置意图是使用 TRL 的 DFT loss，历史 checkpoint 名称也按 DFT 记录；正式复现前必须在 GPU 环境冻结 `pip freeze`，运行 config constructor smoke test，并把实际 TRL 版本写入 run manifest。

不能只因为 YAML 写了 dft 就断言每次历史训练都确实使用了同一实现。

## 12. `run_manifest` 为什么重要

训练开始前记录：

- 阶段和配置路径；
- 输出目录；
- train/valid 文件信息；
- CLI 覆盖项；
- 数据行数、任务分布、source group 数等摘要。

它解决“checkpoint 到底由哪份数据和参数产生”的追溯问题。但若依赖版本、git commit、GPU 信息没完整落盘，仍不能做到严格可复现；这正是当前可继续补强的地方。

## 13. 截断是比 batch 更容易忽略的风险

`max_seq_length=768` 会在 token 化后截断。长 JD 的学历/经验常在尾部，长简历的项目也可能靠后。若被截断，模型不是“抽取失败”，而是根本没看到。

正确验证方式：

1. 统计各任务 token 长度 P50/P90/P95/P99；
2. 统计被截断样本比例；
3. 检查截断样本尾部包含哪些标签字段；
4. 对 768、1024 等长度做小规模质量与显存对照。

在没有这些统计前，不能只凭显存选择长度。

## 14. 一次参数更新如何发生

假设单设备：

```text
micro step 1: forward → loss/16 → backward，累积梯度
...
micro step 16: forward → loss/16 → backward
optimizer.step(): 只更新 LoRA A/B
scheduler.step()
zero_grad()
```

基础 4-bit W 不更新。验证阶段关闭梯度，计算 valid loss；脚本最后保存 metrics、trainer state 和 adapter。

## 15. 如何判断训练真的有效

训练 loss 下降只是必要非充分条件。应同时检查：

- strict JSON valid rate；
- JD/简历字段级指标；
- raw → normalized → product 三层变化；
- 历史 Gold 回归；
- 新的未检查边界集；
- 错误类型是否从格式错误转为语义错误；
- base、旧 adapter、新 adapter 同条件对比。

若只有训练 loss，没有独立评测，不能声称模型质量提升。

## 16. 面试口述模板

> 我用 4-bit NF4 加载 14B 基座，计算精度是 bf16，只在 attention 和 MLP 的七类投影上训练 rank-8 LoRA。messages 经 Qwen chat template 后只对 assistant JSON token计算 loss，micro batch 1、累积 16 步、梯度检查点换显存。配置里的 DFT 是 TRL 的 Dynamic Fine-Tuning loss，不是 DPO。这里我也保留了版本审慎：仓库 TRL 范围较宽，正式复现需要冻结实际版本并验证 assistant mask 与 DFT 接口，而不能只看 YAML 名字。

