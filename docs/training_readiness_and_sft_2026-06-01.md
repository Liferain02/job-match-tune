# 2026-06-01 训练前审计与 14B SFT 决策

更新时间：2026-06-01

## 1. 本轮目标

本轮不是直接盲训，而是先回答三个问题：

1. 当前 JD、resume、match 多任务数据是否达到训练门槛。
2. 14B QLoRA 配置是否适合当前 `3 x NVIDIA L20`。
3. SFT 之后的 DPO 是否有独立、无泄漏、可复现的 preference 数据。

## 2. 发现并修复的训练前问题

### 2.1 JD 人工 holdout 泄漏

旧版 `data/sft_jd_quality/` 中，有 44 条训练来源与 `data/eval/jd_manual_eval_50.jsonl` 重叠。

这会让 50 条人工评估结果偏乐观，因此本轮做了两件事：

1. `build_jd_quality_sft_dataset.py` 默认读取人工 holdout 并排除对应 `source_id`。
2. `report_data_readiness.py` 增加 `holdout_overlap` 门控。

重建后：

- `holdout_overlap = 0`
- JD quality 总规模仍保持 5500 条

### 2.2 resume 模板扩写被误当成独立数据量

旧版 resume 只看 SFT 总行数，容易把模板增强后的 4.8 万条当成 4.8 万份独立简历。

本轮新增 `report_resume_sft_profile.py`，统计：

- 独立 source group 数
- 每个 source group 的扩写倍数
- variant 分布
- bootstrap source group 占比
- bootstrap 样本占比

同时移除了默认的递归物化路径：

```text
sft_resume -> resume_train_pool_from_sft -> combined pool -> sft_resume
```

`resume_train_pool_from_sft` 保留为 legacy 工具，但不再默认进入候选池。

### 2.3 preference 数据为空或污染 holdout

旧版 `data/preference/` 可以从人工 holdout 的预测错例生成，这会污染评估。

本轮新增：

- `build_preference_bootstrap_dataset.py`
- `report_preference_readiness.py`

新的 preference 从 `data/sft_jd_quality/train/valid` 独立生成结构化 hard negatives：

- `unexpected_field`
- `direction_mismatch`
- `responsibility_drop`
- `responsibility_skill_leak`
- `education_experience_mix`

当前 preference 规模：

- train：4400
- valid：550
- holdout overlap：0

这批数据适合做结构化输出对齐的 DPO 起点。后续仍应补人工 preference，不能只依赖合成负例。

## 3. 新增公开简历数据

本轮增加轻量下载器：

- `src/jobmatch_tune/dataset/download_public_resume_samples.py`
- `scripts/data/download_public_resume_samples.sh`

公开来源：

1. `OhMyKing/FairCV`
   - 拉取 1000 条中文模拟简历
   - 用于补充真实简历格式覆盖

2. `PassbyGrocer/resume-ner`
   - 拉取 train parquet
   - 共 3821 条中文简历 NER
   - 只进入外部语料审计，不直接混入 resume_parse SFT

NER 标签和 JSON 结构化解析不是同一种监督信号，因此当前只作为后续字段识别辅助语料。

## 4. 当前 readiness 结果

执行：

```bash
bash scripts/data/report_resume_sft_profile.sh
bash scripts/data/report_preference_readiness.sh
bash scripts/data/report_data_readiness.sh
```

当前结果：

```text
all_ready_for_sft = true
ready_for_dpo_smoke = true
ready_for_dpo = true
```

关键数据：

| 数据线 | 规模 |
| --- | --- |
| JD quality | 4400 / 550 / 550 |
| resume | 39132 / 5083 / 4952 |
| match | 3917 / 486 / 493 |
| multitask | 9800 / 1208 |
| preference | 4400 / 550 |

resume 画像：

| 指标 | 值 |
| --- | ---: |
| source groups | 3529 |
| expansion ratio | 13.9323 |
| bootstrap source groups | 2369 |
| bootstrap source group rate | 0.6713 |
| max variant rate | 0.0718 |

## 5. 对照 TRL 官方 SFT 方法

TRL 官方文档支持：

- conversational messages 数据格式
- `assistant_only_loss=True`
- PEFT / LoRA adapter
- `packing=True`
- `loss_type="dft"`
- Liger Kernel

参考：

- <https://huggingface.co/docs/trl/sft_trainer>
- <https://huggingface.co/docs/trl/dpo_trainer>
- <https://github.com/hiyouga/LLaMA-Factory>
- <https://docs.axolotl.ai/docs/optimizations.html>
- <https://github.com/datajuicer/data-juicer>

当前项目采用：

- `Qwen3-14B`
- `4-bit NF4 QLoRA`
- `assistant_only_loss=True`
- `gradient_checkpointing=True`
- `bf16=True`
- `loss_type="dft"`
- `packing=False`

### 为什么暂时禁用 packing

14B smoke 中，TRL 对 `packing + sdpa` 发出警告：

```text
packing 需要受支持的 FlashAttention 实现，否则可能出现样本间污染
```

当前项目配置是 `attn_implementation=sdpa`，因此正式训练保持：

```yaml
packing: false
```

后续只有在安装并验证 `flash_attention_2` 后，才重新启用 packing。

### 与主流开源实践的对应关系

- TRL：使用 conversational messages、`assistant_only_loss=True`、PEFT adapter、DFT 和 DPO。
- LLaMA-Factory：按训练阶段区分 SFT 与 preference tuning；本项目也把多任务 SFT 和 DPO 配置拆开。
- Axolotl：将 QLoRA、gradient checkpointing、FlashAttention、sample packing 视为独立优化开关；本项目只启用已验证的组合。
- Data-Juicer：把数据质量门控和可审计画像放在训练前；本项目 readiness、risk report、resume profile 和 preference report 均遵循这一思路。

本轮还为 SFT 和 DPO 训练入口增加稳定落盘指标：

```text
train_metrics.json
eval_metrics.json
trainer_state.json
```

这样后续选择 adapter 时不依赖终端滚动日志。

## 6. 14B smoke 结果

本轮在 GPU03 上运行：

1. 标准 `NLL`
2. `DFT + packing`
3. `DFT without packing`

结论：

- `NLL`：可训练。
- `DFT + packing`：可以运行，但因 SDPA 兼容性警告作废。
- `DFT without packing`：训练稳定，作为正式 SFT 推荐配置。

`DFT without packing` smoke：

```text
train_runtime = 18.72s
eval_loss = 0.02958
eval_mean_token_accuracy = 0.8338
```

标准 `NLL` smoke：

```text
train_runtime = 21.18s
eval_loss = 1.264
eval_mean_token_accuracy = 0.8339
```

DFT 和 NLL 的 loss 定义不同，不能直接用 loss 数值横向比较。正式训练仍然保留 NLL 基线，用人工评估和任务级指标比较。

## 7. 正式训练策略

本轮正式启动两条 14B SFT A/B：

1. DFT 主线
   - 输出：`outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601`

2. NLL 基线
   - 输出：`outputs/checkpoints/qwen3-14b-jobmatch-nll-20260601`

两条训练完成后：

1. 读取 trainer state。
2. 跑 JD 人工 holdout。
3. 跑 resume pipeline eval。
4. 跑 match eval。
5. 选择更稳定的 adapter。
6. 再做 DPO smoke 和正式 DPO。

## 8. DPO 使用边界

DPO 不应该替代 SFT，也不应该在 preference 数据为空时启动。

当前顺序固定为：

```text
readiness -> SFT smoke -> 正式 SFT -> 任务评估 -> DPO smoke -> 正式 DPO -> 再评估
```

当前结构化 hard-negative preference 可以验证 DPO 链路。后续要继续增加：

- 人工挑选 chosen / rejected
- API 真实错例
- resume 字段漏抽错例
- match 解释不一致错例
- 边界岗位方向错例

### DPO smoke 结果

使用旧 14B SFT adapter 和新的独立 preference 数据运行 64 条训练样本、16 条验证样本、4 个 optimizer step：

```text
train_loss = 0.6935
eval_loss = 0.6922
eval_rewards/accuracies = 0.5625
eval_rewards/margins = 0.002404
```

1-step smoke 的 reward margin 为 0；4-step smoke 已能拉开 chosen / rejected，说明 DPO 数据格式、adapter 续训和 preference 链路可用。该结果只用于验证链路，正式收益要等 SFT A/B 完成后，在选定 adapter 上重新训练和评估。
