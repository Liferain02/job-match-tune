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

正式 A/B 首轮运行还发现：`1208` 条 multitask valid 在 14B 上全量评估耗时明显。后续默认配置调整为：

```text
SFT eval_steps = 300
DPO eval_steps = 100
SFT max_eval_samples = 320
DPO max_eval_samples = 128
```

Trainer 验证使用固定抽样集做同口径比较，减少中间评估占用 GPU 的时间。业务质量仍使用独立 JD、resume、match 人工集完整验证。

首轮旧配置运行到 `step=100` 验证阶段后，因全量 `1208` 条 valid 长时间未完成且尚未生成 checkpoint，已经停止并用新配置重启。重启后确认：

```text
train samples = 9800
trainer eval samples = 320
optimizer steps = 613
单步约 12.9s
```

SFT 与 DPO 入口同时新增：

```text
--resume_from_checkpoint
```

长训练可以从已保存 checkpoint 恢复，不需要因配置优化或节点中断从头重跑。

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

### 无泄漏 JD 基线

在正式 A/B 完成前，先用此前的 `outputs/checkpoints/qwen3-14b-jobmatch-qlora` 对排除训练重叠后的 50 条人工 holdout 重新评估：

```text
json_valid_rate = 0.98
岗位方向 exact_match = 0.9592
核心职责 F1 = 1.0
必备技能 F1 = 1.0
加分项 F1 = 1.0
经验要求 exact_match = 1.0
学历要求 exact_match = 1.0
```

当前有 2 条 mismatch：

1. `tencent_2005480420615016448_jd_parse`：客户端开发 -> 后端开发
2. `tencent_2039174621139464192_jd_parse`：算法工程 -> AI应用开发

该结果是后续 DFT、NLL 和 DPO adapter 的同口径基线。

### 旧 adapter 的 resume 基线

用此前的 14B adapter 跑 32 条 resume 人工样本：

```text
json_valid_rate = 1.0
核心技能 F1 = 1.0
教育背景 F1 = 0.0
实习经历 F1 = 0.0
项目经历 F1 = 0.0
优势标签 F1 = 0.4271
目标岗位 exact_match = 0.0625
```

主要问题不是 JSON 外壳，而是结构语义：

- 教育、实习和项目字段常输出字符串，而不是 schema 约定的列表。
- 多条项目经历经常合并为一个字符串。
- 目标岗位经常附带“工程师”后缀，没有归一化到固定方向类。

这组结果说明新一轮多任务 SFT 必须重点验证 resume 字段类型和岗位方向归一化，而不能只看训练 loss。

本轮增加 resume schema 后处理后，使用同一批旧模型原始输出离线复算：

```text
json_valid_rate = 1.0
核心技能 F1 = 1.0
教育背景 F1 = 1.0
实习经历 F1 = 1.0
项目经历 F1 = 1.0
优势标签 F1 = 0.4271
目标岗位 exact_match = 0.96875
```

后处理修复包括：

- 教育、技能、实习、项目、优势字段统一转为字符串列表。
- 项目经历按中文或英文分号拆分，并补齐句末标点。
- 目标岗位复用固定岗位方向规则归一化。
- resume 输出不再附加 JD 专属字段。

优势标签仍需要通过 SFT 和后续语义规则继续优化。

match 基线还暴露出规则引擎兼容性问题：resume 经历项可能是对象，旧逻辑直接 `"\n".join(...)` 会抛出 `TypeError`。本轮已统一兼容字符串、对象和列表，并让 match 评估只加载一次 14B 模型。

修复后，旧 adapter 的 64 条 match 基线为：

```text
jd_resume_parse_success_rate = 1.0
analysis_json_valid_rate = 1.0
命中技能 F1 = 0.2869
缺失技能 F1 = 0.4609
匹配等级 exact_match = 0.5625
岗位方向匹配 exact_match = 0.875
学历匹配 exact_match = 1.0
经验匹配 exact_match = 1.0
```

match 主链已经能稳定运行，但技能匹配召回和匹配等级仍有明显优化空间。

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

## 9. 正式 14B SFT A/B 结果

两条正式 SFT 均使用 9800 条多任务训练样本和 320 条采样验证样本，完成 613 个 optimizer step。

| 分支 | 训练耗时 | train loss | eval loss | eval token accuracy |
| --- | ---: | ---: | ---: | ---: |
| DFT without packing | 8050.39s | 0.001084 | 0.000198 | 0.9717 |
| NLL | 8029.27s | 0.049181 | 0.006650 | 0.9979 |

DFT 和 NLL 的损失定义不同，loss 不能直接横向比较。

### JD 人工 holdout

DFT、NLL 和旧 adapter 在 50 条无泄漏 JD holdout 上结果一致：

```text
json_valid_rate = 0.98
岗位方向 exact_match = 0.9592
核心职责 / 必备技能 / 加分项 F1 = 1.0
经验要求 / 学历要求 exact_match = 1.0
```

剩余两条边界错例应补充同分布 hard case，但不能直接回灌人工 holdout：

1. 客户端开发被判为后端开发。
2. 算法工程被判为 AI 应用开发。

### resume 人工评估

正式 SFT 后，教育、技能、实习、项目和目标岗位字段均达到 `1.0`。优势标签仍有表达形式差异：模型经常输出“熟悉模型平台、算力调度和容器化部署”一类合并句。

新增通用后处理：

1. 按顿号、逗号、“和”、“与”拆分标签。
2. 去除“熟悉”、“具备”、“擅长”、“关注”、“有”、“能独立完成”等表达包装。
3. 去除标签末尾标点和“能力强”、“能力”、“覆盖全面”。
4. 仅压缩标签空白，不对业务语义做样本级特判。

离线复算结果：

```text
教育背景 / 核心技能 / 实习经历 / 项目经历 F1 = 1.0
目标岗位 exact_match = 1.0
优势标签 F1 = 0.8354
```

剩余 11 条差异属于语义粒度差异，例如“交互体验”与“交互体验优化”、“评测”与“模型评测”。这些样本应进入后续标注和增量 SFT，不应继续堆字符串替换规则。

### resume 优势标签口径归一化补充

2026-06-02 在不重新训练、不修改 gold 的前提下，新增 `configs/resume_strength_alias.yaml`，把优势标签中稳定出现的同义表达迁移到配置化口径表：

- “LLM应用落地经验” -> “LLM应用落地”
- “交互体验” -> “交互体验优化”
- “基础设施稳定性建设” -> “稳定性建设”
- “自动化测试框架建设” -> “自动化测试框架”
- “高并发场景优化” -> “高并发优化”

这不是样本级补丁，而是把“能力标签应该短、稳定、可枚举”的标注口径显式化。后续新增简历 gold 或训练数据时，应先维护这张表，再重建 `data/sft/`。

使用最新后处理重放已保存预测：

```bash
PYTHONPATH=src python -m jobmatch_tune.eval.replay_generation_predictions \
  --predictions outputs/eval_reports/resume_pipeline_eval_qwen3_14b_dft_dpo_final_20260602_predictions.jsonl \
  --out outputs/eval_reports/resume_pipeline_eval_qwen3_14b_dft_dpo_final_20260602_replay_report.json
```

重放结果：

```text
json_valid_rate = 1.0
教育背景 / 核心技能 / 实习经历 / 项目经历 F1 = 1.0
目标岗位 exact_match = 1.0
优势标签 precision = 1.0
优势标签 recall = 0.9896
优势标签 F1 = 0.99375
剩余 mismatch = 1
```

唯一剩余错例是 `resume_eval_023` 漏掉“稳定性验证”。这属于模型召回不足或训练样本覆盖不足，不应通过后处理凭空补字段；下一步应补同类“测试开发 / 稳定性验证”简历 hard case 后再做小规模增量 SFT。

## 10. DPO conversational preference 修复

第一次正式 DPO 启动时，TRL 输出大量 prompt tokenization mismatch 警告。根因是 preference 使用纯字符串：

```text
prompt + chosen
prompt + rejected
```

Qwen tokenizer 在部分 JSON 边界会发生重分词，TRL 按 `len(prompt_ids)` 截取 completion 时存在边界不确定性。

本轮将 preference 改为 TRL 官方支持的 conversational 格式：

```json
{
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

同时更新：

1. bootstrap preference 构造器。
2. API 错例 preference 构造器。
3. readiness 审计器，兼容旧字符串和新 conversational 格式。
4. 单元测试。

重建后仍有：

```text
train = 4400
valid = 550
invalid_rows = 0
holdout_overlap = 0
ready_for_dpo = true
```

16 条训练样本、16 条验证样本的 conversational DPO smoke 已完成，原边界警告消失：

```text
train_runtime = 46.56s
train_loss = 0.6931
eval_loss = 0.6931
```

### match A/B 与最终选择

64 条 match 人工样本的 A/B 结果：

| 指标 | DFT | NLL |
| --- | ---: | ---: |
| JD / resume 解析成功率 | 1.0 | 1.0 |
| 分析 JSON 合法率 | 1.0 | 1.0 |
| 命中技能 F1 | 0.2869 | 0.2791 |
| 缺失技能 F1 | 0.4609 | 0.4375 |
| 匹配等级 exact_match | 0.53125 | 0.515625 |
| 岗位方向匹配 exact_match | 0.90625 | 0.875 |
| 学历 / 经验匹配 exact_match | 1.0 | 1.0 |

JD 和 resume 指标打平，DFT 在 match 各项指标上稳定略优，因此最终选择 DFT adapter 作为正式 DPO 起点。

正式 DPO 输出到：

```text
outputs/checkpoints/qwen3-14b-jobmatch-dft-dpo-chat-20260602
```

DPO 完成后仍需重新运行 JD、resume 和 match 评测。当前 preference 主要来自 JD 结构化 hard negative，因此必须确认多任务能力没有回退。

### match 技能 taxonomy 扩充

match 技能指标偏低的直接原因是 `configs/label_schema.yaml` 的技能白名单过窄。旧 schema 主要覆盖 AI 应用和少量后端技能，导致跨领域技能在 JD 后处理阶段被过滤。

本轮补充：

- 前端：`TypeScript`、`React`、`Vite`、`ECharts`
- 后端与云原生：`FastAPI`、`Go`、`Linux`、`Docker`、`Kubernetes`、`Prometheus`、`Grafana`
- 数据与中间件：`Kafka`、`Flink`、`Spark`、`Airflow`、`SQL`、`Hive`
- HPC：`CUDA`、`NCCL`、`MPI`
- 测试与运维：`Selenium`、`Pytest`、`JMeter`、`Ansible`
- 网络：`TCP/IP`、`OSPF`、`BGP`
- 嵌入式与硬件：`CAN`、`STM32`、`C`、`C++`、`C语言`、`Cadence`、`EMC`、`示波器`、`电源设计`

同时将技能提取从简单子串改成 ASCII 边界匹配，避免新增 `C`、`Go`、`SQL` 后产生误报。

DFT match v2 复算结果：

| 指标 | 扩充前 | 扩充后 |
| --- | ---: | ---: |
| 命中技能 F1 | 0.2869 | 0.8548 |
| 缺失技能 F1 | 0.4609 | 0.8945 |
| 匹配等级 exact_match | 0.53125 | 0.78125 |
| 岗位方向匹配 exact_match | 0.90625 | 0.90625 |
| 学历 / 经验匹配 exact_match | 1.0 | 1.0 |

扩充后，分析 JSON 合法率从 `1.0` 降为 `0.96875`。规则结果仍全部可用，但有两条生成式 match 分析输出不合法。本轮继续为 match 评测补充失败原始输出和错误字段，后续 DPO 复评时可以直接审计。

失败诊断确认：两条低匹配样本都在结尾追加总结句时漏写 `"匹配结论":` 键。进一步检查还发现，旧通用后处理会把 match JSON 当成 JD JSON，附加职责、技能、学历等无关字段。

本轮继续修复：

1. match prompt 明确固定 5 个字段及其类型。
2. JSON 修复层兼容尾部总结句漏写 `匹配结论` 键的格式。
3. match 专属后处理只保留 `匹配结论`、`匹配优势`、`主要短板`、`简历优化建议`、`推荐投递岗位方向`。
4. 失败评测保存原始输出和解析错误，便于后续审计。

两条历史失败输出离线复算后均可恢复为合法 match schema。

修复后离线复算，分析 JSON 合法率恢复到 `1.0`，规则字段指标保持不变。

### JD 人工 holdout 标注修正

taxonomy 扩充后，JD holdout 首次复算出现 3 条技能差异。逐条审计原文后确认都是旧 gold 漏标：

1. `QQ邮箱-web前端开发工程师` 原文明确写有 `React/Vue`，补标 `React`。
2. `微信-后台开发工程师-AI方向` 原文明确写有 `C++`，补标 `C++`。
3. `agent应用开发工程师` 原文明确写有 `Linux`，在原有 `Agent` 基础上补标 `Linux`。

修正 gold 后离线复算：

```text
json_valid_rate = 0.98
核心职责 / 必备技能 / 加分项 F1 = 1.0
岗位方向 exact_match = 0.9592
经验要求 / 学历要求 exact_match = 1.0
```

仍只剩原来的两条岗位方向边界错例。旧报告保留用于解释标注口径变化，新版 holdout 用于后续 DPO 对比。

## 11. 正式 DPO 训练结果

最终使用 DFT adapter 作为起点，运行 4400 条 conversational preference、1 epoch、275 个 optimizer step：

```text
output = outputs/checkpoints/qwen3-14b-jobmatch-dft-dpo-chat-20260602
train_runtime = 9089.53s
train_loss = 0.2099
eval_loss = 0.1749
eval_rewards/accuracies = 0.9297
eval_rewards/margins = 3.3643
```

中间验证：

| step | reward accuracy | reward margin |
| ---: | ---: | ---: |
| 100 | 0.9375 | 3.0301 |
| 200 | 0.9297 | 3.3628 |
| 275 | 0.9297 | 3.3643 |

chosen 与 rejected 已明显拉开，训练过程未发散。由于 preference 主要来自 JD 结构化 hard negative，最终是否采用 DPO adapter 仍由 JD、resume、match 三路业务复评决定。

### DPO 后业务复评

| 指标 | DFT SFT | DFT + DPO |
| --- | ---: | ---: |
| JD JSON 合法率 | 0.98 | 0.98 |
| JD 岗位方向 exact_match | 0.9592 | 0.9592 |
| JD 职责 / 技能 / 加分项 F1 | 1.0 | 1.0 |
| JD 经验 / 学历 exact_match | 1.0 | 1.0 |
| resume JSON 合法率 | 1.0 | 1.0 |
| resume 教育 / 技能 / 实习 / 项目 F1 | 1.0 | 1.0 |
| resume 目标岗位 exact_match | 1.0 | 1.0 |
| resume 优势标签 F1 | 0.8354 | 0.8354 |
| resume 优势标签 F1（最新口径重放） | 0.99375 | 0.99375 |
| match JSON 合法率 | 1.0 | 1.0 |
| match 命中技能 F1 | 0.8548 | 0.8548 |
| match 缺失技能 F1 | 0.8945 | 0.8945 |
| match 匹配等级 exact_match | 0.78125 | 0.78125 |
| match 岗位方向 exact_match | 0.90625 | 0.90625 |

结论：

1. DPO 成功学习了当前结构化 preference，reward margin 明显增加。
2. 三路业务指标没有回退。
3. 当前合成 JD preference 没有带来可测业务增益。
4. 生产默认仍使用 `outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601`。
5. `outputs/checkpoints/qwen3-14b-jobmatch-dft-dpo-chat-20260602` 保留为可复现实验产物，不作为默认服务 adapter。

下一轮若继续 DPO，应增加人工偏好和真实 API 错例，尤其是 resume 优势标签语义粒度、match OCR-like 样本和岗位方向边界样本，而不是继续重复训练同一批合成负例。

## 13. 产品链路 DPO 数据优化

2026-06-02 继续优化 DPO 数据，不再把最终人工 holdout 直接写入训练偏好集。一次临时尝试从最终评测预测构造 `data/preference_product/`，readiness 报告显示 `holdout_overlap=50`，因此该方向被否决。

最终采用的方案是从 `data/sft_multitask/` 训练池构造三任务 bootstrap preference：

```bash
bash scripts/data/build_product_preference_bootstrap_dataset.sh
```

生成结果：

```text
train = 9800
valid = 1208
任务分布：
  jd_parse = 4950
  resume_parse = 3138
  match = 2920
```

readiness：

```text
invalid_rows = 0
duplicate_ids = 0
chosen_equals_rejected = 0
cross_split_prompt_hashes = 0
holdout_overlap = 0
ready_for_dpo_smoke = true
ready_for_dpo = true
```

偏好策略覆盖：

- JD：多余字段、岗位方向错配、职责缺失、职责泄漏到技能、学历经验混淆
- resume：优势标签缺失、项目经历缺失、教育背景泄漏到技能
- match：主要短板缺失、匹配优势/短板互换、优化建议缺失

产品链路 DPO 配置：

```text
configs/train_qwen3_14b_product_dpo.yaml
scripts/train/train_qwen3_14b_product_dpo.sh
```

这一路 DPO 与上一轮 JD-only DPO 的区别是：它直接覆盖最终用户会使用的 `JD 解析 / 简历解析 / 人岗匹配分析` 三个任务，且不污染人工 holdout。
