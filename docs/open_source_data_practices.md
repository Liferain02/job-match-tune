# 开源 SFT 数据处理实践对照

这份文档记录当前项目参考的开源 SFT 数据处理实践，以及已经落地到本项目的改动。

## 1. 参考项目和可借鉴点

### 1.1 Axolotl

Axolotl 的数据处理强调：

- 先把数据整理成明确的数据格式，再进入训练。
- 支持训练前单独 preprocess，避免每次训练重复处理。
- 对样本长度做过滤或截断。
- 对 train/eval 做去重，避免评估泄漏。
- 支持 packing、multipack 等提升训练吞吐。

本项目已经吸收的点：

- 所有 SFT 数据统一为 chat messages JSONL。
- readiness 增加 `train / valid / test` 内容级去重检查。
- 训练脚本侧保留 packing、Liger Kernel、gradient checkpointing 等选项。

### 1.2 LLaMA-Factory

LLaMA-Factory 的数据准备强调：

- 通过 dataset registry 管理数据集。
- 明确支持 Alpaca / ShareGPT / OpenAI messages 等格式。
- 训练前先把自定义数据映射成稳定格式。

本项目已经吸收的点：

- 统一使用 messages 格式：
  - `system`
  - `user`
  - `assistant`
- 三条任务线都输出同一种训练格式：
  - `jd_parse`
  - `resume_parse`
  - `match`

后续可以继续做：

- 继续扩展项目内 `dataset_registry.yaml`，集中登记更多实验数据线、字段、用途和推荐训练权重。

### 1.3 Open-Instruct

Open-Instruct 的一个重要实践是 decontamination，也就是检查训练集和评估集是否重叠。

本项目已经吸收的点：

- `report_data_readiness.py` 不再只检查 ID。
- 现在会对 `user prompt + assistant JSON` 做内容哈希。
- 如果同一内容跨 `train / valid / test` 出现，则 readiness 判定为不通过。

这次检查直接发现了简历数据的 608 条跨 split 内容重复。修复方式是：

- 在 `build_resume_sft_dataset.py` 中新增 exact content dedup。
- `build_resume_sft_dataset.sh` 改为基于 combined pool 构建。
- 重建后 resume 跨 split 重复降为 0。

### 1.4 Data-Juicer / DataFlow

Data-Juicer 和 DataFlow 的共同点是把数据处理拆成可组合 operator：

- clean
- filter
- deduplicate
- evaluate
- export

本项目已经吸收的点：

- 数据脚本按阶段拆分在 `scripts/data/`。
- readiness 已经变成独立 evaluate gate。
- JD 数据分层为 `strict / strict_plus / quality_weak / bootstrap`。

后续可以继续做：

- 把 JD quality gate 的通过/拒绝原因写成显式 reason code。
- 输出 reject samples，方便人工审计。
- 给每条样本增加 `quality_tier` 和 `quality_score`。

## 2. 本轮已经落地的优化

### 2.1 match 数据扩量

之前：

- combined pool: `1176`
- SFT split: `929 / 132 / 115`

本轮改动：

- `build_match_train_pool_synthetic.py` 默认 `max_jd_rows` 从 `360` 提升到 `720`
- 重建 synthetic match pool

现在：

- combined pool: `2256`
- SFT split: `1799 / 228 / 229`

### 2.2 resume 数据扩量和去泄漏

之前：

- SFT split: `2560 / 320 / 320`
- readiness 没有内容级跨 split 检查

新增检查后发现：

- `cross_split_duplicate_hashes = 608`

修复：

- `build_resume_sft_dataset.py` 增加 exact content dedup。
- `build_resume_sft_dataset.sh` 改为使用 `data/eval/resume_train_pool_combined.jsonl`。

现在：

- SFT split: `38408 / 4850 / 4890`
- 总量：`48148`
- `cross_split_duplicate_hashes = 0`

### 2.3 readiness 门控增强

readiness 现在检查：

- 数量
- pool 规模
- assistant JSON 合法性
- ID 重复
- 跨 split 内容重复
- 关键字段空值率

当前结果：

- `JD`: ready
- `resume`: ready
- `match`: ready
- `multitask`: ready
- `all_ready_for_training = true`

### 2.4 多任务训练集 registry

resume 扩量后达到 `48148` 条，如果直接和 JD、match 混合训练，会明显改变任务分布。参考 LLaMA-Factory 的 dataset registry 和 Axolotl 的 dataset 配置思路，本项目新增：

- [configs/dataset_registry.yaml](/share/home/lifr/workspace/code/job-match-tune/configs/dataset_registry.yaml)
- [build_multitask_sft_dataset.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/dataset/build_multitask_sft_dataset.py)
- [build_multitask_sft_dataset.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/build_multitask_sft_dataset.sh)

当前多任务训练集：

- `train`: `8000`
- `valid`: `1000`

任务配比：

- `JD`: `4000 / 500`
- `resume`: `2400 / 300`
- `match`: `1600 / 200`

这样既保留了 resume 扩量带来的格式多样性，又避免训练时被 resume 单任务压过。

### 2.5 JD quality 可解释质量画像

参考 Data-Juicer / DataFlow 的可审计数据处理思路，JD quality 现在不只输出训练样本，还输出质量画像：

- [outputs/eval_reports/jd_quality_profile.json](/share/home/lifr/workspace/code/job-match-tune/outputs/eval_reports/jd_quality_profile.json)

画像包含：

- `quality_tier`
- `quality_reason`
- 来源分布
- 岗位方向分布
- 字段空值率

当前 JD quality 分层：

- `strict`: `3331`
- `strict_plus`: `275`
- `quality_weak`: `1394`
- `bootstrap`: `0`

同时新增按层级抽样的人工复核种子集：

- [data/eval/jd_quality_review_seed.jsonl](/share/home/lifr/workspace/code/job-match-tune/data/eval/jd_quality_review_seed.jsonl)

默认每个层级抽 `20` 条，共 `60` 条，便于重点复核 `quality_weak`。

## 3. 当前仍应继续优化的方向

1. JD 数据继续补高信任中文官网源，而不是盲目扩大弱源。
2. 给 JD quality 增加 reason code，让每条样本通过原因可解释。
3. 对 `quality_weak` 层做人工抽样评估。
4. 在多任务训练后按任务分别评估，确认 JD、resume、match 没有互相拖累。
5. 根据评估结果调整 `configs/dataset_registry.yaml` 中的采样配比。

## 4. 参考链接

- Axolotl dataset formats: https://docs.axolotl.ai/docs/dataset-formats/index.html
- Axolotl GitHub: https://github.com/axolotl-ai-cloud/axolotl
- LLaMA-Factory data preparation: https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html
- Open-Instruct GitHub: https://github.com/allenai/open-instruct
- Data-Juicer GitHub: https://github.com/modelscope/data-juicer
- Data-Juicer distributed processing: https://datajuicer.github.io/data-juicer/en/main/docs/Distributed.html
