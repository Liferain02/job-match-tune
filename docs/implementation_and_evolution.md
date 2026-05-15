# Implementation And Evolution

## 1. 项目目标

这个项目的目标不是做一个泛化的聊天机器人，而是做一条相对完整的招聘文本结构化抽取链路，覆盖：

1. 公开 JD 抓取
2. JD 清洗和弱标注
3. SFT 数据构造
4. Qwen 模型低成本微调
5. 人工评估与规则修正
6. API 服务和前端演示

最终产物是一个可以直接输入招聘 JD 或简历文本、输出结构化 JSON 的系统。

当前默认方案：

- 基座模型：`Qwen3-14B`
- 微调方式：`4-bit QLoRA`
- 服务形态：
  - `Transformers + PEFT` 本地推理
  - `vLLM + OpenAI-compatible API` 服务化推理
- 结果约束：`JSON Schema structured outputs + 规则后处理`

## 2. 为什么做这个项目

这个问题适合用微调而不是纯 prompt 的原因有两个：

1. 输出结构固定  
   目标字段明确，包括 `岗位方向 / 核心职责 / 必备技能 / 加分项 / 经验要求 / 学历要求`。这类任务适合做监督微调。

2. 误差形式稳定  
   主要错误不是开放式事实幻觉，而是字段边界混淆、方向分类偏差、技能误报。这种误差适合通过数据模板、后处理和人工评估持续收敛。

## 3. 技术栈

### 3.1 模型与训练

- 基座模型：`Qwen3-14B`
- 回退模型：`Qwen3-1.7B`
- 训练方式：`4-bit QLoRA`
- 训练框架：
  - `Transformers`
  - `PEFT`
  - `TRL`
  - `Accelerate`
- 已接入或做过实验的训练技术：
  - `assistant-only loss`
  - `DFT loss`
  - `Liger Kernel`
  - `packing`
  - `gradient checkpointing`

### 3.2 推理与服务

- `Transformers + PEFT` 本地推理
- `FastAPI` 后端 API
- `vLLM` OpenAI-compatible 推理后端
- `OpenAI Python client` 访问 vLLM
- `JSON Schema structured outputs`

### 3.3 数据处理

- `requests`
- `BeautifulSoup`
- `trafilatura`
- `readability-lxml`
- `datasketch`
- `PyYAML`
- `Pydantic`

### 3.4 工程与验证

- `pytest`
- `compileall`
- shell 脚本作为稳定入口

## 4. 当前集群与资源约束

项目实现过程中始终受限于“显存不算奢侈，但也不是极小”的环境。

已知可用资源：

- GPU：`3 x NVIDIA L20`
- 单卡显存：约 `46 GB`
- 系统内存：约 `251 GB`

因此做了两个关键选择：

1. 不做全参数微调，只走 `LoRA / QLoRA`
2. 不直接上 `70B+`，而是先把 `1.7B -> 14B` 跑通

## 5. 目录和模块分工

### 5.1 顶层目录

- `configs/`：配置
- `data/`：数据与人工评估集
- `docs/`：文档和实验记录
- `examples/`：推理样例
- `frontend/`：静态前端
- `models/`：本地模型
- `outputs/`：训练和评估产物
- `scripts/`：运行入口
- `src/jobmatch_tune/`：核心代码

### 5.2 `src/jobmatch_tune/` 关键模块

#### `crawler/`

负责公开招聘数据抓取。

- `tencent_careers.py`
  - 调用腾讯公开招聘接口
  - 支持关键词批量抓取
  - 支持增量写入 JSONL 和 SQLite

#### `preprocess/`

负责文本清洗、字段规则、去重。

- `clean_text.py`
  - 统一清理多余空白、噪声格式
- `normalize_jd.py`
  - 把原始 JD 转为统一 clean 样式
- `jd_field_rules.py`
  - 岗位方向规则
  - 技能抽取
  - 经验 / 学历抽取
- `deduplicate.py`
  - 清洗后的近重复处理

#### `dataset/`

负责训练数据构造。

- `templates.py`
  - system prompt
  - JD / 简历抽取 prompt 模板
- `build_sft_dataset.py`
  - 构造基础 SFT 样本
- `build_direction_hardcase_sft.py`
  - 构造方向 hard case 训练样本
- `build_incremental_sft_dataset.py`
  - 拼接增量训练数据

#### `train/`

- `train_lora.py`
  - 统一训练入口
  - 支持覆盖模型、adapter、数据集、lr、epoch、LoRA rank 等参数
  - 支持现代训练选项：`loss_type / use_liger_kernel / packing / activation_offloading`

#### `inference/`

- `predict.py`
  - 单条推理入口
- `postprocess_json.py`
  - JSON 修复
  - 字段归一化
  - 技能过滤
  - 职责补全
  - 岗位方向规范化
- `structured_output.py`
  - 从 Pydantic schema 构造 `JSON Schema response_format`

#### `api/`

- `server.py`
  - `FastAPI` 服务
  - 支持 `transformers` 和 `vllm` 双后端
  - 提供：
    - `GET /health`
    - `GET /api/status`
    - `POST /api/warmup`
    - `POST /api/parse`

#### `eval/`

- `run_manual_eval.py`
  - 对人工 gold 集做字段级评估
- `metrics.py`
  - 精确匹配、P/R/F1
- `build_manual_eval_dataset.py`
  - 构造人工评估集
- `build_direction_hardcases.py`
  - 构造岗位方向边界样本

## 6. 端到端使用方式

### 6.1 环境准备

```bash
conda create -n tune-demo python=3.11 -y
conda activate tune-demo
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

### 6.2 数据准备

初始化数据库：

```bash
python -m jobmatch_tune.init_db --db data/jobmatch_tune.sqlite3
```

抓取腾讯公开招聘 JD：

```bash
python -m jobmatch_tune.crawler.tencent_careers \
  --keywords-file configs/tencent_keywords.txt \
  --limit 3000 \
  --page-size 50 \
  --max-pages 30 \
  --interval-seconds 0.5 \
  --category 技术 \
  --out data/raw/tencent_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
```

清洗：

```bash
python -m jobmatch_tune.preprocess.normalize_jd \
  --db data/jobmatch_tune.sqlite3 \
  --out data/interim/jd_clean.jsonl \
  --schema configs/label_schema.yaml
```

构造 SFT 数据：

```bash
python -m jobmatch_tune.dataset.build_sft_dataset \
  --jd data/interim/jd_clean.jsonl \
  --out-dir data/sft
```

### 6.3 训练

Smoke：

```bash
bash scripts/train_qwen3_14b_smoke.sh
```

正式训练：

```bash
bash scripts/train_qwen3_14b_full.sh
```

### 6.4 推理

```bash
python -m jobmatch_tune.inference.predict \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --task jd_parse \
  --input examples/jd_ai_app.txt \
  --load-4bit
```

### 6.5 人工评估

```bash
PYTHONPATH=src python -m jobmatch_tune.eval.run_manual_eval \
  --dataset data/eval/jd_manual_eval_50.jsonl \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --out outputs/eval_reports/manual_eval_50_qwen3_14b_v3_report.json \
  --predictions-out outputs/eval_reports/manual_eval_50_qwen3_14b_v3_predictions.jsonl \
  --load-4bit
```

### 6.6 服务

#### FastAPI 本地推理后端

```bash
bash scripts/start_api.sh
```

#### vLLM 服务化后端

先启动 vLLM：

```bash
bash scripts/start_vllm_server.sh
```

再让 API 走 vLLM：

```bash
export JOBMATCH_INFERENCE_BACKEND=vllm
export JOBMATCH_VLLM_BASE_URL=http://127.0.0.1:8010/v1
export JOBMATCH_VLLM_MODEL=jobmatch-lora
bash scripts/start_api.sh
```

## 7. 关键实现细节

### 7.1 为什么要保留规则后处理

这个任务最终不只是“生成一段像 JSON 的文本”，而是“稳定输出结构化字段”。

纯模型生成的问题主要有：

- 字段边界混淆
- 技能误报
- 职责条目漏掉末尾 bullet
- 岗位方向在边界标题上漂移

因此最终方案不是“只靠模型”，而是：

`模型生成 + JSON 修复 + 规则归一化 + 人工评估`

### 7.2 为什么人工评估比 loss 更重要

这个项目里训练 loss 下降很快，但并不能直接代表字段级质量更好。

实际遇到的典型情况：

- `eval_loss` 更低
- 但 `岗位方向` 反而退化
- 或 `必备技能` 误报更多

所以后面判断版本优劣时，主要依据不再是 loss，而是：

1. JSON 合法率
2. 岗位方向 exact match
3. 职责 / 技能 / 加分项 F1
4. 经验 / 学历 exact match

### 7.3 为什么 14B 最终赢在“工程完整度”，而不是单纯模型大小

一开始使用的是 `Qwen3-1.7B`。它的优势是：

- 快
- 便宜
- 适合快速试错

但项目后期的目标已经变成“可展示、可写简历、可交付 demo”。这时 14B 更合适，原因是：

- 对复杂 JD 文本鲁棒性更好
- 在保持规则后处理的前提下更容易做成稳定服务
- 配合 vLLM 更容易扩成完整部署故事

## 8. 迭代过程

下面按阶段记录项目是怎么一步步演进的。

### 阶段 1：搭基础链路

先完成了最小可运行骨架：

- 公开 JD 抓取
- SQLite 入库
- 文本清洗
- 初始 SFT 数据构造
- 1.7B QLoRA smoke

这一阶段的目标不是质量，而是把“数据 -> 训练 -> 推理”最短链路跑通。

### 阶段 2：扩数据

早期数据量太小，不足以支撑微调。

因此扩了腾讯招聘抓取：

- 扩关键词覆盖到多个技术方向
- 改成增量合并写入
- 重建 raw / clean / sft 数据

这一步解决的是训练样本量问题。

### 阶段 3：先用 1.7B 建立基线

在数据量刚扩大时，先用 `Qwen3-1.7B` 做低成本试错：

- smoke 训练
- 正式训练
- DFT / Liger / packing 对照实验

结果很重要：  
训练技术虽然能影响 loss，但对最终字段质量的帮助不总是线性。

### 阶段 4：补人工评估

这是项目的第一个分水岭。

做了几件事：

- 从种子集扩到 50 条人工 gold
- 增加岗位方向 hard case
- 建立字段级指标
- 固化标注口径

从这一步开始，项目进入“以评估驱动迭代”，而不是“以训练驱动迭代”。

### 阶段 5：修岗位方向和技能边界

这个阶段没有盲目继续训，而是围绕错例修系统：

- `岗位方向` 显式优先级规则
- `AI应用开发 / 算法工程 / 后端开发 / 测试开发` 边界收紧
- 技能 canonical schema
- 技能误报过滤
- 职责缺失补回

最后在 50 条 holdout 上把结构化指标收敛到稳定可用。

### 阶段 6：尝试增量 SFT，但没有盲信训练

做过方向 hard case 增量训练，但结果说明一个事实：

- 增量 SFT 不一定提升
- 有时会让方向判断或技能字段退化

因此保留了实验结论，但没有把那几轮 checkpoint 当默认方案。

这一步的价值在于明确了系统瓶颈：  
后期瓶颈不是“训练不够多”，而是“标注边界和后处理逻辑”。

### 阶段 7：从 1.7B 升级到 14B

后期明确要求必须上 14B，于是做了：

- 14B 权重准备
- smoke 微调
- 正式 14B QLoRA 训练
- 14B 人工评估
- 后处理针对 14B 误报再收一轮

最终 14B 版在 holdout 上追平最佳结构化质量，并成为默认服务版本。

### 阶段 8：补成完整应用工程

训练做完之后，项目还补了部署和展示层：

- `FastAPI` 服务
- 静态前端
- 模型预热与状态接口
- `vLLM` 双后端支持
- `JSON Schema structured outputs`

到了这一步，项目就不再只是“微调实验”，而是“一个有训练、有评估、有服务的 LLM 工程项目”。

## 9. 当前结果与结论

当前最佳默认方案：

- `Qwen3-14B`
- `outputs/checkpoints/qwen3-14b-jobmatch-qlora`
- `postprocess_json.py` 最新后处理规则

人工 holdout 最新结果：

- `岗位方向 exact_match = 1.0`
- `核心职责 F1 = 1.0`
- `必备技能 F1 = 1.0`
- `加分项 F1 = 1.0`
- `经验要求 exact_match = 1.0`
- `学历要求 exact_match = 1.0`

需要注意的一点：

这个结果是在当前 50 条人工 holdout 上成立，不意味着任务已经“完全解决”。它说明：

- 当前 pipeline 已经形成稳定闭环
- 当前评估集上质量足够支撑 demo、项目展示和后续扩展

## 10. 后续可以怎么继续做

如果继续往更强的项目形态推进，优先级建议如下。

### 10.1 高优先级

- 增加 `DPO / ORPO` 偏好优化链路
- 增加真实简历评估集
- 扩展 API：
  - 批量解析
  - 异步任务
  - 导出结果

### 10.2 中优先级

- 加上 `vLLM` 延迟和吞吐压测
- 增加多 LoRA / 多任务路由
- 把前端做成更完整的对比评估台

### 10.3 低优先级

- 更激进的 RL 路线，例如 `GRPO`
- 更重的训练栈，例如 `DeepSpeed ZeRO / FSDP`

这些方向不是没价值，而是对当前项目阶段来说不是第一优先级。

## 11. 相关文档

- 结构说明：[project_structure.md](/share/home/lifr/workspace/code/job-match-tune/docs/project_structure.md)
- 简历亮点：[resume_project_highlights.md](/share/home/lifr/workspace/code/job-match-tune/docs/resume_project_highlights.md)
- 岗位方向口径：[job_direction_policy.md](/share/home/lifr/workspace/code/job-match-tune/docs/job_direction_policy.md)
- 历史实验记录：
  - [experiment_results_2026-05-11.md](/share/home/lifr/workspace/code/job-match-tune/docs/experiment_results_2026-05-11.md)
  - [incremental_sft_2026-05-13.md](/share/home/lifr/workspace/code/job-match-tune/docs/incremental_sft_2026-05-13.md)
  - [manual_eval_2026-05-13.md](/share/home/lifr/workspace/code/job-match-tune/docs/manual_eval_2026-05-13.md)
