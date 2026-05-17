# JobMatchTune

面向招聘 JD / 简历结构化抽取的 Qwen3 QLoRA 微调项目。当前默认服务版本为 `Qwen3-14B + LoRA adapter + 规则后处理`。

## 当前默认版本

- 基座模型：`models/Qwen3-14B`
- Adapter：`outputs/checkpoints/qwen3-14b-jobmatch-qlora`
- 服务默认入口：
  - API: [src/jobmatch_tune/api/server.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/api/server.py)
  - 启动脚本: [scripts/serve/start_api.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/serve/start_api.sh)
  - vLLM 服务脚本: [scripts/serve/start_vllm_server.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/serve/start_vllm_server.sh)
- 50 条人工 holdout 最新报告：
  - [outputs/eval_reports/manual_eval_50_qwen3_14b_v3_report.json](/share/home/lifr/workspace/code/job-match-tune/outputs/eval_reports/manual_eval_50_qwen3_14b_v3_report.json)

## 项目结构

详细说明见 [docs/project_structure.md](/share/home/lifr/workspace/code/job-match-tune/docs/project_structure.md)。

核心目录：

- `src/jobmatch_tune/`
  - `crawler/`：公开 JD 抓取
  - `preprocess/`：清洗、去重、规则抽取
  - `dataset/`：SFT / DPO 数据构造
  - `train/`：QLoRA / DPO 训练
  - `inference/`：推理与后处理
  - `api/`：FastAPI 服务
  - `eval/`：人工评估与指标
- `scripts/data/`：抓取、导入、重建数据
- `scripts/train/`：14B 训练入口
- `scripts/serve/`：API / vLLM / 前端启动
- `scripts/dev/`：环境与模型下载
- `scripts/research/`：研究辅助脚本
- `scripts/legacy/`：历史 1.7B 实验脚本归档
- `configs/`：训练、爬取、标签 schema
- `frontend/`：静态前端
- `docs/`：实验记录与口径文档

## 环境

```bash
conda create -n tune-demo python=3.11 -y
conda activate tune-demo
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

查看 `gpu03` 资源：

```bash
ssh -n gpu03 nvidia-smi
```

## 数据链路

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

抓取百度公开招聘 JD：

```bash
python -m jobmatch_tune.crawler.baidu_talent \
  --keywords-file configs/baidu_keywords.txt \
  --interval-seconds 0.5 \
  --out data/raw/baidu_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
```

抓取京东公开招聘 JD：

```bash
python -m jobmatch_tune.crawler.jd_careers \
  --out data/raw/jd_careers_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
```

抓取 Moka 托管招聘官网 JD：

```bash
python -m jobmatch_tune.crawler.moka_careers \
  --sources configs/moka_sources.yaml \
  --out data/raw/moka_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
```

如需一键刷新腾讯数据：

```bash
bash scripts/data/refresh_tencent_data.sh auto
```

如需一键刷新百度数据：

```bash
bash scripts/data/refresh_baidu_data.sh
```

如需一键刷新京东数据：

```bash
bash scripts/data/refresh_jd_data.sh
```

如需一键刷新 Moka 招聘官网数据：

```bash
bash scripts/data/refresh_moka_data.sh
```

如需一键刷新腾讯 + 百度 + 京东 + Moka 并重建下游：

```bash
bash scripts/data/refresh_official_job_data.sh
```

说明：

- `auto`：先尝试抓取，失败则直接用现有 raw 数据重建下游
- `crawl`：强制抓取后再重建
- `rebuild`：只重建清洗、去重和 SFT 数据

导入公开职位导出文件并扩充原始语料：

```bash
bash scripts/data/import_public_job_exports.sh
```

导入大规模中文招聘学历数据：

```bash
bash scripts/data/import_chinese_job_exports.sh
```

当前这条链路会导入三类补充源：

- GitHub `jhcoco/bosszp` CSV
- GitHub `WorkAggregation` CSV
- Hugging Face `open-apply-jobs` 的 Greenhouse / Ashby / Lever parquet 分片
- Hugging Face `job-educational-parser-dataset-08-0-0805` 中文 parquet
- 百度 / 京东 / Moka 招聘官网公开职位抓取

注意：

- 这一步配合腾讯、百度、京东、Moka 官网抓取后，当前 `jd_clean / jd_clean_dedup` 已经达到 `292167 / 273963`。
- 当前去重后语言分布约为：
  - 中文：`221402`
  - 英文：`51330`
  - 其他 / 未知：`927`
- 默认 `data/sft/` 现在是严格质量版：`1408 / 176 / 177`。
- `data/sft_expanded/` 是扩展实验版：`4800 / 600 / 601`。
- 默认训练不再追求先凑满 2 万，而是优先保留高信任官网中文技术岗；弱标注样本只进入扩展实验集，不再直接混入默认集。当前 `20000` 目标只属于扩展实验链路，不代表默认高质量集规模。

清洗与构造训练集：

```bash
python -m jobmatch_tune.preprocess.normalize_jd \
  --db data/jobmatch_tune.sqlite3 \
  --out data/interim/jd_clean.jsonl \
  --schema configs/label_schema.yaml

python -m jobmatch_tune.dataset.build_sft_dataset \
  --jd data/interim/jd_clean.jsonl \
  --out-dir data/sft \
  --quality-profile strict

python -m jobmatch_tune.dataset.build_sft_dataset \
  --jd data/interim/jd_clean.jsonl \
  --out-dir data/sft_expanded \
  --include-weak-tech \
  --quality-profile expanded \
  --target-total 20000
```

构造万级中英混合弱标注 SFT：

```bash
bash scripts/data/build_multilingual_weak_sft.sh
```

说明：

- `data/sft/` 是默认高质量中文集，只保留高信任中文官网样本。
- `data/sft_expanded/` 是扩展实验集，允许少量高置信弱标注样本进入。
- `data/sft_multilingual_weak/` 是规模优先的中英混合弱标注集，适合做第二阶段扩量实验，不建议直接替换默认 demo 版本。

当前中文数据最多的来源：

1. Hugging Face `job-educational-parser-dataset-08-0-0805`
   - 当前导入：`232064` 条中文职位样本
2. 京东公开招聘
   - 当前抓取：`3054` 条
3. 腾讯公开招聘
   - 当前抓取：`935` 条
4. 百度公开招聘
   - 当前抓取：`577` 条
5. Moka 招聘官网公开 API
   - 当前抓取：`2662` 条

## 训练

14B smoke：

```bash
bash scripts/train/train_qwen3_14b_smoke.sh
```

14B 正式训练：

```bash
bash scripts/train/train_qwen3_14b_full.sh
```

如需下载模型快照：

```bash
bash scripts/dev/download_qwen_models_python.sh 14B
```

轻量回退模型：

```bash
bash scripts/dev/download_qwen_models_python.sh 1.7B
```

## 偏好优化

从人工评估预测结果生成偏好数据：

```bash
bash scripts/data/build_preference_dataset.sh
```

14B DPO 训练：

```bash
bash scripts/train/train_qwen3_14b_dpo.sh
```

说明：

- 当前环境中 `trl==1.4.0` 可直接使用 `DPOTrainer`
- `ORPOTrainer` 当前环境不可直接用，所以仓库先接入了 `DPO`
- `GRPO` 属于更重的在线后训练，不是当前第一优先级
- 当前 DPO adapter 评估报告：
  - [outputs/eval_reports/manual_eval_50_qwen3_14b_dpo_report.json](/share/home/lifr/workspace/code/job-match-tune/outputs/eval_reports/manual_eval_50_qwen3_14b_dpo_report.json)

## 推理与评估

单条推理：

```bash
python -m jobmatch_tune.inference.predict \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --task jd_parse \
  --input examples/jd_ai_app.txt \
  --load-4bit
```

50 条人工评估：

```bash
PYTHONPATH=src python -m jobmatch_tune.eval.run_manual_eval \
  --dataset data/eval/jd_manual_eval_50.jsonl \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --out outputs/eval_reports/manual_eval_50_qwen3_14b_v3_report.json \
  --predictions-out outputs/eval_reports/manual_eval_50_qwen3_14b_v3_predictions.jsonl \
  --load-4bit
```

## 前后端分离应用

启动后端：

```bash
source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo
cd /share/home/lifr/workspace/code/job-match-tune
bash scripts/serve/start_api.sh
```

如需切到 `vLLM + OpenAI-compatible API + JSON Schema structured outputs`：

```bash
bash scripts/serve/start_vllm_server.sh

export JOBMATCH_INFERENCE_BACKEND=vllm
export JOBMATCH_VLLM_BASE_URL=http://127.0.0.1:8010/v1
export JOBMATCH_VLLM_MODEL=jobmatch-lora
bash scripts/serve/start_api.sh
```

启动前端：

```bash
cd /share/home/lifr/workspace/code/job-match-tune
bash scripts/serve/start_frontend.sh
```

端口转发：

```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 gpu03
```

浏览器打开 `http://localhost:5173`。

如需切回 1.7B：

```bash
export JOBMATCH_MODEL_PATH=models/Qwen3-1.7B
export JOBMATCH_ADAPTER_PATH=outputs/checkpoints/qwen3-1.7b-dft-lr1e-4
bash scripts/serve/start_api.sh
```

## 文档索引

- 技术方案原文：[微调方案.md](/share/home/lifr/workspace/code/job-match-tune/%E5%BE%AE%E8%B0%83%E6%96%B9%E6%A1%88.md)
- 项目实现与迭代总览：[docs/implementation_and_evolution.md](/share/home/lifr/workspace/code/job-match-tune/docs/implementation_and_evolution.md)
- 简历写法与项目亮点：[docs/resume_project_highlights.md](/share/home/lifr/workspace/code/job-match-tune/docs/resume_project_highlights.md)
- 数据来源：[docs/data_sources.md](/share/home/lifr/workspace/code/job-match-tune/docs/data_sources.md)
- 字节招聘 API 研究记录：[docs/bytedance_api_research.md](/share/home/lifr/workspace/code/job-match-tune/docs/bytedance_api_research.md)
- 岗位方向标注口径：[docs/job_direction_policy.md](/share/home/lifr/workspace/code/job-match-tune/docs/job_direction_policy.md)
- 历史实验记录：
  - [docs/history/experiment_results_2026-05-11.md](/share/home/lifr/workspace/code/job-match-tune/docs/history/experiment_results_2026-05-11.md)
  - [docs/history/incremental_sft_2026-05-13.md](/share/home/lifr/workspace/code/job-match-tune/docs/history/incremental_sft_2026-05-13.md)
  - [docs/history/manual_eval_2026-05-13.md](/share/home/lifr/workspace/code/job-match-tune/docs/history/manual_eval_2026-05-13.md)
