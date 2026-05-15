# JobMatchTune

面向招聘 JD / 简历结构化抽取的 Qwen3 QLoRA 微调项目。当前默认服务版本为 `Qwen3-14B + LoRA adapter + 规则后处理`。

## 当前默认版本

- 基座模型：`models/Qwen3-14B`
- Adapter：`outputs/checkpoints/qwen3-14b-jobmatch-qlora`
- 服务默认入口：
  - API: [src/jobmatch_tune/api/server.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/api/server.py)
  - 启动脚本: [scripts/start_api.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/start_api.sh)
  - vLLM 服务脚本: [scripts/start_vllm_server.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/start_vllm_server.sh)
- 50 条人工 holdout 最新报告：
  - [outputs/eval_reports/manual_eval_50_qwen3_14b_v3_report.json](/share/home/lifr/workspace/code/job-match-tune/outputs/eval_reports/manual_eval_50_qwen3_14b_v3_report.json)

## 项目结构

详细说明见 [docs/project_structure.md](/share/home/lifr/workspace/code/job-match-tune/docs/project_structure.md)。

核心目录：

- `src/jobmatch_tune/`
  - `crawler/`：公开 JD 抓取
  - `preprocess/`：清洗、去重、规则抽取
  - `dataset/`：SFT 数据构造
  - `train/`：QLoRA 训练
  - `inference/`：推理与后处理
  - `api/`：FastAPI 服务
  - `eval/`：人工评估与指标
- `scripts/`：当前保留的可执行入口
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

清洗与构造训练集：

```bash
python -m jobmatch_tune.preprocess.normalize_jd \
  --db data/jobmatch_tune.sqlite3 \
  --out data/interim/jd_clean.jsonl \
  --schema configs/label_schema.yaml

python -m jobmatch_tune.dataset.build_sft_dataset \
  --jd data/interim/jd_clean.jsonl \
  --out-dir data/sft
```

## 训练

14B smoke：

```bash
bash scripts/train_qwen3_14b_smoke.sh
```

14B 正式训练：

```bash
bash scripts/train_qwen3_14b_full.sh
```

如需下载模型快照：

```bash
bash scripts/download_qwen_models_python.sh 14B
```

轻量回退模型：

```bash
bash scripts/download_qwen_models_python.sh 1.7B
```

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
bash scripts/start_api.sh
```

如需切到 `vLLM + OpenAI-compatible API + JSON Schema structured outputs`：

```bash
bash scripts/start_vllm_server.sh

export JOBMATCH_INFERENCE_BACKEND=vllm
export JOBMATCH_VLLM_BASE_URL=http://127.0.0.1:8010/v1
export JOBMATCH_VLLM_MODEL=jobmatch-lora
bash scripts/start_api.sh
```

启动前端：

```bash
cd /share/home/lifr/workspace/code/job-match-tune
bash scripts/start_frontend.sh
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
bash scripts/start_api.sh
```

## 文档索引

- 技术方案原文：[微调方案.md](/share/home/lifr/workspace/code/job-match-tune/%E5%BE%AE%E8%B0%83%E6%96%B9%E6%A1%88.md)
- 项目实现与迭代总览：[docs/implementation_and_evolution.md](/share/home/lifr/workspace/code/job-match-tune/docs/implementation_and_evolution.md)
- 简历写法与项目亮点：[docs/resume_project_highlights.md](/share/home/lifr/workspace/code/job-match-tune/docs/resume_project_highlights.md)
- 数据来源：[docs/data_sources.md](/share/home/lifr/workspace/code/job-match-tune/docs/data_sources.md)
- 岗位方向标注口径：[docs/job_direction_policy.md](/share/home/lifr/workspace/code/job-match-tune/docs/job_direction_policy.md)
- 14B 之前的实验记录：
  - [docs/experiment_results_2026-05-11.md](/share/home/lifr/workspace/code/job-match-tune/docs/experiment_results_2026-05-11.md)
  - [docs/incremental_sft_2026-05-13.md](/share/home/lifr/workspace/code/job-match-tune/docs/incremental_sft_2026-05-13.md)
  - [docs/manual_eval_2026-05-13.md](/share/home/lifr/workspace/code/job-match-tune/docs/manual_eval_2026-05-13.md)
