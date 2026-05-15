# Project Structure

## 目标

仓库按“数据 -> 训练 -> 推理 -> 服务”组织，默认只保留当前仍然有用的主链路入口。

## 顶层目录

- `configs/`
  - `label_schema.yaml`：岗位方向和技能 canonical schema
  - `train_qlora.yaml`：统一训练配置
  - `tencent_keywords.txt`：腾讯招聘批量抓取关键词
- `data/`
  - `eval/`：保留的人工 gold 和 hard case 数据
  - 其余 `raw/`、`interim/`、`sft/`、`*.sqlite3` 都是生成物
- `docs/`
  - 保存实验记录、标注口径和结构说明
- `examples/`
  - 推理 smoke 输入样例
- `frontend/`
  - 静态页面，不依赖前端构建工具
- `models/`
  - 本地基座模型目录，默认不纳入版本控制
- `outputs/`
  - checkpoints、日志、评估报告，默认不纳入版本控制
- `scripts/`
  - 当前保留的运行入口
- `src/jobmatch_tune/`
  - 业务代码

## `src/jobmatch_tune/` 分层

- `crawler/`
  - 公开招聘数据抓取
- `preprocess/`
  - 清洗、分段、去重、规则抽取
- `dataset/`
  - SFT / hard case 数据构造
- `train/`
  - QLoRA 训练入口
- `inference/`
  - prompt、模型加载、JSON 后处理
- `eval/`
  - 人工评估与指标统计
- `api/`
  - FastAPI 服务
- `utils/`
  - 下载和 IO 辅助

## `scripts/` 保留原则

保留两类脚本：

- 当前主链路入口
  - `start_api.sh`
  - `start_frontend.sh`
  - `train_qwen3_14b_smoke.sh`
  - `train_qwen3_14b_full.sh`
  - `download_qwen_models_python.sh`
- 数据准备入口
  - `init_db.sh`
  - `crawl_tencent.sh`
  - `crawl_tencent_large.sh`
  - `clean_jd.sh`
  - `build_sft.sh`

历史实验脚本仍可保留，但不作为默认推荐流程。
