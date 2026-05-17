# Project Structure

## 目标

仓库按“数据 -> 训练 -> 推理 -> 服务”组织，默认只保留当前主链路入口；历史实验和研究脚本单独归档，避免继续堆在根目录。

## 顶层目录

- `configs/`
  - 训练配置、标签 schema、招聘关键词、数据源清单
- `data/`
  - `eval/`：人工 gold 与 hard case
  - 其余 `raw/`、`interim/`、`sft/`、`sft_expanded/`、`preference/`、`*.sqlite3` 都是运行产物
- `docs/`
  - 项目说明、数据源说明、研究记录、历史实验
- `examples/`
  - JD / 简历示例文本
- `frontend/`
  - 静态前端页面
- `models/`
  - 本地模型目录，默认不纳入版本控制
- `outputs/`
  - checkpoint、日志、评估报告，默认不纳入版本控制
- `scripts/`
  - 可执行入口，按职责拆分
- `src/jobmatch_tune/`
  - 业务代码
- `tests/`
  - 单元测试与回归测试

## `src/jobmatch_tune/` 分层

- `crawler/`
  - 腾讯、百度、京东、Moka、公开导出数据导入
- `preprocess/`
  - 清洗、分段、去重、规则抽取
- `dataset/`
  - 默认 SFT、扩展 SFT、偏好数据构造
- `train/`
  - QLoRA / DPO 训练入口
- `inference/`
  - prompt、推理、JSON 后处理、structured output
- `eval/`
  - 人工评估、数据审计、指标统计
- `api/`
  - FastAPI 服务
- `utils/`
  - 下载、IO 等辅助逻辑

## `scripts/` 分层

- `scripts/data/`
  - 数据抓取、导入、重建主链路
  - 当前主入口：
    - `refresh_official_job_data.sh`
    - `refresh_moka_data.sh`
    - `refresh_jd_data.sh`
    - `refresh_baidu_data.sh`
    - `refresh_tencent_data.sh`
    - `import_public_job_exports.sh`
    - `import_chinese_job_exports.sh`
    - `rebuild_data_pipeline.sh`
    - `build_preference_dataset.sh`
    - `build_multilingual_weak_sft.sh`
- `scripts/train/`
  - 14B smoke / full / DPO 训练入口
- `scripts/serve/`
  - API、vLLM、静态前端服务启动
- `scripts/dev/`
  - 环境安装与模型下载
- `scripts/research/`
  - 研究辅助脚本，例如字节 `_signature` 复现
- `scripts/legacy/`
  - 1.7B / 旧实验脚本归档，不再作为默认推荐流程

## 删除与保留原则

当前已经删除这类过时脚本：

- 只包一层旧命令、且已被新主入口覆盖的 wrapper
- 仍指向旧流程、旧 schema、旧目录结构的入口
- 已经被 `scripts/data/rebuild_data_pipeline.sh` 吸收的零散步骤脚本

当前保留的标准：

- 必须服务于当前主链路，或者
- 明确属于 `legacy` / `research`，且能解释其存在理由

## 当前推荐入口

数据刷新：

```bash
bash scripts/data/refresh_official_job_data.sh
```

重建默认训练集：

```bash
bash scripts/data/rebuild_data_pipeline.sh
```

14B 训练：

```bash
bash scripts/train/train_qwen3_14b_full.sh
```

启动 API：

```bash
bash scripts/serve/start_api.sh
```

启动前端：

```bash
bash scripts/serve/start_frontend.sh
```
