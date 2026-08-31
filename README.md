# JobMatchTune

JobMatchTune 是一个面向中文招聘场景的实验性人岗匹配项目，提供 JD 解析、简历解析、可解释匹配以及模型后训练与评测工具。

> 本项目仅用于学习、研究和演示，不应直接用于真实招聘决策。

## 主要功能

- 将 JD 和简历解析为结构化数据
- 结合规则与模型生成匹配结果和解释
- 支持文本、PDF、DOCX 及批量请求
- 提供 FastAPI 接口和 Vue 演示页面
- 包含数据处理、微调和评测脚本

## 快速开始

需要 Python 3.10+ 和 Node.js 18+。GPU 推理需要可用的 CUDA 环境及本地模型。

```bash
conda create -n tune-demo python=3.11 -y
conda activate tune-demo
pip install -r requirements.txt
npm ci --prefix frontend
```

配置模型路径并启动项目：

```bash
export JOBMATCH_MODEL_PATH=/path/to/model
bash scripts/serve/start_project.sh
```

启动后访问：

- Web：`http://127.0.0.1:5174`
- API：`http://127.0.0.1:8000`

停止项目：

```bash
bash scripts/serve/stop_project.sh
```

可通过环境变量调整模型、adapter、端口和推理后端。项目支持 Transformers 和 vLLM，vLLM 需单独安装。

## 主要接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 服务状态 |
| `POST` | `/api/parse` | JD 或简历解析 |
| `POST` | `/api/match` | 文本人岗匹配 |
| `POST` | `/api/match_files` | 文件人岗匹配 |
| `POST` | `/api/batch_parse` | 批量解析 |
| `POST` | `/api/batch_match` | 批量匹配 |

## 项目结构

```text
src/jobmatch_tune/  后端、数据、训练和评测代码
frontend/           Vue 前端
scripts/            启动、数据、训练和评测入口
configs/            可复现配置
tests/              自动化测试
```

## 测试

```bash
ruff check .
pytest -q
node tests/frontend_report_smoke.mjs
```

训练数据、SFT 和评测分别使用统一入口：

```bash
bash scripts/data/build_current_data_pools.sh
bash scripts/train/train_sft.sh
bash scripts/eval/run_product_adapter_suite.sh
```

DPO 是可选阶段，只有偏好数据通过质量门槛后才允许执行 `scripts/train/train_dpo.sh`。

## 数据与安全

`data/`、`models/` 和 `outputs/` 默认不纳入版本控制。请勿提交真实简历、密钥、模型权重或训练产物。服务默认仅监听本机，如需对外部署，应另行增加认证、限流和安全隔离。

## 许可

仓库当前未提供开源许可证。
