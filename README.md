# JobMatchTune

面向中文招聘场景的模型后训练与评测工作台，覆盖 JD 解析、简历解析和可解释的人岗匹配。当前默认推理组合是 `Qwen3-14B + LoRA adapter + 规则后处理`。

它的核心价值在于一条可审计的实验闭环：公开招聘数据进入清洗和分层流程，经 SFT 和可选偏好训练后，用人工留出评测集（holdout）、隐私门禁、数据泄漏检查和产品回归报告决定 adapter 是否可晋级。它目前不是完整 ATS，也不应被当作可直接公网部署的招聘决策系统。

## 当前能力

- `jd_parse`：抽取岗位方向、职责、技能、经验和学历要求。
- `resume_parse`：解析文本、DOCX、文本型 PDF；图片和扫描 PDF 支持 OCR sidecar。
- `match`：组合 JD/简历结构化结果、确定性规则分数和模型生成的解释建议。
- 单条、批量、文本与文件混合 API。
- Vue 3 静态工作台和 Markdown 报告下载。
- QLoRA、DPO、run manifest、训练前 readiness 和产品回归门禁。

默认本地资产：

```text
models/Qwen3-14B
outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601
```

模型、adapter、生成数据和评测输出不随 Git 分发。默认路径可通过 `JOBMATCH_MODEL_PATH`、`JOBMATCH_ADAPTER_PATH` 等环境变量覆盖。

## 工作链路

```text
公开 JD / 合法脱敏简历 / 人工标注
                │
        清洗、标准化、去重、审计
                │
       SFT / preference 数据构造
                │
       QLoRA SFT → 可选 DPO
                │
   人工留出评测集 + 产品回归 + readiness
                │
Transformers 或 vLLM → FastAPI → Web UI
```

“人工留出评测集（holdout）”是提前单独保留、由人工给出正确答案、且不参加训练和调参的一组样本。它相当于最终考试卷，用于检验模型面对未见数据时是否真的变好。发现其中的错例后，不应把同一批答案直接回灌训练，否则再次评测只是在检查模型是否记住了答案。

匹配请求会先分别解析 JD 和简历，再计算技能、方向、学历、年限和项目命中的规则结果，最后让模型基于原文和规则结果生成解释。因此一次匹配当前需要三次生成调用；这保证了可解释性，但延迟和吞吐仍属于实验级。

## 快速开始

推荐 Python 3.11 和 CUDA 环境：

```bash
conda create -n tune-demo python=3.11 -y
conda activate tune-demo
pip install -r requirements.txt
```

`pyproject.toml` 是依赖元数据的唯一来源；`requirements.txt` 只是兼容入口。依赖目前只有最低版本约束、没有 lock file，因此历史训练环境还不能做到字节级复现。

下载默认模型：

```bash
bash scripts/dev/download_qwen_models_python.sh 14B
```

推荐用统一脚本在后台启动 API 和静态前端：

```bash
bash scripts/serve/start_project.sh
```

停止项目：

```bash
bash scripts/serve/stop_project.sh
```

默认地址是 `http://localhost:8000` 和 `http://localhost:5174`，日志写入 `outputs/logs/`，PID 写入 `outputs/runtime/`。本机 `5173` 已由另一个项目使用，所以统一脚本固定使用 `5174`。前端依赖 CDN 版 Vue 3，因此首次打开需要网络。通过登录节点访问 GPU 节点时的双层端口转发见 [项目启动与访问](docs/项目启动与访问.md)。

vLLM 需要另行安装；统一脚本可以同时管理 vLLM、API 和前端：

```bash
JOBMATCH_INFERENCE_BACKEND=vllm bash scripts/serve/start_project.sh
```

vLLM 模式会并行提交一次匹配中的 JD/简历解析，并对批量请求实施受控并发；默认 Transformers 4-bit 路径仍保持串行，以降低单 GPU 显存风险。实现与压测方法见[核心推理流程优化](docs/核心推理流程优化_2026-08-09.md)。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 进程、CUDA 和后端状态 |
| `POST` | `/api/warmup` | 显式加载模型 |
| `POST` | `/api/parse` | JD 或简历文本解析 |
| `POST` | `/api/jd_file_parse` | JD 文件解析 |
| `POST` | `/api/resume_file_parse` | 简历文件解析 |
| `POST` | `/api/match` | 文本人岗匹配 |
| `POST` | `/api/match_files` | 文本/文件混合匹配 |
| `POST` | `/api/batch_parse` | 最多 64 条顺序解析 |
| `POST` | `/api/batch_match` | 最多 32 组顺序匹配 |

文本请求限制为 20,000 字符，单文件默认最多 10 MiB（`JOBMATCH_MAX_UPLOAD_BYTES` 可覆盖），默认 CORS 只接受本机来源（`JOBMATCH_CORS_ORIGINS` 可覆盖）。当前还没有认证、限流、MIME 验证或持久化审计，不要直接暴露到公网。

## 数据与训练

初始化数据库并刷新官方公开职位：

```bash
python -m jobmatch_tune.init_db --db data/jobmatch_tune.sqlite3
bash scripts/data/refresh_official_job_data.sh
bash scripts/data/rebuild_data_pipeline.sh
```

从源数据构建 JD、简历、匹配、多任务 SFT 和两套 preference 数据，并生成最新门禁报告：

```bash
bash scripts/data/build_current_data_pools.sh
```

独立下载和处理外部招聘技能 gold（不自动混入主训练）：

```bash
bash scripts/data/prepare_external_skill_gold.sh
```

正式训练按阶段检查对应字段，避免一个阶段被无关数据误拦：

```text
outputs/eval_reports/data_readiness_report.json
SFT:         summary.all_ready_for_sft = true
JD DPO:      summary.ready_for_dpo = true
产品 DPO:    summary.ready_for_product_dpo = true
完整链路:    summary.all_ready_for_training = true
```

这些是允许训练的目标状态，不是当前状态。2026-08-12 简化模板并完整重建后，Resume SFT 为 15,464 条、2,652 个有效来源组，其中 2,524 个来源组（95.17%）仍来自 bootstrap；多任务 Resume 来源组比例为 train 75.75%、valid 78.70%。Match 中 1,996 条明确年限要求样本又全部为“不满足”，因此当前 `not_ready_tasks=[resume, match, multitask]`；不要绕过 readiness 启动新训练。FairCV 因许可未确认且只允许候选审计，已从 Resume/Match 训练链路移除。

14B smoke 和 SFT：

```bash
bash scripts/train/train_qwen3_14b_smoke.sh
bash scripts/train/train_qwen3_14b_multitask_sft.sh
```

新增 DPO 当前暂停：现有 preference 全为合成数据，且 Match Gold 尚未完成人工复核。训练脚本默认拒绝 DPO，恢复条件和人工审核方法见[人岗匹配持续质量目标](docs/人岗匹配持续质量目标.md)。

训练入口会写入 `run_manifest.json`，记录 Git commit、配置和数据 hash、任务构成、原始来源多样性、偏好数据出处、readiness 摘要和 CLI 覆盖项。正式 adapter 仍需同时通过绝对产品阈值与基线回归阈值：

```bash
ADAPTER_PATH=outputs/checkpoints/<adapter> \
TAG=<tag> \
BASELINE_TAG=qwen3_14b_dft_dpo_final_20260602 \
bash scripts/eval/run_product_adapter_suite.sh
```

## 数据边界

Git 保留评测候选的 builder 和审核契约；当前 `data/` 整体属于本地运行资产，不纳入版本控制。人工审核文件默认放在 `data/private/`，避免误提交真实简历或标注信息：

- `models/`、`outputs/`；
- SQLite、raw、interim、SFT、preference 数据；
- `data/eval/` 下由 builder 生成的 candidate/train pool。

真实简历不得提交到仓库。私有样例应放在 Git 忽略目录中，先经过授权、PII 审计与脱敏，再决定是否进入标注流程：

```bash
bash scripts/data/resume_privacy_audit.sh \
  --input /private/path/resume.pdf \
  --report-out outputs/eval_reports/resume_privacy_report.json \
  --out outputs/eval_reports/resume_sanitized.jsonl
```

## 验证

```bash
ruff check .
pytest -q
node tests/frontend_report_smoke.mjs
python -m compileall -q src
```

测试主要覆盖规则、数据 builder、评测、隐私处理和 API service helper；真实 14B GPU 推理回归需要单独运行，不包含在日常单元测试中。

## 目录

- `src/jobmatch_tune/`：crawler、预处理、数据构造、训练、推理、匹配、评测和 API。
- `scripts/`：当前可执行数据、训练、服务和评测入口。
- `configs/`：schema、数据源、数据配比与当前 14B 训练配置。
- `data/eval/`：本地生成的评测候选、seed 和派生数据；可复现定义位于 `src/jobmatch_tune/eval/`。
- `frontend/`：无构建步骤的 Vue 3 ESM 工作台。
- `docs/`：架构审查、数据与历史实验文档。

详细目录职责见 [项目结构](docs/项目结构.md)，本轮产品与技术审查见 [产品与技术审查](docs/产品与技术审查_2026-08-09.md)，后训练与数据优化见 [后训练与数据流程优化](docs/后训练与数据流程优化_2026-08-09.md)，外部技能 gold 处理见 [外部技能标注数据处理](docs/外部技能标注数据处理.md)，中文匹配数据的来源与许可判断见 [中文人岗匹配数据核验](docs/中文人岗匹配数据核验.md)，产品主流程与后续交互优先级见 [产品交互与优化目标](docs/产品交互与优化目标.md)。

## 当前限制

- 匹配技能仍以 taxonomy 归一化后的精确集合重叠为主，没有语义召回或 reranker。
- Transformers 后端用进程内锁串行生成；vLLM 后端已支持 JD/简历并行和受控批量并发，但当前环境尚未安装 vLLM，仍需真实 GPU 对照。
- API service、上传解析和路由集中在单个较大的模块中，响应没有统一 Pydantic contract。
- 前端依赖 CDN，没有版本锁定、构建产物和浏览器自动化测试。
- 仓库没有 CI、容器定义、依赖 lock file、许可证和安全策略。
- Match 候选留出集已有 25 条且与训练池无重合，但尚未完成人工复核；历史 Match 分数不能等价为真实招聘决策质量。
- Match 配对池已经清除个人资料与敏感字段，但仍为 100% 合成配对；Resume 和多任务来源多样性门禁当前未通过。

## 文档

- [产品与技术审查](docs/产品与技术审查_2026-08-09.md)
- [产品交互与优化目标](docs/产品交互与优化目标.md)
- [项目结构](docs/项目结构.md)
- [项目启动与访问](docs/项目启动与访问.md)
- [核心推理流程优化](docs/核心推理流程优化_2026-08-09.md)
- [项目后续计划](docs/项目后续计划.md)
- [数据处理流程](docs/数据处理流程.md)
- [简历处理流程](docs/简历处理流程.md)
- [岗位方向标注口径](docs/岗位方向标注口径.md)
- [公开数据源清单](docs/公开数据源清单.md)
- [中文人岗匹配数据核验](docs/中文人岗匹配数据核验.md)
- [人岗匹配持续质量目标](docs/人岗匹配持续质量目标.md)
- [训练评测记录（2026-06-01）](docs/训练评测记录_2026-06-01.md)
- [项目建设历程](docs/项目建设历程.md)

## 许可状态

仓库当前没有 `LICENSE`。在明确选择许可证之前，不应把“代码可见”等同于“已授权开源复用”。
