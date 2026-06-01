# 项目从 0 到当前版本的来龙去脉

更新时间：2026-06-01

这份文档按项目真实推进顺序整理：最开始为什么做，第一步怎么落地，后面为什么不断调整数据、模型、训练和应用形态。它不是只列最终命令，而是解释每个阶段的目标、做法、产物和当时踩到的问题。

当前项目已经从一个单纯的 JD 解析微调 demo，迭代成一套面向招聘场景的多任务系统：

- JD 结构化解析
- 简历结构化解析
- JD 与简历匹配度分析
- 中文招聘数据抓取、导入、清洗、审计和建池
- Qwen3-14B QLoRA 微调
- FastAPI 服务与前端演示
- 训练前数据 readiness 门控

---

## 1. 起点：先做 JD 解析，而不是一上来做大系统

项目最初的目标很具体：把中文招聘 JD 解析成固定 JSON。

最早定义的 JD 输出字段是：

- `岗位方向`
- `核心职责`
- `必备技能`
- `加分项`
- `经验要求`
- `学历要求`

选择这个任务作为起点，是因为它有几个适合 SFT 的特点：

1. 输入是非结构化文本，但输出 schema 固定。
2. 错误类型比较稳定，方便通过数据和规则迭代。
3. 中文招聘 JD 有足够多公开数据源。
4. 解析结果后续可以直接用于推荐、筛选、简历匹配和前端展示。

所以项目一开始没有只写 prompt，而是围绕一条可迭代链路展开：

```text
公开 JD -> 原始数据 -> 清洗去重 -> 规则标注 -> SFT 数据 -> 微调 -> 评估 -> API 应用
```

这个选择决定了后面项目一直围绕“数据闭环”推进，而不是只追模型参数。

---

## 2. 第一阶段：搭环境和最小工程骨架

早期先把 Python 工程结构和运行环境固定下来。

当前环境约定是：

```bash
conda create -n tune-demo python=3.11 -y
conda activate tune-demo
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

后来单独维护 `requirements.txt`，原因是：

- 新节点可以复现环境。
- 依赖版本能随代码提交。
- 训练、评估、API、爬虫脚本共享同一环境。
- 项目可以作为完整工程展示，而不是散落脚本。

GPU 资源检查使用：

```bash
nvidia-smi
ssh -n gpu03 nvidia-smi
```

当时确认 `gpu03` 有 3 张 L20，这直接影响了后续模型路线：可以做 14B QLoRA，但不适合全参微调。

---

## 3. 第二阶段：先用腾讯招聘跑通第一条数据链路

项目最早没有盲目抓很多网站，而是先选腾讯招聘作为第一条稳定源。

选择腾讯的原因：

- 技术岗位多。
- 公开接口相对稳定。
- JD 正文结构比较完整。
- 职责、要求、学历、经验等字段相对容易抽取。

相关文件：

- `src/jobmatch_tune/crawler/tencent_careers.py`
- `configs/tencent_keywords.txt`
- `scripts/data/refresh_tencent_data.sh`

当时做法：

1. 按技术关键词请求腾讯招聘公开接口。
2. 分页获取职位列表。
3. 拉取职位详情，补全岗位职责和任职要求。
4. 写入 `data/raw/tencent_jd_raw.jsonl`。
5. 同步写入 `data/jobmatch_tune.sqlite3`。
6. 再进入清洗、去重、规则标注和 SFT 构造。

这个阶段最重要的成果不是数据量，而是验证了工程骨架：

```text
crawler -> raw JSONL / SQLite -> preprocess -> dataset builder -> train/eval
```

后面所有数据源都沿用这套模式。

---

## 4. 第三阶段：扩展更多中文官网数据源

腾讯链路跑通后，开始扩展更多中文官网源。扩源原则不是“能抓就抓”，而是优先满足：

1. 匿名可访问。
2. 接口或页面结构稳定。
3. JD 正文完整。
4. 技术岗占比高。
5. 可以重复运行，不依赖登录态。

### 4.1 百度招聘

百度招聘不是简单 JSON 列表，而是从页面 SSR 数据里解析职位。

相关文件：

- `src/jobmatch_tune/crawler/baidu_talent.py`
- `configs/baidu_keywords.txt`
- `scripts/data/refresh_baidu_data.sh`

做法：

1. 按关键词访问搜索页。
2. 解析页面里的 `window.__INITIAL_DATA__`。
3. 提取职位列表和职位详情。
4. 映射成项目统一 raw schema。

### 4.2 京东招聘

京东招聘后来成为重要来源，因为匿名接口比较稳定。

相关文件：

- `src/jobmatch_tune/crawler/jd_careers.py`
- `scripts/data/refresh_jd_data.sh`

做法：

1. 请求京东招聘匿名职位列表接口。
2. 分页拉取职位。
3. 请求详情接口。
4. 写入 raw JSONL 和 SQLite。

### 4.3 Moka 托管招聘官网

Moka 是后续中文官网扩量最重要的方向之一。很多技术公司把招聘官网托管在 Moka，接口形态比较统一，因此只要维护公司源配置，就能复用同一个抓取器。

相关文件：

- `src/jobmatch_tune/crawler/moka_careers.py`
- `configs/moka_sources.yaml`
- `scripts/data/refresh_moka_data.sh`

做法：

1. 在 `configs/moka_sources.yaml` 维护公司招聘源。
2. 对每个公司源拉职位列表。
3. 拉详情补全文本。
4. 写入 `data/raw/moka_jd_raw.jsonl` 和 SQLite。

Moka 扩源时优先补过这些方向：

- AI
- 芯片
- 自动驾驶
- 云基础设施
- 安全
- 游戏
- 金融科技
- 高性能计算

### 4.4 其他官网源和探测脚本

后续还接入或研究过：

- 携程
- 小米
- 美团
- 滴滴
- 拼多多
- 得物
- 小红书
- 蚂蚁
- B 站
- 字节

能稳定匿名抓取并复现的源进入正式刷新脚本；只能探测、接口不稳定或依赖风控状态的源，保留为 probe 脚本。

这就是为什么 `scripts/data/` 里既有 `refresh_*.sh`，也有 `probe_*.sh`：

- `refresh_*.sh`：已纳入正式数据链路。
- `probe_*.sh`：只做接口可用性研究，不直接进入训练集。

---

## 5. 第四阶段：评估 Boss、智联等招聘平台后，转向官网和公开数据集

项目过程中评估过 Boss 直聘、智联招聘等招聘平台，也研究过一些大厂官网。

最终没有把所有平台硬接进主线，原因是：

- 很多站点需要登录。
- 存在强风控和接口签名。
- 匿名直调不稳定。
- 一次性抓到的数据不一定能复现。
- 平台数据字段未必比企业官网 JD 更干净。

后续策略调整为：

1. 优先中文企业官网。
2. 优先 Moka 等 ATS 托管招聘系统。
3. 优先公开、匿名、可复现接口。
4. 对公共数据集先导入为候选池，不直接进入默认训练集。

这个调整让项目从“尽量多抓”变成“可持续积累高质量数据”。

---

## 6. 第五阶段：接入公开数据集，把 raw pool 做大

官网数据质量较高，但增长慢。为了扩大原始语料池，项目接入了公开职位数据集和导出文件。

相关文件：

- `src/jobmatch_tune/crawler/import_public_job_data.py`
- `configs/public_job_sources.yaml`
- `configs/public_job_sources_zh_large.yaml`
- `scripts/data/import_public_job_exports.sh`
- `scripts/data/import_chinese_job_exports.sh`

接入过的数据包括：

- GitHub `jhcoco/bosszp`
- GitHub `WorkAggregation`
- Hugging Face `open-apply-jobs`
- Hugging Face 中文招聘学历数据

这一阶段形成了一个关键认知：

```text
raw 数据多，不等于高质量 SFT 数据多。
```

公开数据集里有大量噪声：

- 英文 JD。
- 非技术岗。
- 教师、培训、销售、运营、编导等岗位。
- 字段弱标注。
- 文本短或结构缺失。
- 重复岗位。

所以公开数据集先进入 raw pool、candidate pool 或扩展集，而不是直接混入默认高质量训练集。

---

## 7. 第六阶段：建立 raw、clean、dedup、candidate、sft 分层

数据越来越多后，项目开始明确区分不同数据层。

### 7.1 raw 层

raw 层保存原始事实。

主要位置：

- `data/raw/*.jsonl`
- `data/jobmatch_tune.sqlite3`

原则：

- 尽量保留来源原貌。
- 只做最小字段映射。
- 不在 raw 层过度清洗。

### 7.2 clean 层

clean 层把不同来源的 JD 统一成同一个 schema。

相关文件：

- `src/jobmatch_tune/preprocess/clean_text.py`
- `src/jobmatch_tune/preprocess/normalize_jd.py`
- `src/jobmatch_tune/preprocess/jd_field_rules.py`
- `scripts/data/rebuild_data_pipeline.sh`

处理内容：

1. 清理 HTML、特殊空白和页面噪声。
2. 标准化标题、公司、地点、正文和来源。
3. 拆分职责、要求、加分项。
4. 规则抽取学历、经验、技能和岗位方向。
5. 标记是否满足 `sft_ready`。

### 7.3 dedup 层

不同来源会重复收录同一职位，因此需要去重。

相关文件：

- `src/jobmatch_tune/preprocess/deduplicate.py`

处理方式：

- 基于标题、公司、正文摘要等信号去重。
- 输出 `data/interim/jd_clean_dedup.jsonl`。

### 7.4 candidate pool

候选池不是训练集，而是“可能值得进入训练集的储备池”。

例如：

- `data/eval/public_jd_candidate_pool.jsonl`
- `data/eval/jd_train_pool_combined.jsonl`
- `data/eval/resume_train_pool_combined.jsonl`
- `data/eval/match_train_pool_combined.jsonl`

候选池的作用是：

- 从大规模 raw 数据里筛掉明显无效样本。
- 为后续严格训练集扩容提供候选。
- 保留来源和质量信息，方便审计。

### 7.5 sft 层

sft 层才是真正进入训练的数据。

当前核心 SFT 数据包括：

- `data/sft_jd_quality/`
- `data/sft_resume/`
- `data/sft_match/`
- `data/sft_multitask/`

当前规模：

| 数据集 | train | valid | test |
| --- | ---: | ---: | ---: |
| JD quality | 4400 | 550 | 550 |
| resume | 38408 | 4850 | 4890 |
| match | 3917 | 486 | 493 |
| multitask | 9800 | 1208 | - |

这里要特别注意：默认训练不是把所有 sft 子任务全量拼一起，而是通过 `configs/dataset_registry.yaml` 控制多任务采样比例。

---

## 8. 第七阶段：岗位方向口径反复修正

早期岗位方向只覆盖少数几类：

- 前端
- 后端
- 测试
- 算法
- AI 应用

随着真实 JD 增加，发现技术岗位远不止这些，因此逐步扩展：

- 网络与基础设施
- 高性能计算
- AI Infra
- 运维开发
- 硬件研发
- 嵌入式开发
- 客户端开发
- 安全工程
- 汽车软件 / 智驾研发
- 数据开发

相关文档：

- `docs/job_direction_policy.md`

这一阶段得出的经验是：

```text
岗位方向口径不稳定时，不应该盲目增加训练轮数。
```

当时先做了：

1. 人工 gold 样本。
2. 前端、后端、测试、算法、AI 应用边界样本。
3. hard case 样本。
4. 规则修复。
5. 再决定是否重训。

这个流程后来成为项目处理字段边界问题的通用方法。

---

## 9. 第八阶段：从普通 SFT 集升级到 JD quality 集

早期默认 JD 训练集规模不断扩大，但复查后发现：如果把弱标注和公开数据直接混进主训练集，模型会学到噪声。

典型问题包括：

- 非技术岗混入。
- 教师、培训师、编导、销售、运营混入。
- `经验要求` 里混进学历。
- 职责和要求粘连。
- 岗位方向弱标注不稳。
- 文本太短但被当成完整 JD。

因此后来把 JD 训练数据拆成多层：

- `data/sft/`：早期严格集。
- `data/sft_jd_strict_plus/`：从候选池回收的较高质量扩展层。
- `data/sft_jd_quality/`：当前 JD 主质量集。

当前 `data/sft_jd_quality/` 的构建策略是：

1. 先放入 strict 样本。
2. 再补 strict_plus。
3. 再补 quality_weak。
4. 当前不使用 bootstrap 兜底。

最近一次分层统计：

- `strict = 3200`
- `strict_plus = 260`
- `quality_weak = 2040`
- `bootstrap = 0`

这说明当前 JD quality 并不是“随便扩出来的 5500 条”，而是分层回收后的质量集。

---

## 10. 第九阶段：增加 JD 质量审计和风险门控

为了避免低质量样本进入训练，项目增加了训练前质量审计。

相关文件：

- `src/jobmatch_tune/dataset/jd_quality_risk.py`
- `src/jobmatch_tune/eval/report_jd_quality_risks.py`
- `scripts/data/report_jd_quality_risks.sh`

风险门控检查：

- 可疑非技术标题。
- `quality_weak` 层样本。
- 弱公开来源。
- 学历、经验、技能、职责字段为空。
- 职责/要求边界泄漏。
- 单条职责过长。

同时，每条 JD quality 样本的 `meta` 里记录：

- `quality_tier`
- `quality_reason`
- `quality_risk_score`
- `quality_risk_reasons`
- `quality_score`

这样做的价值是：

1. 样本不是黑盒。
2. 可以按分数抽样复核。
3. 可以解释为什么某条数据进入训练。
4. 后续可以做采样加权或剔除。

最近一次 readiness 报告显示：

- `all_ready_for_training = true`
- JD quality 达到当前训练门槛。
- 高风险样本已被过滤。

---

## 11. 第十阶段：扩展到简历解析

用户需求后来从 JD 解析扩展到完整招聘应用，因此必须处理简历。

简历和 JD 最大区别是：简历来源不是稳定网页，而是多种文件格式。

真实简历可能是：

- txt
- docx
- 文本型 pdf
- 扫描版 pdf
- 图片简历
- 双栏简历
- 表格模板
- 中英混排简历

因此项目把简历链路拆成两层：

```text
文档解析 / OCR / 文本恢复 -> 简历结构化抽取
```

当前实现文件：

- `src/jobmatch_tune/resume/ingest.py`
- `src/jobmatch_tune/resume/normalize.py`
- `src/jobmatch_tune/resume/ocr.py`
- `scripts/data/resume_ingest.sh`
- `scripts/data/resume_normalize.sh`
- `scripts/data/resume_ocr_sidecar.sh`

当前支持：

- `txt` 直接读取。
- `docx` 通过 `python-docx` 提取段落。
- `pdf` 通过 `pypdf` 抽文本，并判断文本型、弱文本型或扫描型。
- 图片和扫描 PDF 支持 OCR sidecar 文本。

简历结构化 schema 包括：

- `目标岗位`
- `教育背景`
- `核心技能`
- `实习经历`
- `项目经历`
- `优势标签`

当前 `data/sft_resume/` 规模为：

- train：38408
- valid：4850
- test：4890

需要注意：这批简历数据里有大量模板化增强样本，因此它的训练价值不等同于 4.8 万份真实简历。后续应继续补充更真实的脱敏简历、公开可用简历样本和 OCR 场景样本。

---

## 12. 第十一阶段：扩展到 JD 与简历匹配分析

当 JD 和简历都能结构化后，下一步就是匹配分析。

匹配任务不是简单输出一个分数，而是要解释：

- 总体匹配等级
- 匹配分数
- 匹配优势
- 风险点
- 技能缺口
- 建议补强方向

相关数据构造脚本：

- `scripts/data/build_match_eval_dataset.sh`
- `scripts/data/build_match_train_pool_synthetic.sh`
- `scripts/data/build_match_train_pool_combined.sh`
- `scripts/data/build_match_sft_dataset.sh`

当前 `data/sft_match/` 规模为：

- train：3917
- valid：486
- test：493

这条线目前以合成匹配数据为主：用现有 JD 和简历样本组合，按岗位方向、技能重合、经验学历等规则生成正负样本和匹配等级。

它的价值是快速建立第三个任务，但后续仍需要真实招聘场景的人工评估样本来校准分数和解释质量。

---

## 13. 第十二阶段：从单任务变成多任务 SFT

当任务从 JD 扩展到 resume 和 match 后，项目不再只训练单一 JD 解析模型，而是构造多任务训练集。

相关文件：

- `configs/dataset_registry.yaml`
- `src/jobmatch_tune/dataset/build_multitask_sft_dataset.py`
- `scripts/data/build_multitask_sft_dataset.sh`

当前多任务数据规模：

- `data/sft_multitask/train.jsonl = 9800`
- `data/sft_multitask/valid.jsonl = 1208`

当前训练不会把 `data/sft_resume/` 的 4.8 万条全量混入，而是通过 registry 控制采样比例。这样做是为了避免 resume 模板化样本压过 JD 和 match。

这个阶段形成的原则是：

```text
多任务训练不是简单拼接，而是要控制任务比例和数据质量。
```

---

## 14. 第十三阶段：模型路线从小模型验证升级到 Qwen3-14B

早期可以用小模型验证训练代码和数据格式，但项目目标不是只做 toy demo。

后来用户明确要求使用 14B，并手动下载了：

```bash
modelscope download --model Qwen/Qwen3-14B --local_dir ./Qwen3-14B
```

模型位于：

```text
models/Qwen3-14B
```

当前默认方案：

- 基座模型：`Qwen3-14B`
- 微调方式：`4-bit QLoRA`
- 训练框架：`Transformers + PEFT + TRL + Accelerate`
- 服务推理：`vLLM + OpenAI-compatible API`
- 本地推理：`Transformers + PEFT`

相关训练脚本：

- `scripts/train/train_qwen3_14b_full.sh`
- `scripts/train/train_qwen3_14b_multitask_sft.sh`
- `scripts/train/train_qwen3_14b_resume_sft.sh`
- `scripts/train/train_qwen3_14b_dpo.sh`
- `scripts/train/train_qwen3_14b_smoke.sh`

项目中也逐步接入或预留了更现代的训练技术：

- QLoRA / 4-bit quantization
- LoRA adapter
- assistant-only loss
- DFT loss
- packing
- gradient checkpointing
- Liger Kernel
- DPO preference tuning

这些技术不是为了堆栈好看，而是为了在有限 GPU 上训练 14B，并为后续偏好优化留接口。

---

## 15. 第十四阶段：推理、后处理、API 和前端

训练完成后，项目不只停在训练脚本，而是继续做了应用层。

### 15.1 推理与后处理

推理输出需要保证 JSON 可用，因此增加了：

- JSON 修复。
- 字段归一化。
- 职责/要求/技能/学历/经验后处理。
- 岗位方向规则兜底。

相关目录：

- `src/jobmatch_tune/inference/`
- `src/jobmatch_tune/api/`

### 15.2 API 服务

当前服务入口：

- `src/jobmatch_tune/api/server.py`
- `scripts/serve/start_api.sh`
- `scripts/serve/start_vllm_server.sh`

API 目标是提供：

- JD 解析。
- 简历解析。
- 简历文件上传解析。
- JD 与简历匹配分析。

### 15.3 前端演示

前端目录：

- `frontend/`

前端从最早的简单 HTML/JS/CSS，后来按需求迁移成 Vue 3 ESM 结构，用于演示：

- JD 输入与解析。
- 简历输入与解析。
- 简历文件上传。
- JD 与简历匹配结果展示。

---

## 16. 第十五阶段：项目结构和脚本整理

随着文件变多，项目做过多次结构优化。

当前核心结构：

```text
src/jobmatch_tune/
  api/          FastAPI 服务
  crawler/      JD 和公开数据抓取 / 导入
  dataset/      SFT、DPO、候选池和多任务数据构造
  eval/         人工评估、质量审计、readiness 报告
  inference/    推理和后处理
  preprocess/   JD 清洗、规则抽取、去重
  resume/       简历文件解析、OCR sidecar、规范化
  train/        训练入口逻辑

scripts/
  data/         抓取、导入、审计、建池、一键数据流水线
  train/        14B SFT / DPO / smoke train
  serve/        API、vLLM、前端服务
  dev/          环境和模型下载
  research/     站点接口研究辅助
  legacy/       历史 1.7B 和早期实验脚本归档

configs/        数据源、标签、训练和多任务注册配置
docs/           项目文档、决策记录和数据说明
frontend/       Vue 前端演示
```

同时补了 `.gitignore`，把模型、数据、checkpoint、cache、日志等大文件从 Git 中剥离，避免提交污染。

---

## 17. 第十六阶段：训练前 readiness 门控

用户多次强调：SFT 的核心是数据，在数据没准备好之前不要训练。

因此项目增加了训练前 readiness 报告。

相关文件：

- `src/jobmatch_tune/eval/report_data_readiness.py`
- `scripts/data/report_data_readiness.sh`

它会检查：

- 各任务数据量是否达标。
- JSON 是否合法。
- ID 是否重复。
- train / valid / test 是否有跨 split 内容泄漏。
- JD 字段空值率是否可接受。
- JD high-risk 样本是否为 0。
- match 和 multitask 数据量是否达标。

当前最近一次报告：

- `all_ready_for_training = true`
- `data/sft_jd_quality/` 达到 5500 条。
- `data/sft_resume/` 达到当前规模要求。
- `data/sft_match/` 达到当前规模要求。
- `data/sft_multitask/` 达到默认 14B 小规模多任务 SFT 要求。

但是 readiness 通过不等于马上大规模训练。当前更合理的下一步是：

1. 抽样复核 JD quality 里的 `quality_weak` 层。
2. 检查 resume 的模板化增强比例。
3. 为 match 增加真实人工评估样本。
4. 再做 Qwen3-14B 小规模增量 SFT。

---

## 18. 当前项目做到什么程度

截至 2026-06-01，项目已经具备：

1. 多源中文 JD 抓取和导入能力。
2. JD raw / clean / dedup / candidate / quality SFT 分层。
3. JD 字段规则抽取和质量风险门控。
4. 简历文件解析、OCR sidecar 和 resume SFT 数据构造。
5. match 匹配任务数据构造。
6. 多任务 SFT 数据注册和采样。
7. Qwen3-14B QLoRA 训练脚本。
8. DPO 偏好优化脚本。
9. FastAPI 服务。
10. Vue 前端演示。
11. readiness 训练前质量门控。
12. 项目结构、文档和 Git 忽略规则整理。

从工程完整性看，它已经不是只会训练的项目，而是一套可以继续演进的招聘智能体数据与微调系统。

---

## 19. 当前最重要的风险

项目下一步仍然应该围绕数据，而不是马上盲训。

主要风险是：

1. JD quality 里仍有 `quality_weak` 层，需要继续抽样复核。
2. resume 数据数量大，但模板化增强占比高，需要补真实脱敏样本和公开简历样本。
3. match 数据以合成为主，需要人工标注集校准。
4. 多任务比例需要根据验证集表现调整。
5. 14B 训练成本比小模型高，训练前必须保证数据值得训练。

---

## 20. 下一步建议

推荐按这个顺序继续：

1. 做 JD quality 抽样审计，优先看 `quality_weak`。
2. 增加 resume SFT 画像审计，明确真实样本、模板增强样本和来源分布。
3. 扩充真实或半真实 resume 样本，减少模板化占比。
4. 补 200-500 条人工 match gold，用于校准匹配分数和解释质量。
5. 用当前 `data/sft_multitask/` 做一轮 Qwen3-14B 小规模增量 SFT。
6. 用人工评估集比较 base、SFT、规则后处理和 API 输出。
7. 如果 SFT 后输出可控，再考虑 DPO。

核心原则保持不变：

```text
先数据质量，再训练规模；先字段边界，再训练轮数；先小规模验证，再扩大训练。
```

