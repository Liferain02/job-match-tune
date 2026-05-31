# 项目完整来龙去脉

这份文档按时间线整理项目从 0 到当前版本的完整过程：为什么做、每一步做了什么、怎么做、做完以后留下了什么产物，以及为什么某些方向后来被调整或放弃。

项目当前不是一个单纯的“训练脚本集合”，而是一套围绕招聘场景的数据、训练、推理和评估闭环。核心任务从最初的 JD 解析，逐步扩展到：

1. JD 结构化解析
2. 简历结构化解析
3. JD 与简历匹配分析
4. API 服务
5. 前端演示
6. 训练前数据 readiness 审计

---

## 1. 最初目标：做一个招聘 JD 解析微调项目

项目最早的目标很明确：基于中文招聘 JD，微调一个模型，把非结构化岗位文本解析成固定 JSON。

目标字段是：

- `岗位方向`
- `核心职责`
- `必备技能`
- `加分项`
- `经验要求`
- `学历要求`

这个任务适合 SFT，原因是输出结构固定，错误类型也比较稳定，主要集中在：

- 岗位方向分类不准
- 职责和要求混在一起
- 技能字段漏抽或误抽
- 学历和经验边界不清
- 输出 JSON 偶尔不合法

因此项目一开始就没有把重点放在“写一个很花的 prompt”，而是围绕数据和评估搭建可迭代链路。

---

## 2. 技术方案选择：先低成本跑通，再升级到 14B

最初考虑过 Qwen2.5 系列，但后面结合模型时间线和项目简历价值，调整为 Qwen3 系列。

当前默认路线是：

- 基座模型：`Qwen3-14B`
- 微调方式：`QLoRA`
- 训练框架：`Transformers + PEFT + TRL + Accelerate`
- 推理服务：
  - 本地 `Transformers + PEFT`
  - 服务化 `vLLM + OpenAI-compatible API`
- API：`FastAPI`
- 前端：`Vue 3 ESM`

为什么选 QLoRA：

- 当前节点 GPU 资源是 `3 x NVIDIA L20`
- 单卡显存约 46GB
- 14B 全参微调成本不合适
- 4-bit QLoRA 可以把 14B 微调控制在可接受显存内
- LoRA adapter 便于多轮增量训练和版本回滚

项目中也预留或接入了更现代的训练能力：

- assistant-only loss
- DFT loss
- Liger Kernel
- packing
- gradient checkpointing
- DPO preference tuning

这些不是为了堆名词，而是服务于一个目标：在 GPU 不多的情况下，把 14B 模型稳定训起来，并能持续迭代。

---

## 3. 环境搭建：固定 conda 环境和依赖

项目环境最终统一为：

```bash
conda create -n tune-demo python=3.11 -y
conda activate tune-demo
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

GPU 检查使用：

```bash
nvidia-smi
```

后面所有数据脚本、测试和训练脚本都默认在 `tune-demo` 环境中执行。

依赖管理上，项目从直接安装包逐步调整为维护 `requirements.txt`，这样后续提交代码、换机器或写简历时都更清晰。

---

## 4. 第一条数据链路：腾讯招聘 JD

项目最开始没有盲目爬很多网站，而是先选一个相对稳定的数据源，把 `raw -> clean -> sft` 链路跑通。

第一条稳定来源是腾讯招聘。

原因：

- 技术岗位多
- JD 质量相对高
- 公开接口比较稳定
- 职责和要求结构较清楚

实现文件：

- [tencent_careers.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/tencent_careers.py)
- [tencent_keywords.txt](/share/home/lifr/workspace/code/job-match-tune/configs/tencent_keywords.txt)

做法：

1. 按关键词请求腾讯招聘公开接口。
2. 分页抓取岗位列表。
3. 进入详情补全完整 JD。
4. 写入 `data/raw/tencent_jd_raw.jsonl`。
5. 同步写入 SQLite，方便后续统一处理。

这个阶段的关键成果不是数据量，而是验证了项目的数据骨架：

```text
公开招聘源 -> raw JSONL / SQLite -> 清洗 -> 去重 -> 字段规则 -> SFT JSONL
```

---

## 5. 第二阶段：扩展中文官网数据源

腾讯跑通后，继续扩展高质量中文官网源。

### 5.1 百度招聘

百度招聘不是简单 JSON API，而是从页面中的 SSR 数据里解析职位信息。

实现文件：

- [baidu_talent.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/baidu_talent.py)
- [baidu_keywords.txt](/share/home/lifr/workspace/code/job-match-tune/configs/baidu_keywords.txt)

做法：

1. 通过关键词访问搜索页。
2. 解析页面里的 `window.__INITIAL_DATA__`。
3. 抽出职位列表和详情。
4. 统一映射到项目的 raw schema。

### 5.2 京东招聘

京东招聘后来成为重要数据源，因为匿名职位接口比较稳定。

实现文件：

- [jd_careers.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/jd_careers.py)

做法：

1. 请求京东招聘匿名 API。
2. 分页获取职位。
3. 请求详情接口。
4. 写入 `data/raw/jd_careers_raw.jsonl` 和 SQLite。

### 5.3 Moka 招聘官网

Moka 是后面扩充中文官网数据最有效的一条线。

原因：

- 很多企业招聘官网都托管在 Moka。
- API 形态相对一致。
- 只要维护公司源配置，就能复用同一个抓取器。

实现文件：

- [moka_careers.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/moka_careers.py)
- [moka_sources.yaml](/share/home/lifr/workspace/code/job-match-tune/configs/moka_sources.yaml)

做法：

1. 在 `moka_sources.yaml` 维护公司源。
2. 对每个源拉职位列表。
3. 拉职位详情。
4. 统一写入 raw JSONL 和 SQLite。

这条线接入过 AI、芯片、自动驾驶、云基础设施、金融科技、游戏、安全等公司源。

### 5.4 其他官网源

后续也接入或研究过：

- 携程
- 小米
- 美团
- 滴滴
- 拼多多、得物、小红书等站点探测
- 字节、蚂蚁、B 站等 API 研究

其中有些源能稳定落地，有些只是探测。项目最终的原则是：只把匿名稳定、可复现、质量足够的数据源纳入主数据链路。

---

## 6. 为什么没有硬攻所有招聘平台

项目过程中也评估过 Boss 直聘、智联招聘和一些大型公司官网。

最终没有把所有平台都放进主线，原因包括：

- 需要登录或强风控
- 接口签名不稳定
- 前端网关校验重
- 抓取成本高但结构质量未必更好
- 合规风险和可复现性差

所以项目后面转向更务实的路线：

- 优先中文企业官网
- 优先 Moka 这类托管招聘系统
- 优先能匿名访问的公开接口
- 优先结构化程度高的 JD
- 对公开数据集只做候选池，不直接混入严格主集

---

## 7. 第三阶段：接入公开数据集，把原始池做大

官网数据质量高，但数量增长较慢。为了扩大原始语料池，项目接入了公开数据集和导出文件。

实现文件：

- [import_public_job_data.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/import_public_job_data.py)
- [public_job_sources.yaml](/share/home/lifr/workspace/code/job-match-tune/configs/public_job_sources.yaml)
- [public_job_sources_zh_large.yaml](/share/home/lifr/workspace/code/job-match-tune/configs/public_job_sources_zh_large.yaml)

接入过的数据包括：

- GitHub `jhcoco/bosszp`
- GitHub `WorkAggregation`
- Hugging Face `open-apply-jobs`
- Hugging Face 中文招聘学历数据

这一步的原则很重要：

公开数据集只先进入 raw pool 或 candidate pool，不等于直接进入高质量训练集。

原因是公开数据集里有很多噪声：

- 英文 JD 不适合中国企业主场景
- 有教育任务数据，不一定是真实招聘主线
- 有实习、校招、教师、培训师、销售、运营等非目标岗位
- 有些字段来自弱标注，不够稳定

这也是后来反复强调的结论：几十万原始数据不等于几十万可训练数据。

---

## 8. 清洗、去重和统一 schema

所有来源最终都会进入统一处理链路。

核心模块：

- [clean_text.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/clean_text.py)
- [normalize_jd.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/normalize_jd.py)
- [jd_field_rules.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/jd_field_rules.py)
- [deduplicate.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/deduplicate.py)

处理步骤：

1. 清理 HTML、空白、符号和重复格式。
2. 按标题、公司、地点、正文统一 schema。
3. 切分 `岗位职责 / 任职要求 / 加分项`。
4. 用规则抽取：
   - 学历
   - 经验
   - 技能
   - 岗位方向
5. 做去重，生成 `jd_clean_dedup.jsonl`。

当前数据规模：

- `data/interim/jd_clean.jsonl`: `293582`
- `data/interim/jd_clean_dedup.jsonl`: `269351`
- `data/eval/jd_train_pool_combined.jsonl`: `37796`

---

## 9. 岗位方向 schema 的持续迭代

项目早期的岗位方向比较窄，主要是前端、后端、测试、算法等常见方向。

后来根据真实 JD 和用户要求，扩展了更多技术方向：

- 网络与基础设施
- 高性能计算
- AI Infra
- 前端开发
- 运维开发
- 硬件研发
- 汽车软件 / 智驾研发
- 嵌入式开发
- 安全工程
- AI 应用开发

相关文档：

- [job_direction_policy.md](/share/home/lifr/workspace/code/job-match-tune/docs/job_direction_policy.md)

这一步解决的是模型训练前的标注口径问题。方向口径不稳定时，继续加训练轮数没有意义，所以项目先冻结关键边界，再扩 hard case。

做过的事情：

1. 做人工 gold 样本。
2. 扩前端、后端、测试、算法、AI 应用等边界样本。
3. 补方向 hard case。
4. 修规则。
5. 再决定是否重训。

---

## 10. JD 数据分层：为什么不是所有数据都能训练

项目后来把 JD 数据分成多层。

### 10.1 strict

路径：

- `data/sft/`

含义：

- 高信任中文官网源
- 技术岗标题明确
- 职责或要求结构完整
- 字段质量较高

当前规模：

- `train / valid / test = 2666 / 333 / 334`

### 10.2 strict_plus

路径：

- `data/sft_jd_strict_plus/`

含义：

- 介于 strict 和 bootstrap 之间
- 从 combined pool 中回收结构较完整的样本
- 会过滤低信号标题和文本

### 10.3 quality

路径：

- `data/sft_jd_quality/`

这是当前 JD 训练候选主线。

构建方式：

1. 先取 strict。
2. 再补 strict_plus。
3. 最后补 quality_weak。
4. 当前没有使用 bootstrap 兜底。

当前规模：

- `train / valid / test = 4000 / 500 / 500`
- 总计 `5000`

当前分层来源：

- `strict = 3331`
- `strict_plus = 275`
- `quality_weak = 1394`
- `bootstrap = 0`

### 10.4 bootstrap

路径：

- `data/sft_jd_bootstrap/`

含义：

- 更偏实验性的弱标注数据线
- 可以用于对比或扩展实验
- 不直接替代当前主训练线

---

## 11. 数据质量问题如何被发现和修复

项目过程中反复出现一个问题：数据数量看起来很大，但严格筛选后训练集变小。

原因不是数据丢了，而是质量门槛变严了。

典型问题包括：

- 英文 JD 不适合当前中文企业任务
- 教师、培训师、编剧、编导等非目标岗位混入
- 实习、校招、管培生样本任务口径不同
- “本科 / 硕士”误入经验字段
- 技能字段为空
- JD 正文结构不完整
- 岗位方向被上下文误导

最近一次训练前审计中，抽样发现了 `教师 / 编导 / 新闻编辑` 这类混合招聘被误标为算法工程。随后做了修复：

- strict 层增加非技术标题排除词。
- strict_plus 增加低信号标题过滤。
- quality_weak 增加非技术标题过滤。
- JD 构建时对 `必备技能` 做原文二次抽取。

修复后重新构建 `data/sft_jd_quality/`，仍保持 5000 条，并且标题黑名单命中数降为 0。

---

## 12. 简历解析任务是怎么加入的

项目从单一 JD 解析扩展到简历解析，是因为最终产品目标不只是解析岗位，而是做：

```text
JD 解析 -> 简历解析 -> JD 和简历匹配度分析
```

简历解析目标字段：

- `目标岗位`
- `教育背景`
- `核心技能`
- `实习经历`
- `项目经历`
- `优势标签`

相关模块：

- [resume/ingest.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/resume/ingest.py)
- [resume/normalize.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/resume/normalize.py)
- [resume/ocr.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/resume/ocr.py)
- [build_resume_sft_dataset.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/dataset/build_resume_sft_dataset.py)

简历格式处理思路：

1. 文本简历直接抽取文本。
2. PDF 简历走 PDF 文本抽取。
3. 图片型 PDF 或图片简历标记为 `needs_ocr`。
4. OCR 结果通过 sidecar 文本进入同一 normalize 层。
5. 最后统一生成结构化 SFT 样本。

为了模拟不同简历格式，项目构造了多种渲染变体：

- 原始文本
- bullet list
- compact
- OCR-like
- education-first
- project-first
- table-like
- dense resume
- mixed CN/EN

当前简历 SFT 数据：

- `data/sft_resume/train.jsonl`: `38408`
- `data/sft_resume/valid.jsonl`: `4850`
- `data/sft_resume/test.jsonl`: `4890`

---

## 13. JD 与简历匹配任务是怎么加入的

匹配任务是第三条主线。

输入：

- JD 文本
- 简历文本
- 规则评分结果

输出：

- `匹配结论`
- `匹配优势`
- `主要短板`
- `简历优化建议`
- `推荐投递岗位方向`

相关模块：

- [build_match_sft_dataset.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/dataset/build_match_sft_dataset.py)
- [match_rule_engine.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/match_rule_engine.py)

做法：

1. 用 JD 结构化结果和简历结构化结果做规则匹配。
2. 计算方向、学历、经验、技能命中和缺失。
3. 将规则结果作为输入上下文。
4. 让模型生成可读的匹配分析。

最近一次数据质量修复中，发现部分高匹配样本的 `主要短板` 或 `推荐投递岗位方向` 为空。随后补了兜底说明，保证 match 输出字段完整。

当前 match SFT 数据：

- `data/sft_match/train.jsonl`: `1799`
- `data/sft_match/valid.jsonl`: `228`
- `data/sft_match/test.jsonl`: `229`

---

## 14. 训练链路如何演进

项目训练路线经历了几个阶段。

### 14.1 先用小模型跑通

最初使用较小的 Qwen3 模型验证：

- 数据格式是否正确
- LoRA 训练脚本是否可跑
- 推理是否能输出 JSON
- 后处理是否能兜住格式问题

### 14.2 升级 14B

后面确认当前节点有 3 张 L20，用户明确要求必须上 14B，于是转向：

- `models/Qwen3-14B`
- `4-bit QLoRA`
- 多 GPU 资源充分利用

模型由用户手动下载：

```bash
modelscope download --model Qwen/Qwen3-14B --local_dir ./Qwen3-14B
```

放在：

- `models/Qwen3-14B`

### 14.3 不盲训，先修数据

项目多次明确一个原则：

数据质量和数量是 SFT 之前最重要的事情。

因此在数据没有准备充分之前，不继续盲目训练。尤其是当字段混淆明显时，优先做：

- 人工评估集
- 方向 hard case
- 后处理规则
- 数据分层
- readiness 审计

---

## 15. 人工评估和后处理

项目做过人工 gold 和字段级评估。

评估重点：

- 岗位方向是否正确
- 核心职责是否混入要求
- 必备技能是否漏抽或误抽
- 学历是否提取正确
- 经验是否提取正确

相关模块：

- [run_manual_eval.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/eval/run_manual_eval.py)
- [metrics.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/eval/metrics.py)
- [postprocess_json.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/inference/postprocess_json.py)

后处理做的事情：

- 修复不合法 JSON
- 规范字段名
- 清理职责和要求混杂
- 过滤明显误报技能
- 补齐学历和经验
- 规范岗位方向

这一层非常关键，因为项目目标是稳定输出结构化 JSON，而不是只看生成文本是否像答案。

---

## 16. API 和前端

项目后来补了应用层。

API：

- [server.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/api/server.py)

主要接口：

- `GET /health`
- `GET /api/status`
- `POST /api/warmup`
- `POST /api/parse`
- `POST /api/resume/parse-file`
- `POST /api/match`

前端：

- [frontend/](/share/home/lifr/workspace/code/job-match-tune/frontend)

前端从早期静态 HTML / JS / CSS 逐步改成 Vue 3 ESM 结构，用于演示：

- JD 输入和解析
- 简历文本或文件上传
- JD 与简历匹配分析
- API 状态展示

---

## 17. 项目结构优化

项目后期做过结构整理，把脚本按用途分类：

- `scripts/data/`：抓取、导入、清洗、建池、审计
- `scripts/train/`：训练
- `scripts/serve/`：API、vLLM、前端
- `scripts/dev/`：环境和模型下载
- `scripts/research/`：站点研究和探测
- `scripts/legacy/`：历史脚本归档

同时清理过不必要的旧代码、14B 之前的冗余脚本、被 `.gitignore` 忽略但仍被 git 跟踪的大文件。

---

## 18. 当前训练前 readiness 状态

最近一次训练前数据审计已经通过。

命令：

```bash
bash scripts/data/report_data_readiness.sh
```

当前结论：

```json
{
  "all_ready_for_training": true,
  "not_ready_tasks": []
}
```

三条训练线：

| 任务 | train | valid | test | combined pool |
| --- | ---: | ---: | ---: | ---: |
| JD | 4000 | 500 | 500 | 37796 |
| 简历 | 38408 | 4850 | 4890 | 3137 |
| 匹配 | 1799 | 228 | 229 | 2256 |
| 多任务 SFT | 8000 | 1000 | 0 | 9000 |

readiness 当前检查：

- 数量是否达到门槛
- assistant 输出是否为合法 JSON
- ID 是否重复
- `train / valid / test` 之间是否存在内容级重复
- 关键字段空值率是否超过阈值

JD 当前字段空值率：

- `岗位方向`: `0.0`
- `核心职责`: `0.0456`
- `必备技能`: `0.263`
- `学历要求`: `0.2902`
- `经验要求`: `0.4934`

简历当前字段空值率均为 `0.0`。

匹配当前字段空值率均为 `0.0`。

多任务 SFT 当前不直接全量混合三条数据线，而是通过 `configs/dataset_registry.yaml` 做采样：

- `JD`: `4000 / 500`
- `resume`: `2400 / 300`
- `match`: `1600 / 200`

这样可以避免 `resume` 扩到 48148 条后在训练中压过 JD 和 match。

---

## 19. 当前项目能做什么

当前项目已经具备：

1. 从多个中文招聘官网和公开数据集构建 JD 原始池。
2. 对 JD 做清洗、去重、规则抽取和分层筛选。
3. 构建 5000 条 JD 质量 SFT 数据。
4. 构建 48148 条简历解析 SFT 数据。
5. 构建 2256 条匹配分析 SFT 数据。
6. 对三条数据线做 readiness 审计。
7. 在 14B QLoRA 路线上继续训练。
8. 通过 API 和前端提供可演示能力。

---

## 20. 下一步最合理的工作

当前数据已经达到工程训练门槛，下一步不是继续盲目爬数据，而是：

1. 对 `data/sft_jd_quality/` 做一轮人工抽样审计，尤其是 `quality_weak` 层。
2. 如果抽样通过，启动一轮小规模 14B 增量 SFT。
3. 训练后跑 50 条人工 holdout。
4. 对比训练前后字段指标。
5. 如果 JD 指标提升，再考虑把 resume 和 match 混合进多任务训练。
6. 如果 JD 某些字段下降，优先回到数据和后处理修正。

这个项目到目前为止的核心经验是：

> SFT 的关键不是尽快开训，而是先把任务口径、数据分层、字段质量和评估闭环做稳。数据不稳时，训练只会放大噪声；数据稳定后，小规模增量训练才有意义。
