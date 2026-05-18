# Resume Pipeline

## 1. 问题定义

简历处理不能按 JD 处理方式直接复制。

JD 大多是网页文本，格式相对稳定；简历的真实输入更杂：

- `txt`
- `docx`
- 文本型 `pdf`
- 扫描版 `pdf`
- `png/jpg` 图片简历
- 双栏排版
- 表格模板
- 中英混排

所以简历链路要拆成两层问题：

1. 文档解析与文本恢复
2. 结构化抽取与匹配分析

当前项目里，`Qwen3-14B + LoRA` 负责的是第 2 层，不是直接做端到端视觉解析。

---

## 2. 推荐链路

### 2.1 原始输入层

输入支持四类：

1. 可直接抽文本
   - `txt`
   - `docx`
   - 文本型 `pdf`

2. 需要 OCR
   - 图片简历
   - 扫描件 `pdf`

3. 排版复杂但仍可解析
   - 双栏 PDF
   - 表格型简历

4. 暂不处理为结构化训练输入
   - 视觉设计型简历
   - 信息图式简历

---

## 3. 当前实现

### 3.1 `resume_ingest`

新增入口：

- [src/jobmatch_tune/resume/ingest.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/resume/ingest.py)
- [scripts/data/resume_ingest.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/resume_ingest.sh)

当前已支持：

- `txt`
- `docx`
- `pdf`

当前行为：

- `txt`：直接读取
- `docx`：使用 `python-docx` 提取段落
- `pdf`：使用 `pypdf` 提取文本页，并区分：
  - `text_pdf`
  - `weak_text_pdf`
  - `scanned_pdf`
- `png/jpg/jpeg/webp/bmp`：优先尝试 sidecar OCR 文本；没有 sidecar 时标记为 `needs_ocr=true`
- `pdf`：如果文本抽取为空，也会尝试 sidecar OCR 文本

这一步的目标不是直接输出最终 JSON，而是统一生成中间文本。

命令：

```bash
bash scripts/data/resume_ingest.sh <resume-file-or-dir>
```

可选 sidecar OCR 目录：

```bash
bash scripts/data/resume_ingest.sh <resume-file-or-dir> data/resume_raw/resume_ingest.jsonl <ocr-dir>
```

当前 sidecar 约定：

- `<原文件名>.ocr.txt`
- `<文件 stem>.ocr.txt`

自动生成 sidecar 的入口：

- [src/jobmatch_tune/resume/ocr.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/resume/ocr.py)
- [scripts/data/resume_ocr_sidecar.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/resume_ocr_sidecar.sh)

命令：

```bash
bash scripts/data/resume_ocr_sidecar.sh <image-or-pdf-file-or-dir>
```

默认输出目录：

```text
data/resume_ocr_text/
```

默认输出：

```text
data/resume_raw/resume_ingest.jsonl
```

### 3.2 `resume_normalize`

新增入口：

- [src/jobmatch_tune/resume/normalize.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/resume/normalize.py)
- [scripts/data/resume_normalize.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/resume_normalize.sh)

作用：

- 读取 `resume_ingest` 结果
- 保留文档来源元信息
- 统一生成 `normalized_text`
- 作为 `resume_parse` 的中间输入层

命令：

```bash
bash scripts/data/resume_normalize.sh \
  --input data/resume_raw/resume_ingest.jsonl \
  --out data/resume_interim/resume_clean.jsonl \
  --only-parse-ok
```

---

## 4. 中间表示

`resume_ingest` 输出的每条记录包含：

- `id`
- `file_name`
- `file_path`
- `source_type`
- `extraction_method`
- `page_count`
- `text_char_count`
- `pdf_kind`
- `ocr_used`
- `ocr_source`
- `needs_ocr`
- `parse_ok`
- `raw_text`
- `clean_text`
- `sections`

其中 `sections` 当前按标题做粗分块：

- `header`
- `education`
- `skills`
- `internships`
- `projects`
- `work`
- `awards`
- `profile`

这一步的价值是把后续 `resume_parse` 从原始文档格式中解耦出来。

`resume_normalize` 则把这些字段进一步整理成统一的 `normalized_text`，让：

- `text`
- `docx`
- `text_pdf`
- `ocr sidecar`

最终都在一层收敛，再交给 `resume_parse`。

---

## 5. 为什么需要中间层

如果直接把 PDF 或图片解析结果喂给 `resume_parse`，会遇到三个问题：

1. 断行和换列混乱
2. 标题顺序不稳定
3. OCR 错字和装饰字符会影响字段抽取

因此正确路线是：

1. `resume_raw`
2. `resume_interim`
3. `resume_structured`

当前已经落地第 1 层和第 2 层的一部分：

- 原始文件 -> 文本抽取
- 文本规范化
- 简单标题分块

---

## 6. 当前 resume schema

当前项目 `resume_parse` 默认抽取：

- `目标岗位`
- `教育背景`
- `核心技能`
- `实习经历`
- `项目经历`
- `优势标签`

如果后续要增强匹配效果，建议扩到：

- `工作经历`
- `院校背景`
- `奖项证书`

但这一步应在文档解析和基础简历数据链路稳定后再扩，不要先扩 schema 再补数据。

---

## 7. 图片和扫描件怎么处理

当前项目已经支持 OCR sidecar 生成链路，但运行时依赖需要单独安装。

对图片简历和扫描版 PDF，目前策略是：

- 如果存在 OCR sidecar 文本，直接走文本规范化和分块
- `scanned_pdf`：没有 sidecar 时标记 `needs_ocr=true`
- `weak_text_pdf`：没有 sidecar 时标记 `needs_ocr=true`，因为通常是扫描件混少量可抽文字，直接拿去做结构化风险很高

后续推荐路线：

1. OCR backend
   - 优先 `rapidocr_onnxruntime`
   - 也兼容 `paddleocr`
   - PDF OCR 额外依赖 `PyMuPDF`
2. 补版面恢复
   - 双栏和表格简历需要恢复阅读顺序
3. 再进入 `resume_parse`

也就是说：

- 当前文本微调模型负责文本结构化
- OCR / layout 是前置文档解析层

---

## 8. 当前数据与训练现状

目前 `resume_parse` 已经有：

- 人工评估种子集  
  [data/eval/resume_manual_eval_seed.jsonl](/share/home/lifr/workspace/code/job-match-tune/data/eval/resume_manual_eval_seed.jsonl)

- 分层评估集  
  [data/eval/resume_manual_eval_text_seed.jsonl](/share/home/lifr/workspace/code/job-match-tune/data/eval/resume_manual_eval_text_seed.jsonl)  
  [data/eval/resume_manual_eval_ocr_seed.jsonl](/share/home/lifr/workspace/code/job-match-tune/data/eval/resume_manual_eval_ocr_seed.jsonl)

- bootstrap SFT 数据  
  [data/sft_resume/train.jsonl](/share/home/lifr/workspace/code/job-match-tune/data/sft_resume/train.jsonl)  
  [data/sft_resume/valid.jsonl](/share/home/lifr/workspace/code/job-match-tune/data/sft_resume/valid.jsonl)  
  [data/sft_resume/test.jsonl](/share/home/lifr/workspace/code/job-match-tune/data/sft_resume/test.jsonl)

但这批 SFT 数据仍然是基于人工种子扩写得到的高质量小集，作用是打通训练链路，不是最终规模数据。

---

## 9. 下一步

最合理的顺序：

1. 继续完善 `resume_ingest`
   - 增加 `docx/pdf` 更多边界处理

2. 补 OCR 分支
   - 让图片简历不再只停留在 `needs_ocr`

3. 扩 resume gold eval
   - 区分：
     - 文本简历
     - PDF 简历
     - OCR 简历

4. 扩 resume SFT 数据
   - 不再只靠少量人工种子扩写

5. 再推进 `match` 的 gold eval
   - 让人岗匹配不只靠规则 baseline + LLM 解释

---

## 10. 当前可执行评估入口

文本简历 pipeline 评估：

```bash
bash scripts/data/run_resume_pipeline_eval.sh \
  --dataset data/eval/resume_manual_eval_text_seed.jsonl \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --out outputs/eval_reports/resume_pipeline_text_report.json \
  --predictions-out outputs/eval_reports/resume_pipeline_text_predictions.jsonl \
  --load-4bit
```

OCR-like 简历 pipeline 评估：

```bash
bash scripts/data/run_resume_pipeline_eval.sh \
  --dataset data/eval/resume_manual_eval_ocr_seed.jsonl \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --out outputs/eval_reports/resume_pipeline_ocr_report.json \
  --predictions-out outputs/eval_reports/resume_pipeline_ocr_predictions.jsonl \
  --load-4bit
```

这两个入口的意义是把问题拆开：

- `text_seed` 更接近纯文本简历解析质量
- `ocr_seed` 更接近 OCR 噪声下的鲁棒性
