# 三条数据链路说明

更新时间：2026-05-18

这份文档专门解释项目里常说的三条数据链路：

- `JD`
- `resume`
- `match`

也解释这些步骤分别是什么意思：

- 导入
- 审计
- 候选池
- 合并建池
- 一键流水线

目标不是讲模型，而是回答三个问题：

1. 数据到底从哪里来
2. 数据进仓库之后怎么处理
3. 什么时候这些数据才算“能训练”

---

## 1. 总体原则

项目的数据处理原则一直是：

1. **先拿到数据**
2. **先做统一格式**
3. **先做质量审计**
4. **先做候选池 / 合并建池**
5. **最后才考虑训练**

所以这里的“数据池”不是一上来就等于训练集。

项目里现在明确区分四层：

1. `raw / external`
   - 原始抓取数据
   - 原始公开数据导入
2. `audit`
   - 数据统计和质量报告
3. `candidate / combined pool`
   - 候选池
   - 合并后的可训练候选池
4. `sft / eval`
   - 真正训练或评估在用的数据

---

## 2. JD 这条线

### 2.1 数据从哪里来

JD 当前有两类来源。

#### A. 官网抓取

当前主源：

- 腾讯公开招聘
- 百度招聘
- 京东招聘
- Moka 托管招聘官网

这些通过 `src/jobmatch_tune/crawler/` 下的抓取器进入：

- `tencent_careers.py`
- `baidu_talent.py`
- `jd_careers.py`
- `moka_careers.py`

抓完后写入：

- `data/raw/*.jsonl`
- `data/jobmatch_tune.sqlite3`

#### B. 公开数据集导入

当前接入过的公开 JD 源：

- `open-apply-jobs`
- `job-educational-parser-dataset-08-0-0805`
- GitHub 上的职位导出 CSV

对应脚本：

- `scripts/data/import_public_job_exports.sh`
- `scripts/data/import_chinese_job_exports.sh`

这类数据主要作用是：

- 扩原始语料池
- 扩候选池
- 不直接等于默认高质量训练集

---

### 2.2 “审计”是什么意思

JD 的“审计”就是：  
先不训练，先统计这批公开 JD 到底长什么样。

当前入口：

- `scripts/data/audit_public_jd_data.sh`
- `src/jobmatch_tune/eval/audit_public_jd_data.py`

它会看：

- 总量
- 来源分布
- 语言分布
- 平均标题长度
- 平均正文长度
- 学历覆盖率
- 经验覆盖率
- 薪资覆盖率
- `sft_ready` 分布

审计的作用是：

1. 判断这批数据是不是中文为主
2. 判断字段是不是太浅
3. 判断是不是值得进入下一步候选池

也就是说，**审计不是训练前锦上添花，而是训练前筛风险。**

---

### 2.3 “候选池”是什么意思

JD 的候选池指：

> 从公开导入的大池子里，先筛出一批“看起来像技术岗、文本也够完整”的职位。

当前入口：

- `scripts/data/build_public_jd_candidate_pool.sh`
- `src/jobmatch_tune/dataset/build_public_jd_candidate_pool.py`

它做的事是：

1. 只保留中文
2. 职位标题必须像技术岗
3. 过滤明显业务/非技术岗
4. 正文长度要过阈值
5. 至少具备薪资/学历/经验/职责中的若干信号
6. 去重

输出：

- `data/eval/public_jd_candidate_pool.jsonl`

这个文件的定位不是“默认训练集”，而是：

- **公共 JD 里的高质量候选池**

---

### 2.4 “合并建池”是什么意思

JD 的“合并建池”是把两批东西合在一起：

1. 当前默认严格高质量 JD
2. 公共导入里筛出来的候选 JD

当前入口：

- `scripts/data/build_jd_train_pool_combined.sh`
- `src/jobmatch_tune/dataset/build_jd_train_pool_combined.py`

它会：

1. 从 `data/interim/jd_clean_dedup.jsonl` 里回收 `strict` 高质量 JD
2. 读 `data/eval/public_jd_candidate_pool.jsonl`
3. 合并
4. 再去重
5. 输出：
   - `data/eval/jd_train_pool_combined.jsonl`

注意：

这个合并池仍然不是默认 `data/sft/`。  
它的定位是：

- **下一步扩默认 JD 训练集的储备池**

---

### 2.5 “一键流水线”是什么意思

JD 当前有一条一键流水线：

- `scripts/data/prepare_public_jd_pipeline.sh`

它会顺序跑：

1. `audit_public_jd_data.sh`
2. `build_public_jd_candidate_pool.sh`
3. `build_jd_train_pool_combined.sh`

也就是把：

- 审计
- 候选池
- 合并建池

一次跑完。

---

## 3. resume 这条线

### 3.1 数据从哪里来

resume 目前有三类来源。

#### A. 人工 / 模板高质量样本

这是当前主路。

文件包括：

- `data/eval/resume_manual_eval_seed.jsonl`
- `data/eval/resume_manual_eval_augmented.jsonl`
- `data/eval/resume_manual_train_pool.jsonl`

这批数据的特点：

- 可控
- 脱敏
- 结构清楚
- 适合当前 resume schema

#### B. 原始简历文件接入

当前支持：

- `txt`
- `docx`
- `pdf`
- `图片 + OCR sidecar`

对应链路：

- `resume_ingest`
- `resume_normalize`
- `resume_ocr_sidecar`

这条线解决的是：

- 简历原始文件怎么变成结构化文本

#### C. 公开 resume 数据集导入

当前已经补了入口，但还没大规模落盘：

- `FairCV`
- `resume-ner`

对应入口：

- `scripts/data/import_public_resume_exports.sh`

---

### 3.2 “导入”是什么意思

resume 的“导入”不是 OCR，也不是 parse。  
这里专指：

> 把外部公开简历数据文件映射到仓库内部统一格式。

当前入口：

- `scripts/data/import_public_resume_exports.sh`
- `src/jobmatch_tune/dataset/import_public_resume_data.py`

支持：

- `json`
- `jsonl`
- `csv`
- `parquet`

导入后统一输出：

- `data/external/public_resume_imports.jsonl`

格式里目前允许两种任务：

1. `resume_parse`
2. `resume_ner`

---

### 3.3 “审计”是什么意思

resume 的“审计”就是先看外部公开简历数据到底能不能用。

当前入口：

- `scripts/data/audit_public_resume_data.sh`
- `src/jobmatch_tune/eval/audit_public_resume_data.py`

它会统计：

- 总量
- `task` 分布
- 来源分布
- 语言分布
- 文本长度
- `目标岗位 / 教育背景 / 核心技能 / 实习 / 项目` 覆盖率
- `resume_ner` tag 集合

输出报告一般会落到：

- `outputs/eval_reports/public_resume_import_audit.json`

---

### 3.4 “合并建池”是什么意思

resume 没有单独“候选池”这一步，而是直接做“合并建池”。

原因很简单：

- resume 的主数据现在本来就主要靠人工高质量集
- 外部公开 resume 要先过 schema 和字段覆盖筛选

当前入口：

- `scripts/data/build_resume_train_pool_combined.sh`
- `src/jobmatch_tune/dataset/build_resume_train_pool_combined.py`

它会：

1. 读取人工 `resume_manual_train_pool`
2. 读取 `public_resume_imports`
3. 只接收：
   - `task=resume_parse`
   - 中文
   - 文本长度够
   - 字段信号足够
4. 合并并去重
5. 输出：
   - `data/eval/resume_train_pool_combined.jsonl`

这个文件的定位是：

- **resume 后续可训练池**

---

### 3.5 “一键流水线”是什么意思

resume 当前一键入口：

- `scripts/data/prepare_public_resume_pipeline.sh`

顺序执行：

1. `import_public_resume_exports.sh`
2. `audit_public_resume_data.sh`
3. `build_resume_train_pool_combined.sh`

也就是：

- 先导入
- 再审计
- 再合并建池

---

## 4. match 这条线

### 4.1 数据从哪里来

match 当前有两类来源。

#### A. 人工配对样本

当前主路是人工/模板构造的 JD-简历配对样本。

文件包括：

- `data/eval/match_manual_eval_seed.jsonl`
- `data/eval/match_manual_train_pool.jsonl`

这批数据的作用：

- 先把 `match` 的任务定义和规则评估链路跑通

#### B. 外部公开匹配数据

当前已补入口：

- `resume-job-fit-merged-v1`
- `resume-job-description-fit`

对应入口：

- `scripts/data/import_public_match_exports.sh`

这批数据当前主要是英文，用来做：

- 结构参考
- 弱监督候选池
- 对照实验

不直接等于中文默认高质量匹配训练集。

---

### 4.2 “导入”是什么意思

match 的导入指：

> 把外部公开 JD-resume pair 数据映射成统一 pair 格式。

当前入口：

- `scripts/data/import_public_match_exports.sh`
- `src/jobmatch_tune/dataset/import_public_match_data.py`

导入后统一输出：

- `data/external/public_match_imports.jsonl`

统一结构大致是：

- `jd_text`
- `resume_text`
- `label.raw_label`
- `label.raw_score`
- `meta`

---

### 4.3 “审计”是什么意思

match 的审计重点不是技能字段，而是看 pair 本身像不像可用样本。

当前入口：

- `scripts/data/audit_public_match_data.sh`
- `src/jobmatch_tune/eval/audit_public_match_data.py`

它会看：

- 总量
- 来源分布
- 语言分布
- 标签分布
- 平均 JD 长度
- 平均 resume 长度
- score 覆盖率

---

### 4.4 “合并建池”是什么意思

match 的“合并建池”是：

1. 保留人工高质量 `match_manual_train_pool`
2. 把可用的公开 pair 样本并进来
3. 去重

当前入口：

- `scripts/data/build_match_train_pool_combined.sh`
- `src/jobmatch_tune/dataset/build_match_train_pool_combined.py`

它会筛：

1. `task=match`
2. `jd_text` 和 `resume_text` 都要过长度阈值
3. 至少有 `raw_label` 或 `raw_score`
4. 当前允许：
   - `zh`
   - `zh-cn`
   - `en`

输出：

- `data/eval/match_train_pool_combined.jsonl`

这个文件的定位是：

- **match 的统一弱监督 / 可训练候选池**

---

### 4.5 “一键流水线”是什么意思

match 当前一键入口：

- `scripts/data/prepare_public_match_pipeline.sh`

顺序执行：

1. `import_public_match_exports.sh`
2. `audit_public_match_data.sh`
3. `build_match_train_pool_combined.sh`

---

## 5. 为什么现在还不能训练

虽然三条线的基础设施都齐了，但当前一个关键事实没有变：

> **很多 combined pool 文件还没真正跑出来，或者规模还没验证。**

例如当前仓库里：

- `data/sft/` 有
- `data/sft_resume/` 有
- `data/sft_match/` 有

但：

- `data/eval/jd_train_pool_combined.jsonl`
- `data/eval/resume_train_pool_combined.jsonl`
- `data/eval/match_train_pool_combined.jsonl`

这些合并池文件如果还没实际生成，说明：

- 基础设施是好的
- 但数据规模还没有被真正验证

这就是为什么当前项目一直坚持：

- **先补数据**
- **先跑审计**
- **先看池子规模**
- **最后再决定训不训**

---

## 6. 当前推荐执行顺序

如果现在要继续补数据，建议按这个顺序执行：

### JD

```bash
bash scripts/data/prepare_public_jd_pipeline.sh
```

### resume

```bash
bash scripts/data/prepare_public_resume_pipeline.sh
```

### match

```bash
bash scripts/data/prepare_public_match_pipeline.sh
```

### 最后统一看是否达到训练门槛

```bash
bash scripts/data/report_data_readiness.sh
```

当前 readiness 脚本会检查三类信息：

- 数量是否达到训练门槛：JD 使用 `data/sft_jd_quality/`，resume 使用 `data/sft_resume/`，match 使用 `data/sft_match/`。
- 格式是否可训练：assistant 输出必须是合法 JSON，样本 ID 不能重复，`train / valid / test` 之间不能存在内容级重复。
- 字段覆盖是否达标：按任务分别统计关键字段空值率，超过阈值则不认为 ready。

截至当前版本，三条线都已通过工程门槛：

- `JD`: `4400 / 550 / 550`，combined pool `37796`
- `resume`: `39132 / 5083 / 4952`，combined pool `4137`
- `match`: `3917 / 486 / 493`，combined pool `4896`
- `multitask`: `9800 / 1208`，任务配比约为 `JD 45% / resume 29% / match 27%`

---

## 7. 一句话总结

这三条链路的意思可以简单理解成：

- `导入`：把外部数据变成仓库能读的统一格式
- `审计`：先看质量和字段覆盖，不急着训练
- `候选池`：先筛出看起来可用的一层
- `合并建池`：把人工高质量集和公共可用样本合在一起
- `一键流水线`：把上面几步一次跑完

也就是说，项目现在不是“没数据处理逻辑”，而是：

> 已经把三条数据线的工程骨架搭齐了，接下来要做的是把真实公开文件真正落盘、跑完流程、拿到规模结果，再决定训练。
