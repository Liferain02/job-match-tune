# Data Pipeline Full Guide

## 1. 目标

这份文档只讲数据，不讲模型训练细节。

项目里的数据链路目标一直是两件事：

1. 把公开招聘 JD 持续沉淀成一个可复用的原始职位库。
2. 把原始职位库持续加工成适合当前任务口径的 SFT 数据。

这里的“当前任务口径”指的是固定的结构化抽取字段：

- `岗位方向`
- `核心职责`
- `必备技能`
- `加分项`
- `经验要求`
- `学历要求`

因此，原始 JD 多并不等于可训练样本多。中间还要经过抓取、统一 schema、清洗、去重、规则标注、样本筛选和分层。

---

## 2. 数据链路总览

当前完整链路是：

1. 抓取或导入公开职位数据
2. 写入 `raw JSONL + SQLite`
3. 统一映射到 `jd_raw` 原始表
4. 清洗、抽取段落、规则标注，生成 `jd_clean.jsonl`
5. 去重，生成 `jd_clean_dedup.jsonl`
6. 基于质量分层构造：
   - 默认高质量集 `data/sft/`
   - 扩展实验集 `data/sft_expanded/`
   - 中英弱标注集 `data/sft_multilingual_weak/`

对应代码和脚本：

- 抓取器：`src/jobmatch_tune/crawler/`
- 清洗与规则：`src/jobmatch_tune/preprocess/`
- 数据构造：`src/jobmatch_tune/dataset/`
- 数据脚本：`scripts/data/`

---

## 3. 第一阶段：从单一官网源起步

项目最开始不是大规模抓站，而是先验证一条最小可行链路。

### 3.1 腾讯公开招聘

第一条稳定源是腾讯公开招聘。

原因：

- 公开接口相对稳定
- 技术岗占比高
- JD 结构完整
- 职责、要求、学历、经验等字段比较容易从正文抽出来

抓取器：

- [tencent_careers.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/tencent_careers.py)

关键词配置：

- [tencent_keywords.txt](/share/home/lifr/workspace/code/job-match-tune/configs/tencent_keywords.txt)

抓取方式：

- 按关键词分页请求腾讯公开职位接口
- 详情页再抓完整 JD
- 结果同时写：
  - `data/raw/tencent_jd_raw.jsonl`
  - SQLite `jd_raw`

这个阶段的目标不是追求大，而是先把 `raw -> clean -> sft` 跑通。

---

## 4. 第二阶段：扩公开官网中文源

只靠腾讯不够，后面按“稳定、匿名、结构完整”的优先级，逐步接了更多中文官网源。

### 4.1 百度招聘

百度招聘不是直接调一个公开 JSON 列表，而是利用 SSR 页面里埋的 `window.__INITIAL_DATA__`。

抓取器：

- [baidu_talent.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/baidu_talent.py)

关键词配置：

- [baidu_keywords.txt](/share/home/lifr/workspace/code/job-match-tune/configs/baidu_keywords.txt)

做法：

1. 搜索页按关键词拿职位列表
2. 详情页解析 SSR 数据
3. 提取岗位标题、正文、地点、公司等字段
4. 写入：
   - `data/raw/baidu_jd_raw.jsonl`
   - SQLite `jd_raw`

### 4.2 京东招聘

京东这条线价值很高，因为匿名列表接口比较稳定。

抓取器：

- [jd_careers.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/jd_careers.py)

做法：

1. 调京东公开招聘匿名 API 拉职位列表
2. 分页遍历职位
3. 再抓详情并映射到统一 raw schema
4. 写入：
   - `data/raw/jd_careers_raw.jsonl`
   - SQLite `jd_raw`

### 4.3 Moka 托管招聘官网

Moka 是后面最有效的中文官网扩量来源之一。

原因：

- 很多公司官网都基于同一套 Moka 招聘 API
- 接口一致性好
- JD 通常比较完整
- 技术公司占比高

抓取器：

- [moka_careers.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/moka_careers.py)

源配置：

- [moka_sources.yaml](/share/home/lifr/workspace/code/job-match-tune/configs/moka_sources.yaml)

做法：

1. 在 `moka_sources.yaml` 里维护可用公司源
2. 每个源通过 `org_id` 或站点配置调用同一套职位接口
3. 统一拉列表和详情
4. 写入：
   - `data/raw/moka_jd_raw.jsonl`
   - SQLite `jd_raw`

这条线后面陆续接入了：

- 岚图
- 申万宏源
- 东方财富
- 中控
- 智源
- 华勤
- 微步
- 华虹
- 阶跃星辰
- 若干游戏、云、基础设施公司

### 4.4 为什么没有把字节、阿里、华为接进主链路

不是没研究，而是性价比不够高。

结论是：

- 字节：接口、签名都定位到了，但网关态校验更重，匿名直调不稳定
- 华为：前端线索明确，但匿名 API 没真正打通
- 阿里：更像 SPA 壳，列表接口没有确认到稳定公开 JSON

所以主链路优先接的是：

- 腾讯
- 百度
- 京东
- Moka

这几条匿名稳定的中文官网源。

---

## 5. 第三阶段：导入公开数据集扩原始语料池

只靠官网抓取，原始语料增长还是不够快。于是后面把“公开导出数据集”也接进来了。

统一导入器：

- [import_public_job_data.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/import_public_job_data.py)

脚本：

- [import_public_job_exports.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/import_public_job_exports.sh)
- [import_chinese_job_exports.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/import_chinese_job_exports.sh)

### 5.1 接入过的数据集

#### GitHub 导出

- `jhcoco/bosszp` CSV
- `WorkAggregation` CSV

#### Hugging Face 开放职位语料

- `open-apply-jobs`
  - Greenhouse
  - Ashby
  - Lever

#### Hugging Face 中文招聘学历数据

- `job-educational-parser-dataset-08-0-0805`

这一步的作用不是“直接拿来训默认模型”，而是先把原始职位库做大。

---

## 6. 原始数据怎么落盘

每条职位最终会进入两个地方：

1. `data/raw/*.jsonl`
2. `data/jobmatch_tune.sqlite3` 的 `jd_raw`

这样做有两个目的：

- JSONL 适合审查和离线重放
- SQLite 适合统一重建下游、做统计和增量写入

raw 层的原则是：

- 尽量保留来源原貌
- 不在 raw 层做过多“聪明处理”
- 只做最基本的字段映射和去空值

---

## 7. 第四阶段：把原始职位统一成 clean schema

raw 数据来自不同站点，不统一就没法训练。

统一清洗入口：

- [normalize_jd.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/normalize_jd.py)

对应脚本：

- [rebuild_data_pipeline.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/rebuild_data_pipeline.sh)

### 7.1 `normalize_jd` 做了什么

它做的事情包括：

1. 统一字段名
2. 规范正文文本
3. 提取结构段落
4. 抽语言
5. 做规则标注
6. 标记 `sft_ready`

输出文件：

- `data/interim/jd_clean.jsonl`

### 7.2 段落拆分

在 clean 层会尝试把正文拆成：

- `responsibilities`
- `requirements`
- `bonus`

这一步是后面构造训练 labels 的基础。

### 7.3 基础规则抽取

规则都放在：

- [jd_field_rules.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/jd_field_rules.py)

主要负责：

- `岗位方向` 推断
- `学历要求` 抽取
- `经验要求` 抽取
- `技能` 抽取

这一步很关键，因为很多数据源本身并没有结构化标签，只能靠规则和正文信号做弱监督。

---

## 8. 第五阶段：去重

清洗后的职位很多会重复。

去重入口：

- [deduplicate.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/deduplicate.py)

输出：

- `data/interim/jd_clean_dedup.jsonl`

去重目的：

- 避免同一个职位被多个渠道重复写入
- 避免训练时重复样本放大偏差
- 保持评估集和训练集的分布更干净

---

## 9. 第六阶段：为什么“原始数据多”不等于“高质量 SFT 多”

这是这个项目里最重要的一个经验。

项目过程中，原始职位库已经做到二十多万中文、总量接近三十万。但默认高质量 SFT 集并没有同步到万级。

原因不是抓不到数据，而是下面这几个问题：

1. 很多职位不是当前任务范围内的软件/AI 技术岗
2. 很多职位是业务、运营、销售、供应链、制造、交付岗
3. 很多职位正文不完整
4. 很多职位方向在当前 schema 下不应该硬标

所以后面明确把数据分成了多层，而不是只保留一个 `data/sft/`。

---

## 10. 数据分层是怎么做的

核心逻辑在：

- [build_sft_dataset.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/dataset/build_sft_dataset.py)

### 10.1 默认高质量集 `data/sft/`

这是默认主训练集，只保留高信任样本。

当前准入条件大致是：

1. 中文
2. `sft_ready=true`
3. 来源属于高信任官网源
4. 职位标题有明确技术信号
5. 标题没有明显业务/销售/运营信号
6. 必须能识别出 `岗位方向`
7. 职责/要求结构要基本完整
8. 至少有学历、经验、技能中的一个

这里故意收得严，因为这套数据要直接拿去做默认 SFT。

### 10.2 扩展实验集 `data/sft_expanded/`

这是第二层数据。

特点：

- 允许更多 `sft_ready` 样本进入
- 可以混入少量高置信弱标注技术岗
- 适合做扩量实验、第二阶段 SFT、对照实验

它不是默认 demo 版本的数据来源。

### 10.3 中英混合弱标注集 `data/sft_multilingual_weak/`

这是规模优先的实验集。

用途：

- 做扩量实验
- 做第二阶段增量训练
- 做多语言或弱监督对照

不适合直接替代默认高质量集。

---

## 11. 为什么后来把“2 万高质量集”撤回了

这个问题必须说明白。

中间有一段时间，我确实把默认集拉到过 `2 万`。但后面在你明确强调“数据质量是核心中的核心”之后，我重新审了准入逻辑和样本来源，结论是：

- 那版 `2 万` 里混了过多弱标注样本
- 还混了不少其实不属于当前 schema 的职位
- 如果直接拿去训默认模型，会把错误方向和错误边界学进去

因此后面做了两件事：

1. 把默认 `data/sft/` 收回到严格高质量口径
2. 把大样本量方案拆到 `data/sft_expanded/` 和其他实验链路

这不是推翻数据工作，而是把“数量”和“默认训练质量”分开。

---

## 12. 第七阶段：修复最严重的数据质量问题

### 12.1 问题一：标题规则过窄，误杀真实技术岗

后面我们发现默认高质量集掉得太快，不是因为数据不够，而是因为标题准入规则太窄。

当时统计过的主要失败原因里，`title_not_included` 是最大头。

后来做的修复包括：

- 扩充技术标题信号
- 加入方向相关的标题补充规则
- 把 `产品经理 / 客户端 / 嵌入式 / 运维 / 安全` 等真实技术类纳入 schema

### 12.2 问题二：上游方向推断会把未知职位默认映射到第一个类

这是后面发现的更严重问题。

之前 `infer_job_direction()` 在完全匹配不到时，会回落到 schema 的第一个类。这个行为会把很多非技术岗硬标成 `AI应用开发`。

后面已经修掉：

- 匹配不到就返回空，不再伪造方向
- 只对明确技术模式做方向归类

这一步虽然会让默认高质量集数量下降，但质量会明显更可信。

### 12.3 问题三：高信任官网里仍然有大量“方向为空”的样本

修掉错误回退后，出现了一个更真实的结果：

- 大量高信任官网样本现在 `岗位方向=""`

这不是坏事，而是说明：

- 之前有一部分样本其实被错误标了方向
- 现在需要从这些空方向样本里“回收真正的技术岗”

后续优化重点就是这一步。

---

## 13. 当前数据脚本怎么用

### 13.1 抓官网数据

一键刷新官网源：

```bash
bash scripts/data/refresh_official_job_data.sh
```

单独刷新某个源：

```bash
bash scripts/data/refresh_tencent_data.sh auto
bash scripts/data/refresh_baidu_data.sh
bash scripts/data/refresh_jd_data.sh
bash scripts/data/refresh_moka_data.sh
```

### 13.2 导入公开数据集

```bash
bash scripts/data/import_public_job_exports.sh
bash scripts/data/import_chinese_job_exports.sh
```

### 13.3 重建 clean 和训练集

```bash
bash scripts/data/rebuild_data_pipeline.sh
```

它会依次产出：

1. `data/interim/jd_clean.jsonl`
2. `data/interim/jd_clean_dedup.jsonl`
3. `data/sft/`
4. `data/sft_expanded/`

---

## 14. 当前数据状态

截至当前这版仓库，数据层大致是：

- `jd_clean.jsonl`: `292167`
- `jd_clean_dedup.jsonl`: `267949`

当前默认高质量集：

- `data/sft/train.jsonl`: `1421`
- `data/sft/valid.jsonl`: `177`
- `data/sft/test.jsonl`: `179`

当前扩展实验集：

- `data/sft_expanded/train.jsonl`: `4524`
- `data/sft_expanded/valid.jsonl`: `565`
- `data/sft_expanded/test.jsonl`: `566`

这里最重要的不是数字本身，而是口径：

- `data/sft/` 是默认主训练集
- `data/sft_expanded/` 是扩量实验集
- 不能再把这两层混为一谈

---

## 15. 这条数据链路目前最清楚的结论

### 15.1 做对了的事

1. 没有只靠一个站点
2. 原始职位库已经做大
3. 官网源和公开导出源已经统一到一套 clean schema
4. 数据已经分层，不再把弱标注直接混进默认集
5. 已经把一个严重的方向误标 bug 修掉

### 15.2 还没做完的事

1. 默认高质量集离 `2 万` 还很远
2. 当前 `岗位方向` schema 还是偏窄
3. 高信任官网里还有大量 `方向为空` 的职位没有被正确回收
4. 默认集里的方向分布还不够均衡

### 15.3 接下来最应该做的事

1. 持续扩高信任中文官网源
2. 从 `方向为空` 的高信任样本里回收真实技术岗
3. 必要时扩 schema，而不是硬把更多职位塞进现有标签
4. 始终把默认集和扩展集分开管理

---

## 16. 相关代码入口

抓取器：

- [tencent_careers.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/tencent_careers.py)
- [baidu_talent.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/baidu_talent.py)
- [jd_careers.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/jd_careers.py)
- [moka_careers.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/moka_careers.py)
- [import_public_job_data.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/crawler/import_public_job_data.py)

清洗与规则：

- [normalize_jd.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/normalize_jd.py)
- [deduplicate.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/deduplicate.py)
- [jd_field_rules.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/preprocess/jd_field_rules.py)

数据构造：

- [build_sft_dataset.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/dataset/build_sft_dataset.py)

脚本：

- [refresh_official_job_data.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/refresh_official_job_data.sh)
- [rebuild_data_pipeline.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/rebuild_data_pipeline.sh)
- [import_public_job_exports.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/import_public_job_exports.sh)
- [import_chinese_job_exports.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/import_chinese_job_exports.sh)
