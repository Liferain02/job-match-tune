# 数据采集、SQLite 与清洗函数级实现

## 一、这部分在主项目中的位置

数据采集不是在线 Match 请求的一部分，而是离线数据生产入口。它要解决“把不同官网/API 的职位转成统一 raw contract”，但不能替来源做许可判断，也不能直接产出训练标签。

## 二、统一 Raw JD Contract

不同 crawler 最终尽量输出这些字段：

```text
id, source, url, crawl_time,
job_title, company, location, salary,
raw_text, html, meta
```

其中 `raw_text` 是后续清洗和规范化输入；`html` 只在 SQLite 原始层保留，JSONL 导出通常移除以控制体积；`meta` 保存平台特定字段，避免为了统一 schema 丢失信息。

## 三、通用网页抓取器

`src/jobmatch_tune/crawler/crawl_jd.py` 面向给定 URL 列表。

### 稳定 ID

`stable_id(url)` 对 URL 做 SHA1 并截取12位，形成 `jd_<digest>`。重复抓同一 URL 会得到相同主键，便于 SQLite upsert。它不能识别两个不同 URL 指向同一岗位，因此后面仍需内容去重。

### 请求

`fetch_url` 使用 timeout、自定义 User-Agent 和中文优先 Accept-Language，调用 `raise_for_status`，并依据 apparent encoding 解码。

当前通用抓取器没有复杂指数重试；平台特定适配器可以有自己的 retry。批量 URL 间按 `interval_seconds` sleep，避免无节制请求。

### 结构化优先

`extract_job_posting` 先查 `<script type="application/ld+json">` 中的 `JobPosting`。若存在，`parse_job_posting` 从 title、hiringOrganization、jobLocation、description 提取字段。

优先 JSON-LD 的原因是语义比 DOM selector 稳定；但不少招聘站不提供，因此仍有正文 fallback。

### 正文 fallback

`extract_main_text` 优先调用 trafilatura；结果至少80字符才接受，否则用 `clean_text(html, is_html=True)`。标题优先 h1，其次 title；BeautifulSoup 不可用时还有有限正则 fallback。

### 页面有效性

`is_valid_job_page` 拒绝404/not found、robots noindex，以及正文少于120字符或完全没有“岗位/职位/任职/职责”关键词的页面。

这是启发式过滤，不是网页分类器；误报/漏报应进入数据审计，而不是无限堆 selector。

## 四、平台 API 适配器示例

`src/jobmatch_tune/crawler/tencent_careers.py` 展示 API 型来源。

`fetch_tencent_posts` 组装 keyword/page/pageSize/language 参数，最多重试3次，并对网络错误、JSON 错误和业务 Code 非200统一重试，sleep 随 attempt 线性增加。

`convert_post` 将 PostId、名称、地点、事业群、职责和年限转成统一 contract；原平台字段如 category、product 和更新时间保存在 meta。

`crawl_tencent` 在关键词和页码上循环，通过 `seen post_id` 去重，可按 CategoryName allowlist 过滤，并同时受 limit、max pages、total count 和 interval 控制。

写文件时默认读取已有 JSONL 按 ID 合并，避免一次关键词增量覆盖历史；SQLite 仍使用 upsert。

其他官网 adapter 位于同目录，核心原则相同：平台解析留在 adapter，统一下游字段不随来源扩散条件分支。

## 五、SQLite 数据层

`src/jobmatch_tune/database.py` 使用 SQLite，当前 schema 包含：

- `jd_raw`：原始职位、HTML、meta；ID 主键，URL 索引。
- `jd_clean`：关联 raw_id 的清洗正文、sections 和 labels。
- `resume_clean`：简历清洗正文与 labels。
- `sft_samples`：task、split 和 messages；split/task 复合索引。

数据库初始化启用 WAL，适合本地读写并发但不是分布式数据库。

### Upsert

`upsert_jd_raw` 和 `upsert_jd_clean` 使用 `INSERT ... ON CONFLICT(id) DO UPDATE`。dict 中 meta/sections/labels 在写入前序列化为中文 JSON。

好处是重复刷新同一来源不产生主键重复；风险是源站内容更新会覆盖原内容，因此 crawl_time 和后续 manifest 必须一起记录。当前没有保存每次变更历史。

### 批量读取

`iter_table_batches` 用 cursor `fetchmany(batch_size)` 流式读取，限制 table 只能来自 allowlist，防止用户字符串直接进入 SQL table 名；batch size 必须为正。

`fetch_table` 适合测试和小表，会一次加载全部行。几十万 JD 的正式管线应优先 batch iterator。

### 为什么使用 SQLite

项目是单机研究/演示，SQLite 部署成本低、事务和 schema 明确，比一开始引入 PostgreSQL 更合适。若多进程大规模爬取、远程协作或在线更新成为需求，再迁移数据库。

## 六、文本清洗

`src/jobmatch_tune/preprocess/clean_text.py::clean_text` 的顺序很重要：

1. HTML 输入先 `strip_html`，移除 script/style/noscript/nav/footer/header。
2. HTML entity `unescape`。
3. `mask_private_info` 替换手机号、邮箱和微信号。
4. `normalize_space` 统一全角空格、水平空白、空行和首尾空白。
5. `remove_boilerplate` 删除投递、沟通、收藏、地图等按钮文案。
6. `deduplicate_lines` 按去空格小写 key 保留首次行。
7. 再执行一次空白规范。

为什么先去 HTML 再去重？DOM 导航和脚本会制造大量无意义行。为什么最后再 normalize？中间删除步骤会重新产生空行。

PII regex 是基础保护，不保证识别姓名、地址或所有变形号码。Resume 还有专门的 privacy 模块和 readiness。

## 七、确定重复与近重复

`src/jobmatch_tune/preprocess/deduplicate.py` 不是全局两两比较，因为几十万文本会是平方复杂度。

### Bucket

`build_bucket_key` 优先使用 source+company+title+location；缺公司时使用 source+title+location+正文前缀。只在同 bucket 内比较，降低复杂度。

这个设计可能漏掉跨 source 复制的同一职位，因此 Gold isolation 不能只依赖这一步。

### Exact Fingerprint

`fingerprint` 对小写、去空白文本做 SHA1。它捕获空白差异后的完全重复。

### Near Duplicate

`normalize_similarity_text` 仅保留中文、字母和数字；`build_shingles` 默认生成5-gram集合；`text_similarity` 取 Jaccard 与 SequenceMatcher ratio 的较大值。

`iter_deduplicated_rows` 默认 threshold 0.9，保留 bucket 中第一条，后续 exact fingerprint 或 near similarity 超阈值则丢弃。

为何取 max？局部编辑可能让 shingle 或 sequence 中一项更敏感；取 max 更积极去重。代价是相似模板岗位可能被合并，因此 bucket 和阈值需要按任务抽查。

## 八、规范化与 Freshness

清洗后 `normalize_jd.py` 根据 title、text、meta 和 label schema 提取字段。Normalization manifest 记录 raw DB 状态、输出 hash 和 transform hash。

`pipeline_freshness.py` 再比较输入、转换代码和输出 mtime/hash，判断某条依赖是否需要重建。当前 SFT 链 fresh，preference 链 stale。

Freshness 只回答“当前产物是否落后于上游”，不回答标签是否真实、来源是否合法或数据是否分布均衡。

## 九、抓取失败如何处理

- 单 URL 通用 crawler 的网络异常会中止当前命令，便于发现系统性问题。
- 腾讯等批量 adapter 对分页做有限重试；某关键词某页持续失败会打印并跳出该关键词分页。
- 非职位页面被明确 skip。
- 输出使用稳定 ID 和 upsert，命令可重复执行。

项目没有分布式任务队列、断点调度和全局速率协调；在当前单机数据刷新规模下是刻意取舍。

## 十、许可和伦理边界

Crawler 的存在不意味着所有站点内容可以训练。抓取前应遵守 robots、服务条款、访问频率和用途限制；公共职位描述也可能包含联系方式。来源是否训练可用由 source registry/admission 决定。

面试中不要把“爬了29万 JD”当核心亮点。更好的表达是：建立来源适配和 raw/clean 分层，并在训练前单独做许可、质量和来源集中审计。

## 十一、如果规模扩大怎么演进

真实需求出现后可以按顺序：

1. 把 source state、etag/last-modified 和失败重试持久化。
2. 对 API 来源做增量 cursor。
3. 将 raw HTML 放对象存储，数据库只保 URI/hash。
4. 使用任务队列协调速率和重试。
5. 引入 PostgreSQL 保存版本历史。
6. 建来源级质量、许可和删除审计。

这些是演进方向，不是当前已实现能力。
