# 数据源可行性评估

更新时间：2026-05-16

## 结论

优先级建议：

1. 公司招聘官网公开 API / 公开职位页。
2. 公开聚合站中无需登录、无验证码、无 robots 限制的详情页。
3. BOSS 直聘、智联招聘这类综合招聘平台不作为主数据源。

## BOSS 直聘

结论：不建议批量爬取职位搜索页和职位详情页。

原因：

1. `https://www.zhipin.com/robots.txt` 对大量搜索参数、推荐页、部分职位详情路径和所有带 query 的 URL 做了限制。
2. 搜索和详情页通常带登录态、风控和反爬逻辑。
3. 项目不应绕过登录、验证码、风控或反爬。

可接受用法：

1. 不接入自动爬虫。
2. 如果用户手工保存少量公开 JD 文本，可作为本地导入数据处理。

## 智联招聘

结论：不建议批量爬取。

原因：

1. 访问 `https://www.zhaopin.com/robots.txt` 时触发 Tencent Cloud EdgeOne 安全验证。
2. 智联首页明确提示“未经 Zhaopin.com 同意，不得转载本网站之所有招聘信息及作品”。
3. 这类平台职位数据属于核心内容资产，不适合作为未经授权的训练数据主来源。

可接受用法：

1. 不接入自动爬虫。
2. 优先寻找智联官方开放接口、合作授权或公开数据集。

## 腾讯招聘

结论：推荐接入。

原因：

1. `https://careers.tencent.com/search.html` 为公开职位搜索页。
2. 前端公开 API 可返回职位名称、地点、事业群、岗位职责、更新时间等字段。
3. 当前项目已实现 `jobmatch_tune.crawler.tencent_careers`。

示例：

```bash
python -m jobmatch_tune.crawler.tencent_careers --keyword 大模型 --limit 20
```

当前已使用 `configs/tencent_keywords.txt` 批量抓取腾讯技术类岗位，并按 `PostId` 去重写入 SQLite。爬虫默认会将新结果与现有 `data/raw/tencent_jd_raw.jsonl` 按 `id` 增量合并，避免覆盖历史批次。2026-05-15 当前结果：

1. `data/raw/tencent_jd_raw.jsonl`：933 条。
2. SQLite `jd_raw`：至少包含当前 933 条 raw 合并结果与历史样例。
3. `data/interim/jd_clean.jsonl`：936 条。
4. `data/interim/jd_clean_dedup.jsonl`：927 条。
5. `data/sft/train.jsonl`：741 条。
6. `data/sft/valid.jsonl`：92 条。
7. `data/sft/test.jsonl`：94 条。

当前运行环境还存在外网 / DNS 不稳定的情况，因此仓库增加了：

```bash
bash scripts/refresh_tencent_data.sh auto
```

这个入口会在抓取失败时自动退回到“只重建清洗和训练数据”的模式。

## 公开数据集导入

结论：可以作为扩量补充源。

原因：

1. 不直接访问受风控的招聘平台页面。
2. 先导入公开发布的职位导出文件，再统一走现有清洗、去重、SFT 管道。
3. 适合在当前网络和反爬约束下快速扩充 `jd_raw`。

当前已接入：

1. GitHub `jhcoco/bosszp` 公开 CSV。
2. 导入脚本：`jobmatch_tune.crawler.import_public_job_data`
3. 源清单：`configs/public_job_sources.yaml`
4. 一键入口：

```bash
bash scripts/import_public_job_exports.sh
bash scripts/rebuild_data_pipeline.sh
```

注意：

1. 这类公开导出文件通常字段较少，适合作为 `岗位名称 / 公司 / 地点 / 薪资 / 学历 / 经验 / 福利` 的补充样本。
2. 它们不能替代官网 JD 详情页，尤其在“职责 / 任职要求”文本丰富度上会弱一些。
3. 所以当前策略是“官网公开职位 + 公开导出数据”混合扩量，而不是单押一个源。

2026-05-16 当前导入结果：

已接入公开源：

1. `github_jhcoco_bosszp`
   - 文件：`data/external/public_job_exports/jhcoco_bosszp.csv`
   - 类型：BOSS 导出 CSV
   - SQLite `jd_raw`：125 条有效样本
2. `github_workaggregation_test`
   - 文件：`data/external/public_job_exports/workaggregation_test.csv`
   - 类型：多站聚合 CSV
   - SQLite `jd_raw`：270 条有效样本
3. `hf_open_apply_greenhouse_tech_2026_05_15`
   - 文件：`data/external/public_job_exports/open_apply_greenhouse_2026-05-15.parquet`
   - 类型：公开 ATS parquet 分片
   - 当前过滤后导入：20000 条技术岗位
4. `hf_open_apply_ashby_tech_2026_05_15`
   - 文件：`data/external/public_job_exports/open_apply_ashby_2026-05-15.parquet`
   - 类型：公开 ATS parquet 分片
   - 当前过滤后导入：18970 条技术岗位
5. `hf_open_apply_lever_tech_2026_05_15`
   - 文件：`data/external/public_job_exports/open_apply_lever_2026-05-15.parquet`
   - 类型：公开 ATS parquet 分片
   - 当前过滤后导入：13208 条技术岗位

当前整体规模：

1. `data/raw/tencent_jd_raw.jsonl`：933 条。
2. `data/raw/public_job_datasets_raw.jsonl`：52573 条。
3. `data/raw/baidu_jd_raw.jsonl`：由 `jobmatch_tune.crawler.baidu_talent` 增量写入。
4. SQLite `jd_raw`：包含腾讯、百度和公开导出源的统一原始库。
5. `data/interim/jd_clean.jsonl`：由 SQLite `jd_raw` 全量重建。
6. `data/interim/jd_clean_dedup.jsonl`：由 `jd_clean` 去重得到。
7. 默认高质量 SFT `train/valid/test`：仅保留当前高质量中文样本。

注意：

1. 当前默认 `data/sft/` 没有直接吞掉新增 2 万级公开语料。
2. 这是刻意设计，不是漏数。
3. 原因是：
   - GitHub 导出 CSV 往往只有标题、地点、薪资等浅字段；
   - `open-apply-jobs` 这批新增语料虽然量大且详情完整，但以英文 JD 为主；
   - 当前默认规则标注和字段拆分主链路仍然针对中文 JD。
4. 所以项目现在明确区分：
   - `jd_raw / jd_clean`：规模优先，用于扩语料和后续多语种增强
   - `data/sft/`：质量优先，只保留当前默认可控的中文训练集

本轮补充结论：

1. 网易当前候选公开路径返回 `404 NoSuchKey`，还没有验证出可复用职位 API。
2. 字节公开招聘页可以正常访问，但当前直接试探到的接口路径返回 404 或非职位 JSON。
3. 百度招聘 `https://talent.baidu.com/jobs/social-list` 可直接返回 SSR 的 `window.__INITIAL_DATA__`，并支持服务端 `search` 参数，已经适合做公开中文 JD 增量抓取。
4. 所以“高质量中文官网职位”这条线已经从只靠腾讯，扩展为“腾讯 + 百度 + 公开导出源”，字节 / 网易 / JD / 阿里继续作为后续候选。

一键入口：

```bash
bash scripts/refresh_baidu_data.sh
bash scripts/import_public_job_exports.sh
bash scripts/rebuild_data_pipeline.sh
```

## 字节跳动招聘

结论：可作为候选数据源，但需要进一步确认 API 参数和职位详情接口。

原因：

1. `https://jobs.bytedance.com/robots.txt` 明确允许 `/experienced`、`/society`、`/campus`、`/en`、`/jp`，禁止 `/referral`。
2. `https://jobs.bytedance.com/experienced/position` 是公开招聘列表页。
3. 页面为前端渲染，数据接口需要进一步解析，不能误用猎头平台或内部接口。

建议：

1. 后续单独实现 ByteDance crawler。
2. 只使用公开招聘站允许路径。
3. 不访问 referral、登录态、内推、猎头或候选人相关接口。

## 其他公司官网

推荐优先尝试：

1. 腾讯招聘：`careers.tencent.com`
2. 字节跳动招聘：`jobs.bytedance.com`
3. 阿里巴巴招聘：`talent.alibaba.com`
4. 美团招聘、百度招聘、京东招聘、快手招聘等公开官网

接入标准：

1. 不需要登录。
2. 不触发验证码。
3. robots 未禁止目标路径。
4. 页面或 API 返回的是公开职位信息。
5. 只采集岗位名称、公司、地点、职责、要求、发布时间/更新时间、URL。
6. 不采集候选人、招聘者联系方式、聊天、投递、内推、登录态数据。
