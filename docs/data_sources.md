# 数据源可行性评估

更新时间：2026-05-17

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
bash scripts/data/refresh_tencent_data.sh auto
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
bash scripts/data/import_public_job_exports.sh
bash scripts/data/rebuild_data_pipeline.sh
```

注意：

1. 这类公开导出文件通常字段较少，适合作为 `岗位名称 / 公司 / 地点 / 薪资 / 学历 / 经验 / 福利` 的补充样本。
2. 它们不能替代官网 JD 详情页，尤其在“职责 / 任职要求”文本丰富度上会弱一些。
3. 所以当前策略是“官网公开职位 + 公开导出数据”混合扩量，而不是单押一个源。

## Moka 招聘官网

结论：推荐接入。

原因：

1. Moka 官方文档直接公开了招聘官网职位接口：
   - `GET https://api.mokahr.com/api-platform/v1/jobs/{orgId}`
2. 许多企业的公开招聘站都基于同一套接口，复用价值高。
3. 返回字段完整，包含 `title / description / education / minExperience / maxExperience / locations / department` 等。
4. 比字节、阿里、华为这类带网关或签名校验的站点更适合快速扩量。

当前项目已接入：

1. 抓取器：`jobmatch_tune.crawler.moka_careers`
2. 源清单：`configs/moka_sources.yaml`
3. 刷新脚本：

```bash
bash scripts/data/refresh_moka_data.sh
```

当前默认样本站包括：

1. 东方财富 `eastmoney`
2. 幻方量化 `high-flyer`
3. 岚图汽车 `voyah`
4. 施耐德电气 `se`
5. 江苏省规划设计集团 `jspdg`
6. 中控技术 `supcon`
7. 天弘基金 `thfund`
8. 阶跃星辰 `step`
9. 华虹集团 `huahong`
10. 友塔游戏 `xmyanquhr`
11. 北京智源人工智能研究院 `baai`
12. 烽火通信 `whfhtx`
13. 华勤技术 `hq`
14. 微步在线 `threatbook`

说明：

1. `configs/moka_sources.yaml` 里维护的是 `org_id / company / source_url / modes / site_ids`。
2. 后续新增 Moka 企业时，不需要改 crawler 逻辑，只需要补配置。
3. 当前会按职位文本自动判断 `language`，并对中文技术岗标记 `sft_ready=true`。

2026-05-17 当前导入结果：

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
6. `hf_job_educational_train_2026_05_17`
   - 文件：`data/external/public_job_exports/job_educational_train_2026-05-17.parquet`
   - 类型：中文招聘学历抽取 parquet
   - 当前导入：187606 条原始样本
7. `hf_job_educational_validation_2026_05_17`
   - 文件：`data/external/public_job_exports/job_educational_validation_2026-05-17.parquet`
   - 类型：中文招聘学历抽取 parquet
   - 当前导入：43744 条原始样本
8. `hf_job_educational_test_2026_05_17`
   - 文件：`data/external/public_job_exports/job_educational_test_2026-05-17.parquet`
   - 类型：中文招聘学历抽取 parquet
   - 当前导入：714 条原始样本
9. `zhaopin.jd.com`
   - 类型：京东公开招聘匿名 API
   - 当前抓取：3054 条原始样本
10. `moka_voyah`
   - 类型：Moka 招聘官网公开 API
   - 当前抓取：1199 条原始样本
11. `moka_se`
   - 类型：Moka 招聘官网公开 API
   - 当前抓取：691 条原始样本
12. `moka_eastmoney`
   - 类型：Moka 招聘官网公开 API
   - 当前抓取：239 条原始样本
13. `moka_supcon`
   - 类型：Moka 招聘官网公开 API
   - 当前抓取：152 条原始样本

当前整体规模：

1. `data/raw/tencent_jd_raw.jsonl`：933 条。
2. `data/raw/baidu_jd_raw.jsonl`：577 条。
3. `data/raw/jd_careers_raw.jsonl`：3054 条。
4. `data/raw/moka_jd_raw.jsonl`：2963 条。
5. `data/raw/public_job_datasets_raw.jsonl`：284637 条。
6. SQLite `jd_raw` 总量：292167 条。
7. `data/interim/jd_clean.jsonl`：292167 条。
8. `data/interim/jd_clean_dedup.jsonl`：273963 条。
9. 默认高质量 SFT `train/valid/test`：`1408 / 176 / 177`。
10. 扩展实验版 SFT `train/valid/test`：`4800 / 600 / 601`。
11. 去重后语言分布：
   - 中文：`221402`
   - 英文：`51330`
   - 其他 / 未知：`927`

注意：

1. 当前默认 `data/sft/` 没有直接吞掉新增公开语料。
2. 这是刻意设计，不是漏数。
3. 原因是：
   - GitHub 导出 CSV 往往只有标题、地点、薪资等浅字段；
   - `open-apply-jobs` 这批新增语料虽然量大且详情完整，但以英文 JD 为主；
   - 当前默认规则标注和字段拆分主链路仍然针对中文 JD。
4. 所以项目现在明确区分：
   - `jd_raw / jd_clean`：规模优先，用于扩语料和后续多语种增强
   - `data/sft/`：质量优先，只保留当前默认可控的中文训练集
5. `job-educational` 这批中文数据仍然不是完整人工主结构化标注口径，所以现在只允许极少量高置信样本进入扩展实验集，不再直接混入默认 SFT。
6. `data/sft_expanded/` 的 `20000` 目标是实验扩量上限，不代表默认高质量集应当按相同标准扩容。

本轮补充结论：

1. 网易当前候选公开路径返回 `404 NoSuchKey`，还没有验证出可复用职位 API。
2. 字节公开招聘页可以正常访问，但当前直接试探到的接口路径返回 404 或非职位 JSON。
3. 百度招聘 `https://talent.baidu.com/jobs/social-list` 可直接返回 SSR 的 `window.__INITIAL_DATA__`，并支持服务端 `search` 参数，当前已稳定抓到 577 条公开中文 JD。
4. 京东 `https://zhaopin.jd.com/web/job/job_info_list/3` 公开页的匿名接口已经确认：
   - `/web/job/job_allparams`
   - `/web/job/job_count`
   - `/web/job/job_list`
   当前已稳定抓取 `3054` 条公开职位。
5. 阿里 `https://talent.alibaba.com/` 当前更像门户 SPA 壳，公开职位列表接口还没确认。
6. 当前中文数据最多的地方是 Hugging Face 上公开发布的中文招聘 parquet，其次是京东、腾讯、百度这些公开官网接口。
7. 所以“高质量中文官网职位”这条线已经从只靠腾讯，扩展为“腾讯 + 百度 + 京东 + 中文公开 parquet”，字节 / 网易 / 阿里继续作为后续候选。

## 2026-05-17 最新官网探测结论

### 字节跳动

当前结论：**签名已还原，但网关仍未放行，暂未正式接入抓取器。**

已确认：

1. 公开社招页：`https://jobs.bytedance.com/experienced/position`
2. 前端 bundle 中可解析出官方接口定义：
   - `GET /api/v1/config/job/filters/2`
   - `GET /api/v1/search/job/posts`
   - `GET /api/v1/job/posts/{id}`
3. `portal_type=2` 对应大陆社招主站。
4. 过滤接口在 `job.toutiao.com` 域名下可匿名访问，例如：

```text
https://job.toutiao.com/api/v1/config/job/filters/2
```

当前阻塞：

1. `2350.894ccf9a.js` 分片里的 `57195` 模块已经能导出 `sign`，说明 `_signature` 生成逻辑已可复现。
2. 即使带上正确 `_signature`、`Portal-Channel: office`、`Portal-Platform: pc`、`website-path: society` 和公开页 cookie，职位列表接口仍会被网关重写到猎头平台页面。
3. 这说明当前剩余阻塞已经不是“找不到接口”或“不会算签名”，而更像是额外的网关上下文 / 浏览器态校验。
4. 在不继续模拟更重的浏览器环境前，当前不适合把字节职位列表抓取器直接接进主链路。

补充研究记录见：

- [bytedance_api_research.md](/share/home/lifr/workspace/code/job-match-tune/docs/bytedance_api_research.md)

### 华为

当前结论：**前端函数已确认，但匿名后端接口还没打通。**

已确认：

1. 公开社招页：`https://career.huawei.com/reccampportal/portal5/social-recruitment.html`
2. 页面会调用：
   - `HW.Portal.Recmng.Job.findJobListConf`
   - `HW.Portal.Recmng.Job.findJobList`
   - `HW.Portal.Recmng.Job.getJobDetail`
3. 从 `HwPortalRecmng.js` 可确认对应路由名称：
   - `services/rec/baseTalent/pub/callNewHr`
   - `services/portal/portalpub/getJob/newHr/page/{pageSize}/{curPage}`
   - `/reccampportal/services/portal/portalpub/getJobDetail/newHr?jobId=...&dataSource=...`

当前阻塞：

1. 直接匿名请求列表和筛选接口会返回 404 / unknown 错误。
2. 带页面 cookie 和 referer 后仍然没有打通。
3. 说明这条线至少还依赖额外的前端态或网关校验，当前不适合直接接入主链路。

### 阿里巴巴

当前结论：**前端路由已看到，匿名职位列表接口尚未确认。**

已确认：

1. 官网：`https://talent.alibaba.com/`
2. 前端 bundle 中能看到以下路由字符串：
   - `/off-campus/position-list`
   - `/campus/position-list`
   - `/social/position/recommendPosition.json`
   - `/social/position/similarPositions.json`

当前阻塞：

1. `position-list` 看起来更像前端路由，不是可直接调用的公开 JSON 接口。
2. 目前还没有在匿名条件下确认出稳定的列表接口和详情接口组合。

### 当前优先级调整

基于这一轮结果，中文官网数据源优先级建议更新为：

1. **已打通并稳定**：腾讯、百度、京东
2. **高价值但仍需继续攻**：字节跳动
3. **有前端线索但还没打通**：华为、阿里
4. **暂未继续**：网易、美团

一键入口：

```bash
bash scripts/data/refresh_baidu_data.sh
bash scripts/data/refresh_jd_data.sh
bash scripts/data/import_chinese_job_exports.sh
bash scripts/data/import_public_job_exports.sh
bash scripts/data/rebuild_data_pipeline.sh
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
