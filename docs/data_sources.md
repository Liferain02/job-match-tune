# 数据源可行性评估

更新时间：2026-05-10

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

当前已使用 `configs/tencent_keywords.txt` 批量抓取腾讯技术类岗位，并按 `PostId` 去重写入 SQLite。爬虫默认会将新结果与现有 `data/raw/tencent_jd_raw.jsonl` 按 `id` 增量合并，避免覆盖历史批次。2026-05-10 本地结果：

1. `data/raw/tencent_jd_raw.jsonl`：892 条。
2. SQLite `jd_raw`：895 条，其中腾讯 894 条、BeBee 1 条。
3. `data/interim/jd_clean.jsonl`：895 条，包含历史 BeBee 样例和腾讯历史入库记录。
4. `data/interim/jd_clean_dedup.jsonl`：887 条。
5. `data/sft/train.jsonl`：709 条。
6. `data/sft/valid.jsonl`：88 条。
7. `data/sft/test.jsonl`：90 条。

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
