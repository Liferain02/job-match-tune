# 字节招聘公开 API 研究记录

更新时间：`2026-05-17`

这份文档只记录一件事：字节招聘公开页到底探到了什么，当前为什么还没有正式接进抓取主链路。

## 当前结论

已经拿到两块关键信息：

1. 前端公开 bundle 里能确认职位相关接口定义。
2. `_signature` 生成逻辑已经能在本地复现。

但列表接口在匿名 HTTP 请求下仍会被网关改写到猎头平台页面，所以 **当前还不能稳定批量抓取职位列表**。

这意味着：

1. 问题已经不是“找不到接口”。
2. 也不是“不会算 `_signature`”。
3. 剩下的阻塞更可能是网关上下文、cookie、风控态或浏览器环境校验。

## 已确认的公开页和接口

公开页：

- `https://jobs.bytedance.com/experienced/position`

前端 bundle 可见的接口：

- `GET /api/v1/config/job/filters/2`
- `GET /api/v1/search/job/posts`
- `GET /api/v1/job/posts/{id}`

其中：

- `portal_type=2` 对应大陆社招主站
- `website-path` 来自页面注入配置，当前是 `society`
- 前端请求头里还会带：
  - `Portal-Channel: office`
  - `Portal-Platform: pc`

## 已打通的部分

过滤接口可匿名访问：

```text
https://job.toutiao.com/api/v1/config/job/filters/2
```

这说明：

1. 路由本身存在。
2. 不是所有 `/api/v1/*` 都完全封死。

## `_signature` 还原结果

在 `2350.894ccf9a.js` 分片里可以定位到模块 `57195`，该模块导出：

- `dfp`
- `sign`

项目里新增了一个最小复现实验脚本：

```bash
node scripts/research/bytedance_sign.js \
  --chunk /tmp/bytedance_chunks_2350.894ccf9a.js \
  --url '/api/v1/search/job/posts?keyword=&limit=10&offset=0&portal_type=2'
```

脚本会直接输出 `_signature`。

这说明签名层已经被拆开，不需要再手工读混淆代码。

## 当前还没过的地方

即使带上：

- 正确 `_signature`
- `Portal-Channel: office`
- `Portal-Platform: pc`
- `website-path: society`
- 公开页 referer
- 公开页 cookie

列表接口仍然会返回“字节跳动猎头平台”HTML，而不是职位 JSON。

也就是说，当前阻塞点大概率在：

1. 网关路由上下文
2. 浏览器运行态
3. 额外 cookie / token
4. 更细的风控参数

## 当前建议

现阶段不要把字节抓取器直接接进主链路。

更合理的做法是：

1. 继续使用已稳定的中文公开源：腾讯、百度、京东。
2. 把这份研究结果保留，避免以后从零开始逆向。
3. 如果后续要继续打字节，直接从“网关上下文 / 浏览器态复现”开始，而不是重新找接口和重做签名。

## 复现步骤

1. 下载公开页 HTML。
2. 下载公开页首屏 JS 分片。
3. 在分片里定位 `57195:function`。
4. 用 `scripts/research/bytedance_sign.js` 计算 `_signature`。
5. 再用带签名的请求验证列表接口是否仍被网关改写。

## 为什么先停在这里

项目当前目标是扩充 **可持续复用的中文职位数据**。

对这个目标来说：

1. 字节这条线的研究已经有产出。
2. 但还没有达到“稳定抓取”的工程条件。
3. 继续硬攻的边际收益，当前低于继续接入其他已公开的中文官网源。
