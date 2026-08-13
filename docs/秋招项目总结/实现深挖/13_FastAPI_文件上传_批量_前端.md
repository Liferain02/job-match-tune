# FastAPI、文件上传、批量、前端

## API 与前端职责

`api/server.py` 是当前单进程编排层：校验请求、加载模型、解析、匹配、批量调度和返回执行元数据。前端位于 `frontend/`，只收集输入、调用 API、呈现结构和导出 Markdown，不复制规则逻辑。

## 入口与执行

- `/health`、`/api/status` 暴露后端和加载状态；`/api/warmup` 显式预热。
- `/api/parse`、`/api/match` 接收 Pydantic JSON；两个解析在 vLLM 且并发槽至少为 2 时并行，Transformers 因共享模型锁保持顺序。
- `/api/batch_parse`、`/api/batch_match` 在安全并发范围执行并逐项返回状态。
- 文件接口先做有界读取与 `validate_upload_content`，再调用提取器；结构化 HTTP detail 让前端区分大小、格式伪装和损坏。
- Vue 3/Vite 依赖由 `frontend/package-lock.json` 固定。`start_frontend.sh` 先 `npm run build` 再托管 `dist`；`start_project.sh` 与 `stop_project.sh` 管理 API/前端 PID 和日志。

前端 `App.js` 管理 task/request mode、表单和结果；`components/ResultPanel.js` 呈现结果；`utils/text.js` 生成报告。UI 明示“启发式匹配分数”，不叫成功概率。

## 示例、错误、测试和限制

上传名为 `.pdf` 但内容是 PNG 时返回 `extension_content_mismatch`，不会进入 OCR。批量中某一项失败，其余项仍能显示。

测试包括 `tests/test_server.py`、`test_upload_validation.py`、`test_benchmark_match_api.py` 与 `tests/frontend_report_smoke.mjs`；构建用 `npm ci && npm run build` 验证。

当前没有鉴权、速率限制、恶意文件沙箱、后台任务、真实浏览器 E2E 和生产 Web Server。原型阶段保留单个 server 文件是刻意取舍，暂不拆微服务。
