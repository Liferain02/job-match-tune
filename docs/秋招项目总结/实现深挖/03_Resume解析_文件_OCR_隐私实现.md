# Resume 解析、文件、OCR、隐私实现

## 为什么拆成四层

简历输入既可能是文本，也可能是 PDF、DOCX 或图片。安全的顺序必须是“校验内容 → 提取文本 → OCR 兜底 → 去标识/结构化”，否则伪装扩展名或损坏文件会先占用高成本解析资源。

## 入口、参数和转换

- HTTP 入口：`server.py::resume_file_parse` 与 `match_files`，参数含 `UploadFile`、`max_new_tokens`。
- `upload_validation.py::validate_upload_content(file_name, content)` 返回 `UploadDescriptor(extension, detected_type)`。
- PDF 先检查 `%PDF` 并由 pypdf 验证页结构；DOCX 必须是 ZIP 且含 `[Content_Types].xml` 与 `word/document.xml`；图片用 Pillow 真正 decode 并核对格式；文本必须 UTF-8、无 NUL 和异常控制字符。
- `resume/ingest.py` 按类型抽取文本，扫描 PDF 无文本时由 `resume/ocr.py` 走 PyMuPDF 渲染和 RapidOCR；`server.py::parse_uploaded_resume_bytes` 给在线流程复用。
- `resume/privacy.py` 与 `preprocess/clean_text.py::mask_private_info` 处理手机号、邮箱等标识信息；离线流程由 `scripts/data/resume_ingest.sh`、`resume_ocr_sidecar.sh`、`resume_privacy_audit.sh` 驱动。
- 文本最终进入 `resume_parse_prompt`，由 `postprocess_json.py::_normalize_resume_fields` 返回目标岗位、教育、技能、实习、项目、优势标签。

## 返回、消费方和异常

提取接口返回文本、文件元信息与执行信息；解析接口再返回结构 JSON。匹配流程消费结构字段，而不会把原始二进制传给规则引擎。

错误码区分超限、类型不支持、扩展名与内容不符、文件损坏；校验失败时不会调用 pypdf 深解析、OCR 或模型。OCR 不可用/无结果要显式报告，不能把空字符串视为一份有效简历。

## 测试和限制

`tests/test_upload_validation.py` 与 `tests/test_server.py` 覆盖 PDF/DOCX/图片/文本及伪装文件；`tests/test_resume_ingest.py`、`test_resume_ocr.py`、`test_resume_privacy.py` 覆盖提取、OCR 和隐私。

当前不是恶意文件沙箱：没有杀毒、PDF JavaScript 深度检测或租户隔离。去标识化是数据准入必要条件，但不能替代许可核验；原文件不应进入 Git 或训练集。
