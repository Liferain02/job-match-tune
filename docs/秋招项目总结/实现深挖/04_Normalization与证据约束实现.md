# Normalization 与证据约束实现

## 设计目的

Normalization 不是“尽可能猜对”，而是把同义形式收敛、保留来源证据，并拒绝无依据补全。它位于模型输出和规则评分之间，决定后续硬条件是否可信。

## 调用链与关键函数

- 在线：`parse_json_output(text, context_text)` → `normalize_parsed_data`。
- JD：`_split_misplaced_fields`、`_merge_missing_responsibilities`、`canonicalize_job_direction`、`collect_skill_evidence`。
- Resume：`_normalize_resume_fields`、`_canonicalize_resume_direction`、`_normalize_resume_tags`。
- 离线：`normalize_jd.py::normalize_jd_row` 调用 `clean_text`、section/字段规则并写数据库或 JSONL。

输入参数必须包含生成文本和原始上下文。没有 `context_text` 时，只能做格式归一，不能证明技能来自要求段。

## 数据变化示例

原始模型可能输出 `{"必备技能":["C + +","Kafka"]}`。若原文要求段只有 `C + +`、Kafka 仅在职责段，则 canonicalization 把前者收敛为 `C++`，证据层过滤 Kafka，公开结果为 `必备技能=["C++"]`；内部证据仍记录 Kafka 的 responsibility 来源，便于审计。

返回值是 `{ok, data, raw_json, error}`。`data` 被 API、规则引擎和评测器消费；`raw_json` 用于定位模型问题，不能越过 normalization 直接作为产品事实。

## 失败策略、测试和边界

- JSON 修复只处理围栏、前后杂质和常见格式问题，不发明字段内容。
- Pydantic/字段类型异常会变成可见错误；列表去重保持首次出现顺序。
- 测试：`tests/test_postprocess_json.py`、`tests/test_skill_canonicalization.py`、`tests/test_jd_skill_evidence.py`。
- 边界：section 识别依赖明确标题或句式；OCR 恢复限定在已知词表；模型正确输出也可能被更保守的证据边界过滤，因此要分别报告 Raw 与 Normalized 指标。
