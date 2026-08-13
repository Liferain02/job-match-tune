# 技能 Canonicalization 与 OCR 实现

## 目标与约束

技能名称包含大小写、别名、符号和 OCR 断字。实现目标是恢复已知技能，而不是用模糊匹配猜一个相似词。入口在 `preprocess/skill_canonicalization.py`，词表来自 `configs/label_schema.yaml`。

## 函数与数据流

- `_vocabulary` 汇总 canonical 技能及显式 aliases。
- `_lookup_maps` 建精确 key 和去空白 compact key；compact key 只有唯一映射时才启用。
- `canonicalize_skill_name` 先精确匹配，再尝试唯一 compact form。
- `contains_skill_candidate` 生成带边界的正则；`_ocr_candidate_pattern` 只对合适的字母数字技能允许字符间 OCR 空白/连字符。
- `extract_known_skills` 在原文中查词表，`canonicalize_skill_list` 则处理模型给出的候选列表并去重。

例子：`Py thon` → `Python`、`Kubernet es` → `Kubernetes`、`C + +` → `C++`、`Node . js` → `Node.js`、`My SOL` → `MySQL`。反例 `pytesting`、普通单词 `go` 的子串不会被误认；不存在词表中的词不会凭编辑距离补出来。

## 调用方、返回和错误

`postprocess_json.py` 用它规范模型结果，`jd_skill_evidence.py` 用它寻找原文证据，`rule_engine.py` 用规范形式求交集。函数返回 canonical 字符串、列表或 `None`，无匹配时保留不了“猜测技能”。词表别名冲突会让 compact 映射失去唯一性，从而退回精确形式。

## 测试与限制

`tests/test_skill_canonicalization.py` 覆盖空格、符号、大小写、别名、重复、子串假阳性和歧义。当前无法恢复任意拼写错误，如完全未知的 `Kuberntes`；这是刻意边界。扩词必须基于独立错误样本并补正反例测试，不能为一条 Gold 特判。
