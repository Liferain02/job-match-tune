# Gold 评测与三层 Evaluator 实现

## 为什么分三层

只报最终准确率会把模型生成、后处理和规则收益混在一起。`eval/run_match_eval.py::build_report` 同时计算：Raw Model Derived（原始生成 JSON 直接推规则）、Normalized（规范化结构推规则）、Product Final（规则结果加解释）。差值能回答改善来自模型还是工程约束。

## 输入、函数与指标

输入 JSONL 含 id、JD、Resume、人工 label、source_type/source_group/meta。`run_predictions` 依次生成两类解析和解释，并保留 raw/normalized/product 三份结果；`evaluate_rows` 计算技能 precision/recall/F1，匹配等级、方向、学历、经验 exact match；`build_error_analysis` 分类漏召回、误识别、方向、等级和解释矛盾。

`explanation_grounding.py::evaluate_explanation` 又拆成结构一致性和证据 grounding。建议是否真的提高求职结果被标为 `not_evaluated/unsupported_by_current_data`，不会用“解释没矛盾”冒充建议有效。

## Gold V1 的解释边界

V1 共 25 条人工复核样本，且有实体/内容/来源等独立性审计。历史 Product Final 的匹配等级和方向 exact match 为 0.96。因为本轮已经阅读过这些错误，之后重放必须标记 `historical_gold_v1_regression` 和 `REGRESSION AFTER INSPECTION`，只能防退化，不能再称 blind generalization。

2026-08-13 的保存预测重放由 `eval/replay_match_regression.py::replay_current_rules` 完成：不加载模型，不重新生成，只用冻结 normalized JD/Resume 重算当前规则。六项结构指标为 1.0；旧解释结构一致率 1.0，但 evidence-grounding 仅 0.92，其中两条仍携带修复前的 Agent/Pytest 陈述。新报告写入 `outputs/eval_reports/match_gold_v1_semantic_boundary_regression_20260813.json`，历史文件未覆盖。

输出报告被产品回归比较器、文档和人工审查消费；predictions 保留逐样本证据。旧报告和 Gold 哈希冻结，任何新回归写新文件。

## 失败、测试与限制

未人工复核数据自动降级为 `provisional_candidate_diagnosis`；结构不可用计入失败。测试包括 `tests/test_run_match_eval.py`、`test_explanation_grounding.py`、`test_audit_match_gold.py`。25 条只适合稳定回归，不代表行业全分布；Gold V2 必须来自独立新样本和盲标注。
