# 从一个 Case 贯穿整个系统

## 为什么选这个 Case

使用冻结 Gold V1 中预测正确的 `match_eval_002`。它不是已知失败的 001、008、candidate_007，能展示正常链路；但因为已被查看，只是实现说明和历史回归，不是新的盲测证据。

## 原始输入

JD：后端开发，职责是交易链路服务开发治理；任职要求明确 Java、Spring Boot、MySQL、Redis、Kafka，本科及以上、三年以上。

Resume：目标后端开发，软件工程硕士；技能 Java、Spring Boot、MySQL、Redis、Docker；有订单中心和限流治理项目，但没有明确三年以上。

`build_prompt` 给 JD/Resume 分别加结构化 JSON 指令。历史模型原始输出由 `run_match_eval.py` 保存在 `outputs/eval_reports/match_gold_final_predictions.jsonl`；后处理再带原文调用 `parse_json_output`。

## Normalized 与规则结果

JD 要求证据确认五个必备技能；Resume 规范技能保留五项。集合计算得到：

```json
{
  "命中技能": ["Java", "MySQL", "Redis", "Spring Boot"],
  "缺失技能": ["Kafka"],
  "岗位方向匹配": true,
  "学历匹配": true,
  "经验匹配": false
}
```

方向是 exact；学历满足；简历没写三年以上，所以经验 false。冻结历史策略得到 66 分和“较匹配”。新策略仍把该数字标记为 heuristic，并附分项/阈值，不把 66 解释成 66% 录用率。

## 解释、返回与消费

规则 JSON 被放进 `match_prompt`。历史解释的优势是方向、学历和四项技能；短板是年限与 Kafka；建议补 Kafka 并明确相关项目。`evaluate_explanation` 检查这些陈述没有把缺失技能写成优势，也没有把经验 false 说成满足。

API 将 `jd_parse`、`resume_parse`、`rule_result`、`analysis` 返回给 `ResultPanel`；Markdown 导出保留同样结构和原始 JSON。若任一解析失败，就停止规则和解释，不生成看似完整的结论。

## 测试和边界

该 Case 覆盖正常路径，但不覆盖 OCR、方向兼容、职责/要求混淆和文件损坏；这些由 Candidate V2 与专项单元测试承担。它也不能证明建议能提高投递成功率。
