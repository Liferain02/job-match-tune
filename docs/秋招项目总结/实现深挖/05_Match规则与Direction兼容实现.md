# Match 规则与 Direction 兼容实现

## 为什么规则决定结论

方向、学历、经验和技能集合是可核查的硬条件。`match/rule_engine.py::compute_match_rule_result(jd_data, resume_data, jd_text, resume_text)` 负责确定性结论；LLM 只根据该结果写解释，不能改分。

## 计算过程

1. `_skill_lists` 从已经过要求证据过滤的 JD 技能与 Resume 技能求交集/差集。
2. `_extract_required_years` 与 `_extract_years` 解析显式年限；简历没写年限时不能因为项目多就自动满足。
3. 学历映射为可比较等级；JD 未要求时视为不构成阻塞。
4. `evaluate_direction_compatibility` 返回 `exact`、`compatible` 或 `mismatch` 及证据。完全一致直接 exact；兼容必须由有限 taxonomy、职责/上下文信号和至少两个共享技能共同支持；“算法”和“后端”不会默认相容。
5. `scoring.py::compute_score_breakdown` 使用冻结策略：方向 20、技能 45、学历 10、经验 15、项目 10；阈值 85/65/45 映射高度匹配/较匹配/一般/不匹配。

返回同时包含 `岗位方向关系`、`岗位方向证据`、`技能证据`、`匹配分项` 和 `评分策略`。消费方是解释 Prompt、API 和三层评测器。

## 示例与异常

JD 为后端，Resume 为后端且命中 Java/MySQL，方向 exact。若 Resume 写“算法工程师”，仅命中 Python，并无平台职责证据，则 mismatch；不能用宽泛“都属于技术岗”加满方向分。

策略对象 `MatchScoringPolicy` 在初始化时校验权重总和与阈值顺序，配置错误会立即失败。该分数的 `calibration_status=heuristic`：它是可解释排序指标，不是录用概率，也没有置信区间。

## 测试与限制

`tests/test_rule_engine.py`、`tests/test_scoring.py`、`tests/test_direction_compatibility.py` 覆盖集合、年限、学历、阈值、exact/compatible/mismatch 和反例。方向 taxonomy 刻意较小；项目命中仍依赖技能文本共现；没有真实招聘结果前不拟合权重。
