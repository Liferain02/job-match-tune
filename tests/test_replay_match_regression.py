from jobmatch_tune.eval.replay_match_regression import replay_current_rules


def test_replay_current_rules_reuses_generation_and_updates_all_rule_views() -> None:
    rows = [
        {
            "id": "case-1",
            "jd_text": "岗位：后端开发\n任职要求：熟悉 Java。",
            "resume_text": "目标岗位：后端开发\n核心技能：Java",
            "jd_ok": True,
            "resume_ok": True,
            "analysis_ok": True,
            "label": {
                "匹配等级": "较匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Java"],
                "缺失技能": [],
            },
            "normalized": {
                "structured_result_available": True,
                "jd_parse": {
                    "岗位方向": "后端开发",
                    "必备技能": ["Java"],
                    "学历要求": "不限",
                    "经验要求": "不限",
                },
                "resume_parse": {"目标岗位": "后端开发", "核心技能": ["Java"]},
                "rule_result": {"stale": True},
            },
            "product_final": {"analysis_ok": True, "analysis": {}, "rule_result": {"stale": True}},
        }
    ]

    replayed = replay_current_rules(rows)

    assert replayed[0]["rule_result"]["命中技能"] == ["Java"]
    assert replayed[0]["normalized"]["rule_result"] == replayed[0]["rule_result"]
    assert replayed[0]["product_final"]["rule_result"] == replayed[0]["rule_result"]
    assert replayed[0]["regression_replay"] == {
        "current_rules_recomputed": True,
        "generation_reused": True,
        "evaluation_context": "historical_gold_v1_regression",
    }
