from jobmatch_tune.eval.run_match_eval import build_report, evaluate_rows


def test_evaluate_rows_basic():
    rows = [
        {
            "id": "match1",
            "source_type": "text",
            "jd_ok": True,
            "resume_ok": True,
            "analysis_ok": True,
            "rule_result": {
                "匹配等级": "高匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Python", "RAG"],
                "缺失技能": [],
            },
            "label": {
                "匹配等级": "高匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Python", "RAG"],
                "缺失技能": [],
            },
        }
    ]
    report = evaluate_rows(rows)
    assert report["jd_resume_parse_success_rate"] == 1.0
    assert report["analysis_json_valid_rate"] == 1.0
    assert report["field_metrics"]["匹配等级"]["exact_match"] == 1.0


def test_build_report_groups_by_source():
    rows = [
        {
            "id": "a",
            "source_type": "text",
            "jd_ok": True,
            "resume_ok": True,
            "analysis_ok": True,
            "rule_result": {
                "匹配等级": "高匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Python"],
                "缺失技能": [],
            },
            "label": {
                "匹配等级": "高匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Python"],
                "缺失技能": [],
            },
        },
        {
            "id": "b",
            "source_type": "ocr_like",
            "jd_ok": True,
            "resume_ok": True,
            "analysis_ok": False,
            "rule_result": {
                "匹配等级": "较匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": False,
                "命中技能": ["Python"],
                "缺失技能": ["MySQL"],
            },
            "label": {
                "匹配等级": "较匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": False,
                "命中技能": ["Python"],
                "缺失技能": ["MySQL"],
            },
        },
    ]
    report = build_report(rows)
    assert report["overall"]["num_samples"] == 2
    assert "text" in report["by_source_type"]
    assert "ocr_like" in report["by_source_type"]
