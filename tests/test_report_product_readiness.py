from jobmatch_tune.eval.report_product_readiness import build_product_readiness_report


def _jd_report(direction_score: float = 0.96):
    return {
        "json_valid_rate": 0.98,
        "field_metrics": {
            "核心职责": {"f1": 1.0},
            "必备技能": {"f1": 1.0},
            "加分项": {"f1": 1.0},
            "岗位方向": {"exact_match": direction_score},
            "经验要求": {"exact_match": 1.0},
            "学历要求": {"exact_match": 1.0},
        },
    }


def _resume_report(strength_score: float = 0.99):
    return {
        "overall": {
            "json_valid_rate": 1.0,
            "field_metrics": {
                "教育背景": {"f1": 1.0},
                "核心技能": {"f1": 1.0},
                "实习经历": {"f1": 1.0},
                "项目经历": {"f1": 1.0},
                "优势标签": {"f1": strength_score},
                "目标岗位": {"exact_match": 1.0},
            },
        }
    }


def _match_report(level_score: float = 0.78):
    return {
        "overall": {
            "jd_resume_parse_success_rate": 1.0,
            "analysis_json_valid_rate": 1.0,
            "field_metrics": {
                "命中技能": {"f1": 0.86},
                "缺失技能": {"f1": 0.90},
                "匹配等级": {"exact_match": level_score},
                "岗位方向匹配": {"exact_match": 0.91},
                "学历匹配": {"exact_match": 1.0},
                "经验匹配": {"exact_match": 1.0},
            },
        }
    }


def test_product_readiness_accepts_passing_reports():
    report = build_product_readiness_report(
        jd_report=_jd_report(),
        resume_report=_resume_report(),
        match_report=_match_report(),
    )
    assert report["ready_for_user"] is True
    assert report["num_failed_checks"] == 0


def test_product_readiness_reports_failed_checks():
    report = build_product_readiness_report(
        jd_report=_jd_report(direction_score=0.80),
        resume_report=_resume_report(strength_score=0.70),
        match_report=_match_report(level_score=0.50),
    )
    failed_names = {item["name"] for item in report["not_ready_checks"]}
    assert report["ready_for_user"] is False
    assert "jd_direction_exact_match" in failed_names
    assert "resume_strength_f1" in failed_names
    assert "match_level_exact_match" in failed_names
