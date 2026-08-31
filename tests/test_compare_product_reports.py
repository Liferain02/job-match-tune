from jobmatch_tune.eval.compare_product_reports import build_product_regression_report


def _jd_report(direction_score: float = 0.96):
    return {
        "evaluation_validity": "blind_holdout",
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
        "evaluation_validity": "blind_holdout",
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


def _match_report(hit_skill_score: float = 0.86, level_score: float = 0.80):
    return {
        "evaluation_validity": "blind_holdout",
        "dataset_profile": {"decision_evaluation_ready": True},
        "overall": {
            "jd_resume_parse_success_rate": 1.0,
            "analysis_json_valid_rate": 1.0,
            "field_metrics": {
                "命中技能": {"f1": hit_skill_score},
                "缺失技能": {"f1": 0.90},
                "匹配等级": {"exact_match": level_score},
                "岗位方向匹配": {"exact_match": 0.91},
                "学历匹配": {"exact_match": 1.0},
                "经验匹配": {"exact_match": 1.0},
            },
            "decision_metrics": {"macro_f1": level_score},
        }
    }


def test_product_regression_allows_small_metric_drift():
    report = build_product_regression_report(
        candidate_jd_report=_jd_report(direction_score=0.955),
        candidate_resume_report=_resume_report(strength_score=0.986),
        candidate_match_report=_match_report(hit_skill_score=0.856, level_score=0.796),
        baseline_jd_report=_jd_report(direction_score=0.96),
        baseline_resume_report=_resume_report(strength_score=0.99),
        baseline_match_report=_match_report(hit_skill_score=0.86, level_score=0.80),
        max_regression=0.005,
    )
    assert report["candidate_ready_for_user"] is True
    assert report["no_product_regression"] is True
    assert report["ready_to_promote"] is True


def test_product_regression_blocks_metric_drop_even_when_candidate_is_ready():
    report = build_product_regression_report(
        candidate_jd_report=_jd_report(direction_score=0.955),
        candidate_resume_report=_resume_report(strength_score=0.99),
        candidate_match_report=_match_report(hit_skill_score=0.850, level_score=0.78),
        baseline_jd_report=_jd_report(direction_score=0.98),
        baseline_resume_report=_resume_report(strength_score=0.99),
        baseline_match_report=_match_report(hit_skill_score=0.90, level_score=0.82),
        max_regression=0.005,
    )
    failed_names = {item["name"] for item in report["failed_regressions"]}
    assert report["candidate_ready_for_user"] is True
    assert report["no_product_regression"] is False
    assert report["ready_to_promote"] is False
    assert "jd_direction_exact_match" in failed_names
    assert "match_hit_skill_f1" in failed_names
    assert "match_level_exact_match" in failed_names
