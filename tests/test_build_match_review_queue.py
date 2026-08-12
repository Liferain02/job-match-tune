from jobmatch_tune.eval.build_match_review_queue import build_match_review_queue


def _gold() -> dict:
    return {
        "id": "case-1",
        "jd_text": "后端工程师，本科及以上，三年以上经验，熟悉 Python。",
        "resume_text": "本科，掌握 Python 和 Redis，2023年毕业。",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python"],
            "缺失技能": [],
        },
        "meta": {"annotation_status": "needs_human_review", "difficulty_tags": ["单项硬门槛不满足"]},
    }


def _prediction() -> dict:
    return {
        "id": "case-1",
        "product_final": {
            "rule_result": {
                "匹配等级": "较匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": False,
                "命中技能": ["Python", "Redis"],
                "缺失技能": [],
            }
        },
    }


def test_review_queue_prioritizes_explicit_experience_label_conflict() -> None:
    report = build_match_review_queue([_gold()], [_prediction()])
    item = report["items"][0]

    assert item["evidence"]["jd_required_years"] == 3
    assert item["evidence"]["resume_explicit_years"] == 0
    assert "经验标签与原文明示年限矛盾" in item["review_reasons"]
    assert "标签未覆盖评分器找到的技能证据" in item["review_reasons"]
    assert report["review_order"] == ["case-1"]


def test_review_queue_does_not_auto_promote_or_rewrite_label() -> None:
    gold = _gold()
    original_label = dict(gold["label"])

    report = build_match_review_queue([gold], [])

    assert gold["label"] == original_label
    assert report["items"][0]["annotation_status"] == "needs_human_review"
    assert "缺少对应预测结果" in report["items"][0]["review_reasons"]
