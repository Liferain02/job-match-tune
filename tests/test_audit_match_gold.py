from jobmatch_tune.eval.audit_match_gold import DIFFICULTY_TAGS, audit_match_gold


def _gold_row(row_id: str = "gold-1") -> dict:
    return {
        "id": row_id,
        "source_group": row_id,
        "jd_text": "岗位名称：后端开发工程师\n要求：熟悉 Python。",
        "resume_text": "目标岗位：后端开发\n项目：使用 Python 开发接口。",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python"],
            "缺失技能": [],
        },
        "meta": {
            "annotation_status": "human_verified",
            "annotator_id": "reviewer-a",
            "reviewed_at": "2026-08-11",
            "rationale": "项目中有直接 Python 开发证据。",
            "difficulty_tags": sorted(DIFFICULTY_TAGS),
        },
    }


def test_audit_match_gold_accepts_verified_independent_rows():
    report = audit_match_gold([_gold_row()], [], min_rows=1)

    assert report["gold_ready"] is True
    assert report["regression_ready"] is True
    assert report["blind_ready"] is False
    assert report["decision_ready"] is False
    assert report["ranking_ready"] is False
    assert "evaluation_role_not_blind_holdout" in report["blind_blockers"]
    assert report["training_overlap_count"] == 0


def test_audit_match_gold_accepts_declared_unseen_blind_rows():
    row = _gold_row()
    row["meta"]["evaluation_role"] = "blind_holdout"
    row["meta"]["inspection_status"] = "unseen"

    report = audit_match_gold([row], [], min_rows=1)

    assert report["blind_ready"] is True
    assert report["blind_blockers"] == []


def test_audit_match_gold_rejects_training_overlap():
    row = _gold_row()
    report = audit_match_gold([row], [dict(row)], min_rows=1)

    assert report["gold_ready"] is False
    assert report["training_overlap_ids"] == ["gold-1"]
    assert report["training_jd_overlap_ids"] == ["gold-1"]
    assert report["training_resume_overlap_ids"] == ["gold-1"]


def test_audit_match_gold_rejects_constituent_entity_overlap():
    row = _gold_row()
    same_jd = dict(row)
    same_jd["resume_text"] = "目标岗位：测试开发\n项目：使用 Pytest。"
    same_resume = dict(row)
    same_resume["jd_text"] = "岗位名称：测试开发工程师\n要求：熟悉 Pytest。"

    report = audit_match_gold([row], [same_jd, same_resume], min_rows=1)

    assert report["gold_ready"] is False
    assert report["training_overlap_count"] == 0
    assert report["training_jd_overlap_ids"] == ["gold-1"]
    assert report["training_resume_overlap_ids"] == ["gold-1"]


def test_audit_match_gold_rejects_jd_and_resume_task_training_overlap():
    row = _gold_row()
    report = audit_match_gold(
        [row],
        [],
        jd_training_rows=[{"raw_text": row["jd_text"]}],
        resume_training_rows=[{"text": row["resume_text"]}],
        min_rows=1,
    )

    assert report["gold_ready"] is False
    assert report["jd_task_training_overlap_ids"] == ["gold-1"]
    assert report["resume_task_training_overlap_ids"] == ["gold-1"]


def test_audit_match_gold_rejects_unverified_legacy_seed():
    row = _gold_row()
    row["meta"] = {"annotation_status": "legacy_unverified", "difficulty_tags": ["技能同义词"]}

    report = audit_match_gold([row], [], min_rows=1)

    assert report["gold_ready"] is False
    assert report["human_verified_rows"] == 0
