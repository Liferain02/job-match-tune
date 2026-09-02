import pytest

from jobmatch_tune.eval.run_manual_eval import align_saved_predictions, evaluate_predictions


def test_evaluate_predictions_for_jd_parse():
    rows = [
        {
            "id": "jd1",
            "task": "jd_parse",
            "ok": True,
            "parsed": {
                "岗位方向": "后端开发",
                "核心职责": ["开发接口"],
                "必备技能": ["Python"],
                "加分项": [],
                "经验要求": "三年以上工作经验",
                "学历要求": "本科及以上",
            },
            "label": {
                "岗位方向": "后端开发",
                "核心职责": ["开发接口"],
                "必备技能": ["Python"],
                "加分项": [],
                "经验要求": "三年以上工作经验",
                "学历要求": "本科及以上",
            },
        }
    ]
    report = evaluate_predictions(rows)
    assert report["task"] == "jd_parse"
    assert report["json_valid_rate"] == 1.0
    assert report["field_metrics"]["岗位方向"]["exact_match"] == 1.0
    assert report["confidence_intervals"]["json_valid_rate"]["lower"] == 1.0
    assert report["field_confidence_intervals"]["核心职责"]["metric"] == "row_macro_f1"
    assert report["slice_metrics"]["target_role"]["后端开发"]["num_samples"] == 1
    assert report["slice_metrics"]["target_role"]["后端开发"]["small_slice_warning"] is True


def test_evaluate_predictions_for_resume_parse():
    rows = [
        {
            "id": "resume1",
            "task": "resume_parse",
            "ok": True,
            "parsed": {
                "目标岗位": "AI应用开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Python", "RAG"],
                "实习经历": ["在平台团队实习"],
                "项目经历": ["做过知识库问答系统"],
                "优势标签": ["LLM应用落地"],
            },
            "label": {
                "目标岗位": "AI应用开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Python", "RAG"],
                "实习经历": ["在平台团队实习"],
                "项目经历": ["做过知识库问答系统"],
                "优势标签": ["LLM应用落地"],
            },
        }
    ]
    report = evaluate_predictions(rows)
    assert report["task"] == "resume_parse"
    assert report["json_valid_rate"] == 1.0
    assert report["field_metrics"]["目标岗位"]["exact_match"] == 1.0
    assert report["field_metrics"]["核心技能"]["f1"] == 1.0


def test_invalid_json_counts_as_end_to_end_field_failure():
    rows = [
        {
            "id": "jd-invalid",
            "task": "jd_parse",
            "ok": False,
            "parsed": None,
            "label": {
                "岗位方向": "后端开发",
                "核心职责": ["开发接口"],
                "必备技能": ["Python"],
                "加分项": [],
                "经验要求": "不限",
                "学历要求": "本科",
            },
        }
    ]

    report = evaluate_predictions(rows)

    assert report["field_metrics"]["必备技能"]["f1"] == 0.0
    assert report["field_metrics"]["加分项"]["f1"] == 0.0
    assert report["valid_json_only_field_metrics"]["必备技能"]["num_rows"] == 0
    assert report["complete_row_exact_match_rate"] == 0.0


def test_blind_parse_eval_requires_human_verified_unseen_metadata():
    row = {
        "id": "resume-blind",
        "task": "resume_parse",
        "ok": True,
        "parsed": {},
        "label": {},
        "meta": {
            "annotation_status": "human_verified",
            "evaluation_role": "blind_holdout",
            "inspection_status": "unseen",
        },
    }

    report = evaluate_predictions([row], evaluation_context="blind_holdout")

    assert report["evaluation_validity"] == "blind_holdout"


def test_replay_predictions_requires_exact_frozen_inputs():
    dataset = [
        {
            "id": "jd1",
            "task": "jd_parse",
            "text": "岗位名称：后端开发",
            "label": {"岗位方向": "后端开发"},
        }
    ]
    predictions = [
        {
            "id": "jd1",
            "task": "jd_parse",
            "text": "岗位名称：前端开发",
            "ok": True,
            "parsed": {"岗位方向": "前端开发"},
        }
    ]

    with pytest.raises(ValueError, match="saved prediction input differs"):
        align_saved_predictions(dataset, predictions)
