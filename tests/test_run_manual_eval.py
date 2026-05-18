from jobmatch_tune.eval.run_manual_eval import evaluate_predictions


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
