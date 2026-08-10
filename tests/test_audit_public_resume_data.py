from jobmatch_tune.eval.audit_public_resume_data import compute_report


def test_compute_public_resume_report():
    rows = [
        {
            "task": "resume_parse",
            "source_type": "public_text",
            "text": "姓名：张三\n目标岗位：后端开发",
            "label": {"目标岗位": "后端开发", "教育背景": ["本科"], "核心技能": ["Java"]},
            "meta": {"source_name": "faircv", "language": "zh"},
        },
        {
            "task": "resume_ner",
            "source_type": "public_text",
            "text": "张三",
            "tokens": ["张", "三"],
            "ner_tags": ["B-NAME", "I-NAME"],
            "meta": {"source_name": "resume_ner", "language": "zh"},
        },
    ]
    report = compute_report(rows)
    assert report["total_rows"] == 2
    assert report["task_distribution"][0][0] in {"resume_parse", "resume_ner"}
    assert report["resume_parse_label_coverage"]["target_job"] == 1.0
    assert report["resume_ner_tag_count"] == 2
    assert report["resume_ner_rows_with_sensitive_entities"] == 1
    assert report["resume_ner_training_ready"] is False
