from jobmatch_tune.eval.run_resume_pipeline_eval import build_report
from jobmatch_tune.resume.normalize import normalize_resume_eval_row


def _row(source_type: str):
    return {
        "id": f"resume_{source_type}",
        "task": "resume_parse",
        "source_type": source_type,
        "text": "姓名：张三\n目标岗位：AI 应用开发工程师\n教育背景：本科，计算机科学与技术\n核心技能：Python、RAG",
        "label": {
            "目标岗位": "AI应用开发",
            "教育背景": ["本科，计算机科学与技术"],
            "核心技能": ["Python", "RAG"],
            "实习经历": [],
            "项目经历": [],
            "优势标签": [],
        },
    }


def test_normalize_resume_eval_row_builds_normalized_text():
    row = normalize_resume_eval_row(_row("text"))
    assert row["normalized_text"]
    assert "教育背景：" in row["normalized_text"]
    assert row["source_type"] == "text"


def test_build_report_groups_by_source_type():
    predictions = [
        {
            "id": "a",
            "task": "resume_parse",
            "source_type": "text",
            "ok": True,
            "parsed": {
                "目标岗位": "AI应用开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Python", "RAG"],
                "实习经历": [],
                "项目经历": [],
                "优势标签": [],
            },
            "label": {
                "目标岗位": "AI应用开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Python", "RAG"],
                "实习经历": [],
                "项目经历": [],
                "优势标签": [],
            },
        },
        {
            "id": "b",
            "task": "resume_parse",
            "source_type": "ocr_like",
            "ok": True,
            "parsed": {
                "目标岗位": "AI应用开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Python", "RAG"],
                "实习经历": [],
                "项目经历": [],
                "优势标签": [],
            },
            "label": {
                "目标岗位": "AI应用开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Python", "RAG"],
                "实习经历": [],
                "项目经历": [],
                "优势标签": [],
            },
        },
    ]
    report = build_report(predictions)
    assert report["overall"]["task"] == "resume_parse"
    assert "text" in report["by_source_type"]
    assert "ocr_like" in report["by_source_type"]
