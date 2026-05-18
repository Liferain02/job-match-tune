from jobmatch_tune.resume.normalize import normalize_ingest_row, render_normalized_resume


def _row():
    return {
        "id": "resume_001",
        "file_name": "resume.pdf",
        "file_path": "resume.pdf",
        "source_type": "pdf",
        "pdf_kind": "text_pdf",
        "ocr_used": False,
        "ocr_source": "",
        "extraction_method": "pypdf",
        "parse_ok": True,
        "needs_ocr": False,
        "clean_text": "目标岗位：后端开发工程师",
        "sections": {
            "header": "目标岗位：后端开发工程师",
            "education": "本科，软件工程",
            "skills": "Python\nMySQL",
            "projects": "服务平台开发",
        },
    }


def test_render_normalized_resume():
    text = render_normalized_resume(_row())
    assert "目标岗位：后端开发工程师" in text
    assert "教育背景：" in text
    assert "核心技能：" in text
    assert "项目经历：" in text


def test_normalize_ingest_row_keeps_metadata():
    row = normalize_ingest_row(_row())
    assert row["id"] == "resume_001"
    assert row["source_type"] == "pdf"
    assert row["pdf_kind"] == "text_pdf"
    assert row["normalized_text"]
