from pathlib import Path

from jobmatch_tune.eval.validate_resume_sample import build_resume_sample_report


def test_user_resume_pdf_sample_is_ready_for_file_parse():
    sample_path = Path("docs/个人简历-李福润.pdf")
    report = build_resume_sample_report(path=sample_path, min_text_chars=2000)
    assert report["ready_for_resume_file_parse"] is True
    assert report["source_type"] == "pdf"
    assert report["pdf_kind"] == "text_pdf"
    assert report["extraction_method"] == "pypdf"
    assert report["text_char_count"] >= 2000
    assert set(report["required_sections"]).issubset(report["sections_found"])
    assert "clean_text" not in report
    assert "raw_text" not in report
