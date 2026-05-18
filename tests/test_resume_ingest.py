from pathlib import Path
from types import SimpleNamespace
import sys

from jobmatch_tune.resume.ingest import (
    detect_source_type,
    extract_pdf_payload,
    ingest_resume,
    normalize_resume_text,
    split_resume_sections,
)


def test_detect_source_type():
    assert detect_source_type(Path("resume.txt")) == "text"
    assert detect_source_type(Path("resume.docx")) == "docx"
    assert detect_source_type(Path("resume.pdf")) == "pdf"
    assert detect_source_type(Path("resume.png")) == "image"


def test_normalize_resume_text():
    text = "姓名：张三\r\n\r\n 目标岗位：AI 应用开发 \r\n\t核心技能：Python  RAG "
    normalized = normalize_resume_text(text)
    assert normalized == "姓名：张三\n\n目标岗位：AI 应用开发\n核心技能：Python RAG"


def test_split_resume_sections():
    text = (
        "姓名：张三\n"
        "目标岗位：AI 应用开发\n"
        "教育背景\n"
        "本科，计算机科学与技术\n"
        "核心技能\n"
        "Python\n"
        "项目经历\n"
        "知识库问答系统\n"
    )
    sections = split_resume_sections(text)
    assert "header" in sections
    assert "教育背景" not in sections["education"]
    assert "本科，计算机科学与技术" in sections["education"]
    assert "Python" in sections["skills"]


def test_ingest_text_resume(tmp_path: Path):
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text(
        "姓名：张三\n目标岗位：后端开发工程师\n教育背景\n本科，软件工程\n核心技能\nPython\n项目经历\n服务平台开发",
        encoding="utf-8",
    )
    row = ingest_resume(resume_path)
    assert row["parse_ok"] is True
    assert row["source_type"] == "text"
    assert "目标岗位：后端开发工程师" in row["clean_text"]
    assert row["sections"]["education"] == "本科，软件工程"


def test_ingest_image_resume_marks_needs_ocr(tmp_path: Path):
    image_path = tmp_path / "resume.png"
    image_path.write_bytes(b"fake")
    row = ingest_resume(image_path)
    assert row["parse_ok"] is False
    assert row["needs_ocr"] is True
    assert row["parse_error"] == "image_resume_requires_ocr"


def test_ingest_image_resume_from_sidecar_ocr(tmp_path: Path):
    image_path = tmp_path / "resume.png"
    image_path.write_bytes(b"fake")
    sidecar = tmp_path / "resume.ocr.txt"
    sidecar.write_text("目标岗位：AI应用开发工程师\n核心技能\nPython\n项目经历\n问答系统", encoding="utf-8")
    row = ingest_resume(image_path)
    assert row["parse_ok"] is True
    assert row["ocr_used"] is True
    assert row["ocr_source"] == "sidecar"
    assert row["needs_ocr"] is False


def test_ingest_pdf_falls_back_to_sidecar_ocr(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(b"%PDF")
    sidecar = tmp_path / "resume.ocr.txt"
    sidecar.write_text("目标岗位：后端开发工程师\n教育背景\n本科", encoding="utf-8")

    monkeypatch.setattr(
        "jobmatch_tune.resume.ingest.extract_pdf_payload",
        lambda path: {
            "raw_text": "",
            "page_count": 1,
            "total_chars": 0,
            "avg_chars_per_page": 0.0,
            "pdf_kind": "scanned_pdf",
        },
    )
    row = ingest_resume(pdf_path)
    assert row["parse_ok"] is True
    assert row["ocr_used"] is True
    assert row["ocr_source"] == "sidecar"
    assert row["pdf_kind"] == "scanned_pdf"


def test_ingest_scanned_pdf_without_ocr_marks_needs_ocr(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(b"%PDF")
    monkeypatch.setattr(
        "jobmatch_tune.resume.ingest.extract_pdf_payload",
        lambda path: {
            "raw_text": "",
            "page_count": 2,
            "total_chars": 0,
            "avg_chars_per_page": 0.0,
            "pdf_kind": "scanned_pdf",
        },
    )
    row = ingest_resume(pdf_path)
    assert row["parse_ok"] is False
    assert row["needs_ocr"] is True
    assert row["parse_error"] == "scanned_pdf_requires_ocr"


def test_ingest_weak_text_pdf_without_ocr_marks_review(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(b"%PDF")
    monkeypatch.setattr(
        "jobmatch_tune.resume.ingest.extract_pdf_payload",
        lambda path: {
            "raw_text": "姓名 张三",
            "page_count": 2,
            "total_chars": 10,
            "avg_chars_per_page": 5.0,
            "pdf_kind": "weak_text_pdf",
        },
    )
    row = ingest_resume(pdf_path)
    assert row["parse_ok"] is False
    assert row["needs_ocr"] is True
    assert row["parse_error"] == "weak_text_pdf_requires_ocr_review"


def test_extract_pdf_payload_classifies_text_density(monkeypatch, tmp_path: Path):
    class FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, path: str):
            self.pages = [FakePage("姓名：张三\n教育背景\n本科"), FakePage("项目经历\n平台开发经验" * 10)]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))
    payload = extract_pdf_payload(tmp_path / "resume.pdf")
    assert payload["pdf_kind"] == "text_pdf"
    assert payload["page_count"] == 2
    assert payload["total_chars"] > 80
