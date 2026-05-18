from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

from jobmatch_tune.resume.ocr import find_sidecar_ocr_text
from jobmatch_tune.utils.io import write_jsonl


TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PDF_MIN_TOTAL_CHARS = 80
PDF_MIN_AVG_CHARS_PER_PAGE = 40


def detect_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix == ".docx":
        return "docx"
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def normalize_resume_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    compact_lines: list[str] = []
    for line in lines:
        if not line:
            if compact_lines and compact_lines[-1] != "":
                compact_lines.append("")
            continue
        compact_lines.append(line)
    normalized = "\n".join(compact_lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def split_resume_sections(text: str) -> dict[str, str]:
    heading_patterns = {
        "education": r"(教育背景|教育经历|学历背景)",
        "skills": r"(专业技能|核心技能|技能清单|技能栈)",
        "internships": r"(实习经历|实践经历)",
        "projects": r"(项目经历|项目经验)",
        "work": r"(工作经历|工作经验)",
        "awards": r"(获奖经历|个人奖项|荣誉奖项|证书)",
        "profile": r"(自我评价|个人优势|个人总结)",
    }
    current = "header"
    sections: dict[str, list[str]] = {key: [] for key in ["header", *heading_patterns]}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = None
        for key, pattern in heading_patterns.items():
            if re.fullmatch(rf"{pattern}[：:]?", line, flags=re.I):
                matched = key
                break
        if matched:
            current = matched
            continue
        sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items() if value}


def extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_text_from_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("python-docx is required for .docx resume ingestion") from exc
    document = Document(path)
    lines = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(lines)


def extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pypdf is required for .pdf resume ingestion") from exc
    reader = PdfReader(str(path))
    lines: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            lines.append(text.strip())
    return "\n\n".join(lines)


def extract_pdf_payload(path: Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pypdf is required for .pdf resume ingestion") from exc
    reader = PdfReader(str(path))
    page_texts: list[str] = []
    char_counts: list[int] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        page_texts.append(text)
        char_counts.append(len(text))
    raw_text = "\n\n".join(text for text in page_texts if text)
    total_chars = sum(char_counts)
    page_count = len(char_counts)
    avg_chars = total_chars / page_count if page_count else 0.0
    if total_chars == 0:
        pdf_kind = "scanned_pdf"
    elif total_chars < PDF_MIN_TOTAL_CHARS or avg_chars < PDF_MIN_AVG_CHARS_PER_PAGE:
        pdf_kind = "weak_text_pdf"
    else:
        pdf_kind = "text_pdf"
    return {
        "raw_text": raw_text,
        "page_count": page_count,
        "total_chars": total_chars,
        "avg_chars_per_page": round(avg_chars, 1),
        "pdf_kind": pdf_kind,
    }


def ingest_resume(path: Path, ocr_dir: Path | None = None) -> dict[str, Any]:
    source_type = detect_source_type(path)
    row: dict[str, Any] = {
        "id": hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16],
        "file_name": path.name,
        "file_path": str(path),
        "source_type": source_type,
        "ocr_used": False,
        "needs_ocr": False,
        "parse_ok": False,
        "raw_text": "",
        "clean_text": "",
        "sections": {},
        "ocr_source": "",
        "extraction_method": source_type,
        "page_count": 1,
        "text_char_count": 0,
        "pdf_kind": "",
    }

    if source_type == "text":
        raw_text = extract_text_from_txt(path)
    elif source_type == "docx":
        raw_text = extract_text_from_docx(path)
    elif source_type == "pdf":
        pdf_payload = extract_pdf_payload(path)
        raw_text = pdf_payload["raw_text"]
        row["page_count"] = pdf_payload["page_count"]
        row["text_char_count"] = pdf_payload["total_chars"]
        row["pdf_kind"] = pdf_payload["pdf_kind"]
        row["extraction_method"] = "pypdf"
        if pdf_payload["pdf_kind"] != "text_pdf":
            ocr_text = find_sidecar_ocr_text(path, ocr_dir=ocr_dir)
            if ocr_text:
                raw_text = ocr_text
                row["ocr_used"] = True
                row["ocr_source"] = "sidecar"
                row["extraction_method"] = "sidecar_ocr"
            elif pdf_payload["pdf_kind"] == "scanned_pdf":
                row["needs_ocr"] = True
                row["parse_error"] = "scanned_pdf_requires_ocr"
                return row
            elif pdf_payload["pdf_kind"] == "weak_text_pdf":
                row["needs_ocr"] = True
                row["parse_error"] = "weak_text_pdf_requires_ocr_review"
                return row
    elif source_type == "image":
        ocr_text = find_sidecar_ocr_text(path, ocr_dir=ocr_dir)
        if ocr_text:
            raw_text = ocr_text
            row["ocr_used"] = True
            row["ocr_source"] = "sidecar"
            row["extraction_method"] = "sidecar_ocr"
        else:
            row["needs_ocr"] = True
            row["parse_error"] = "image_resume_requires_ocr"
            return row
    else:
        row["parse_error"] = "unsupported_file_type"
        return row

    clean_text = normalize_resume_text(raw_text)
    row["raw_text"] = raw_text
    row["clean_text"] = clean_text
    row["sections"] = split_resume_sections(clean_text)
    if source_type != "pdf":
        row["text_char_count"] = len(clean_text)
    row["parse_ok"] = bool(clean_text)
    if not clean_text:
        row["parse_error"] = "empty_text_after_extraction"
    return row


def collect_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Resume file or directory")
    parser.add_argument("--out", default="data/resume_raw/resume_ingest.jsonl")
    parser.add_argument("--ocr-dir", default=None, help="Optional directory containing sidecar OCR text files")
    args = parser.parse_args()

    input_path = Path(args.input)
    ocr_dir = Path(args.ocr_dir) if args.ocr_dir else None
    rows = [ingest_resume(path, ocr_dir=ocr_dir) for path in collect_files(input_path)]
    write_jsonl(args.out, rows)
    success = sum(1 for row in rows if row.get("parse_ok"))
    needs_ocr = sum(1 for row in rows if row.get("needs_ocr"))
    print(f"wrote {len(rows)} rows to {args.out} (parse_ok={success}, needs_ocr={needs_ocr})")


if __name__ == "__main__":
    main()
