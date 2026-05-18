from __future__ import annotations

import argparse
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_jsonl


SECTION_TITLES = {
    "header": "基础信息",
    "education": "教育背景",
    "skills": "核心技能",
    "internships": "实习经历",
    "projects": "项目经历",
    "work": "工作经历",
    "awards": "奖项证书",
    "profile": "个人优势",
}


def build_section_map_from_text(text: str) -> dict[str, Any]:
    from jobmatch_tune.resume.ingest import normalize_resume_text, split_resume_sections

    clean_text = normalize_resume_text(text)
    sections = split_resume_sections(clean_text)
    return {
        "clean_text": clean_text,
        "sections": sections,
        "normalized_text": render_normalized_resume({"clean_text": clean_text, "sections": sections}),
    }


def render_normalized_resume(row: dict[str, Any]) -> str:
    sections = row.get("sections") or {}
    ordered_keys = ["header", "education", "skills", "internships", "projects", "work", "awards", "profile"]
    blocks: list[str] = []
    for key in ordered_keys:
        value = str(sections.get(key) or "").strip()
        if not value:
            continue
        title = SECTION_TITLES[key]
        if key == "header":
            blocks.append(value)
        else:
            blocks.append(f"{title}：\n{value}")
    if not blocks:
        return str(row.get("clean_text") or "").strip()
    return "\n\n".join(blocks).strip()


def normalize_ingest_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized_text = render_normalized_resume(row)
    normalized = {
        "id": row["id"],
        "file_name": row.get("file_name", ""),
        "file_path": row.get("file_path", ""),
        "source_type": row.get("source_type", ""),
        "pdf_kind": row.get("pdf_kind", ""),
        "ocr_used": row.get("ocr_used", False),
        "ocr_source": row.get("ocr_source", ""),
        "extraction_method": row.get("extraction_method", ""),
        "parse_ok": row.get("parse_ok", False),
        "needs_ocr": row.get("needs_ocr", False),
        "clean_text": row.get("clean_text", ""),
        "normalized_text": normalized_text,
        "sections": row.get("sections", {}),
    }
    if row.get("parse_error"):
        normalized["parse_error"] = row["parse_error"]
    return normalized


def normalize_resume_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    rendered = build_section_map_from_text(str(row.get("text") or ""))
    return {
        "id": row["id"],
        "task": row.get("task", "resume_parse"),
        "source_type": row.get("source_type", "text"),
        "text": row.get("text", ""),
        "label": row.get("label", {}),
        "clean_text": rendered["clean_text"],
        "sections": rendered["sections"],
        "normalized_text": rendered["normalized_text"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/resume_raw/resume_ingest.jsonl")
    parser.add_argument("--out", default="data/resume_interim/resume_clean.jsonl")
    parser.add_argument("--only-parse-ok", action="store_true")
    args = parser.parse_args()

    rows = [normalize_ingest_row(row) for row in read_jsonl(args.input)]
    if args.only_parse_ok:
        rows = [row for row in rows if row.get("parse_ok")]
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
