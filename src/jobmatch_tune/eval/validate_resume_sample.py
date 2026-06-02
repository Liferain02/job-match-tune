from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jobmatch_tune.resume.ingest import ingest_resume
from jobmatch_tune.resume.privacy import detect_resume_pii, summarize_pii
from jobmatch_tune.utils.io import write_text


DEFAULT_REQUIRED_SECTIONS = ["education", "skills", "internships", "projects"]


def build_resume_sample_report(
    *,
    path: str | Path,
    min_text_chars: int = 800,
    required_sections: list[str] | None = None,
) -> dict[str, Any]:
    required = required_sections or DEFAULT_REQUIRED_SECTIONS
    ingest_row = ingest_resume(Path(path))
    sections = ingest_row.get("sections") or {}
    privacy = summarize_pii(detect_resume_pii(str(ingest_row.get("clean_text") or "")))
    found_sections = sorted(sections.keys())
    missing_sections = [section for section in required if section not in sections]
    text_char_count = int(ingest_row.get("text_char_count") or len(ingest_row.get("clean_text") or ""))
    checks = [
        {
            "name": "parse_ok",
            "passed": bool(ingest_row.get("parse_ok")),
            "actual": bool(ingest_row.get("parse_ok")),
            "expected": True,
        },
        {
            "name": "does_not_need_ocr",
            "passed": not bool(ingest_row.get("needs_ocr")),
            "actual": bool(ingest_row.get("needs_ocr")),
            "expected": False,
        },
        {
            "name": "min_text_chars",
            "passed": text_char_count >= min_text_chars,
            "actual": text_char_count,
            "threshold": min_text_chars,
        },
        {
            "name": "required_sections",
            "passed": not missing_sections,
            "actual": found_sections,
            "required": required,
            "missing": missing_sections,
        },
    ]
    failed = [item for item in checks if not item["passed"]]
    return {
        "ready_for_resume_file_parse": not failed,
        "file_name": Path(path).name,
        "source_type": ingest_row.get("source_type", ""),
        "pdf_kind": ingest_row.get("pdf_kind", ""),
        "extraction_method": ingest_row.get("extraction_method", ""),
        "page_count": ingest_row.get("page_count", 0),
        "text_char_count": text_char_count,
        "sections_found": found_sections,
        "privacy": privacy,
        "required_sections": required,
        "num_checks": len(checks),
        "num_failed_checks": len(failed),
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Resume file used as a private product smoke sample")
    parser.add_argument("--min-text-chars", type=int, default=800)
    parser.add_argument(
        "--required-sections",
        nargs="*",
        default=DEFAULT_REQUIRED_SECTIONS,
        help="Required normalized section keys, for example education skills projects",
    )
    parser.add_argument("--out", default="outputs/eval_reports/resume_sample_validation_report.json")
    args = parser.parse_args()

    report = build_resume_sample_report(
        path=args.input,
        min_text_chars=args.min_text_chars,
        required_sections=args.required_sections,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
