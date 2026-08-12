from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobmatch_tune.resume.ingest import ingest_resume
from jobmatch_tune.utils.io import read_jsonl, write_jsonl, write_text


PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d(?:[-\s]?\d){8}(?!\d)")
EMAIL_RE = re.compile(r"[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WECHAT_RE = re.compile(r"(?i)\b(?:微信|wechat|vx)[:：]?\s*([A-Za-z][A-Za-z0-9_-]{5,})")
QQ_RE = re.compile(r"(?:QQ|qq)[:：]?\s*([1-9][0-9]{4,11})")
AGE_RE = re.compile(r"(?<!\d)([1-5]?\d)\s*岁")
NAME_LINE_RE = re.compile(r"^(?:姓名[:：]?\s*)?([\u4e00-\u9fff]{2,4})$")
PRIVATE_CONTEXT_RE = re.compile(r"(电话|手机|邮箱|年龄|政治面貌|\[手机号\]|\[邮箱\]|\[年龄\])")
RESUME_FILE_NAME_RE = re.compile(r"(?<=[-_])[\u4e00-\u9fff]{2,4}(?=\.)")
PRIVATE_PROFILE_FIELDS = {
    "姓名": "name",
    "年龄": "age",
    "性别": "gender",
    "婚姻状况": "marital_status",
    "户口地": "hukou",
    "籍贯": "hukou",
    "身体状况": "health_status",
    "残障情况": "health_status",
    "政治面貌": "political_status",
    "出生日期": "birth_date",
    "民族": "ethnicity",
}
TRAINING_PRIVATE_FIELDS = tuple(
    PRIVATE_PROFILE_FIELDS
    | {
        "电话": "phone",
        "手机": "phone",
        "联系电话": "phone",
        "邮箱": "email",
        "联系邮箱": "email",
        "微信": "wechat",
        "QQ": "qq",
    }
)


@dataclass(frozen=True)
class PiiFinding:
    kind: str
    value: str


def detect_resume_pii(text: str) -> list[PiiFinding]:
    findings: list[PiiFinding] = []
    patterns = [
        ("phone", PHONE_RE),
        ("email", EMAIL_RE),
        ("wechat", WECHAT_RE),
        ("qq", QQ_RE),
        ("age", AGE_RE),
    ]
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            value = match.group(1) if match.lastindex else match.group(0)
            findings.append(PiiFinding(kind=kind, value=value))

    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or any(token in stripped for token in ["教育", "实习", "项目", "技能"]):
            continue
        match = NAME_LINE_RE.fullmatch(stripped)
        previous_line = lines[index - 1] if index > 0 else ""
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        near_private_context = bool(PRIVATE_CONTEXT_RE.search(previous_line + "\n" + next_line))
        explicit_name_field = stripped.startswith("姓名")
        if match and (explicit_name_field or near_private_context):
            findings.append(PiiFinding(kind="name", value=match.group(1)))
        normalized_line = re.sub(r"^[\s#>*+\-]+", "", stripped).replace("**", "")
        profile_match = re.match(r"([^：:]+)\s*[：:]\s*(.+)$", normalized_line)
        if profile_match:
            field = profile_match.group(1).strip()
            kind = PRIVATE_PROFILE_FIELDS.get(field)
            if kind == "age" and AGE_RE.search(stripped):
                continue
            if kind and kind != "name":
                findings.append(PiiFinding(kind=kind, value=profile_match.group(2).strip()))
    return findings


def redact_resume_pii(text: str) -> str:
    redacted = PHONE_RE.sub("[手机号]", text)
    redacted = EMAIL_RE.sub("[邮箱]", redacted)
    redacted = WECHAT_RE.sub(lambda match: match.group(0).replace(match.group(1), "[微信]"), redacted)
    redacted = QQ_RE.sub(lambda match: match.group(0).replace(match.group(1), "[QQ]"), redacted)
    redacted = AGE_RE.sub("[年龄]", redacted)

    source_lines = redacted.splitlines()
    lines: list[str] = []
    for index, line in enumerate(source_lines):
        stripped = line.strip()
        match = NAME_LINE_RE.fullmatch(stripped)
        previous_line = source_lines[index - 1] if index > 0 else ""
        next_line = source_lines[index + 1] if index + 1 < len(source_lines) else ""
        near_private_context = bool(PRIVATE_CONTEXT_RE.search(previous_line + "\n" + next_line))
        explicit_name_field = stripped.startswith("姓名")
        if match and (explicit_name_field or near_private_context) and not any(
            token in stripped for token in ["教育", "实习", "项目", "技能"]
        ):
            lines.append(line.replace(match.group(1), "[姓名]"))
        else:
            lines.append(line)
    return "\n".join(lines)


def sanitize_resume_text_for_training(text: str) -> str:
    redacted = redact_resume_pii(text)
    kept_lines = []
    for line in redacted.splitlines():
        normalized_line = re.sub(r"^[\s#>*+\-]+", "", line.strip()).replace("**", "")
        if normalized_line in {"[姓名]", "[手机号]", "[邮箱]"}:
            continue
        if any(re.match(rf"{re.escape(field)}\s*[：:]", normalized_line) for field in TRAINING_PRIVATE_FIELDS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def redact_resume_metadata(value: str) -> str:
    value = RESUME_FILE_NAME_RE.sub("[姓名]", value)
    value = re.sub(
        r"(^|/)(?!个人)([\u4e00-\u9fff]{2,4})(?=简历)",
        lambda match: f"{match.group(1)}[姓名]",
        value,
    )
    return value


def summarize_pii(findings: list[PiiFinding]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    return {
        "has_pii": bool(findings),
        "total": len(findings),
        "counts": counts,
    }


def sanitize_resume_row(row: dict[str, Any]) -> dict[str, Any]:
    text_fields = ["raw_text", "clean_text", "normalized_text", "text"]
    combined_text = "\n".join(str(row.get(field) or "") for field in text_fields)
    findings = detect_resume_pii(combined_text)
    sanitized = dict(row)
    for field in text_fields:
        if field in sanitized and isinstance(sanitized[field], str):
            sanitized[field] = sanitize_resume_text_for_training(sanitized[field])
    for field in ["file_name", "file_path"]:
        if field in sanitized and isinstance(sanitized[field], str):
            sanitized[field] = redact_resume_metadata(sanitized[field])
    if isinstance(sanitized.get("sections"), dict):
        sanitized["sections"] = {
            key: sanitize_resume_text_for_training(str(value))
            for key, value in sanitized["sections"].items()
        }
    sanitized["privacy"] = summarize_pii(findings)
    sanitized["privacy"]["redacted"] = bool(findings)
    return sanitized


def build_resume_privacy_report(
    *,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row_reports = []
    aggregate_counts: dict[str, int] = {}
    for row in rows:
        text = "\n".join(
            str(row.get(field) or "") for field in ["raw_text", "clean_text", "normalized_text", "text"]
        )
        findings = detect_resume_pii(text)
        summary = summarize_pii(findings)
        for kind, count in summary["counts"].items():
            aggregate_counts[kind] = aggregate_counts.get(kind, 0) + int(count)
        row_reports.append(
            {
                "id": row.get("id", ""),
                "file_name": redact_resume_metadata(str(row.get("file_name", ""))),
                "has_pii": summary["has_pii"],
                "pii_total": summary["total"],
                "pii_counts": summary["counts"],
            }
        )
    rows_with_pii = sum(1 for item in row_reports if item["has_pii"])
    return {
        "row_count": len(rows),
        "rows_with_pii": rows_with_pii,
        "pii_row_rate": rows_with_pii / len(rows) if rows else 0.0,
        "pii_counts": aggregate_counts,
        "rows": row_reports,
    }


def _load_rows_from_file(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        return list(read_jsonl(file_path))
    row = ingest_resume(file_path)
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Resume file or JSONL rows")
    parser.add_argument("--out", default="")
    parser.add_argument("--report-out", default="outputs/eval_reports/resume_privacy_report.json")
    args = parser.parse_args()

    rows = _load_rows_from_file(args.input)
    report = build_resume_privacy_report(rows=rows)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.report_out, rendered + "\n")
    if args.out:
        write_jsonl(args.out, [sanitize_resume_row(row) for row in rows])


if __name__ == "__main__":
    main()
