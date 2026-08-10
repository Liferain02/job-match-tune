from __future__ import annotations

import argparse
import hashlib
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import (
    HIGH_TRUST_SOURCES,
    get_effective_direction,
)
from jobmatch_tune.eval.report_jd_strict_rejections import classify_rejection
from jobmatch_tune.eval.report_jd_strict_tech_candidates import is_tech_like_title
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


REPAIRABLE_REASONS = {
    "missing_direction",
    "missing_sections",
    "missing_edu_exp_skill",
    "clean_text_too_short",
}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def is_repairable_row(row: dict[str, Any]) -> bool:
    if row.get("source") not in HIGH_TRUST_SOURCES:
        return False
    title = _normalize_text(row.get("job_title"))
    clean_text = _normalize_text(row.get("clean_text"))
    if not title or not clean_text:
        return False
    if not is_tech_like_title(title):
        return False

    labels = row.get("labels") or {}
    original_direction = _normalize_text(labels.get("岗位方向"))
    direction = get_effective_direction(row)
    reason = classify_rejection(row)

    repair_reason = reason
    if not original_direction and direction:
        repair_reason = "missing_direction"

    if repair_reason not in REPAIRABLE_REASONS:
        return False

    sections = row.get("sections") or {}
    has_education = bool(_normalize_text(labels.get("学历要求")) or extract_education_requirement(clean_text))
    has_experience = bool(_normalize_text(labels.get("经验要求")) or extract_experience_requirement(clean_text))
    has_skills = bool(labels.get("必备技能"))
    responsibilities = _normalize_text(sections.get("responsibilities"))
    requirements = _normalize_text(sections.get("requirements"))
    has_structure_marker = any(
        marker in clean_text
        for marker in (
            "岗位职责",
            "工作职责",
            "职位描述",
            "工作内容",
            "职责描述",
            "任职要求",
            "岗位要求",
            "职位要求",
            "任职资格",
            "能力要求",
            "技能要求",
        )
    )

    if repair_reason == "missing_direction":
        return bool(direction) and len(clean_text) >= 120 and (has_experience or has_skills or has_education)
    if repair_reason == "missing_sections":
        return bool(direction) and len(clean_text) >= 180 and has_structure_marker and (has_education or has_experience or has_skills)
    if repair_reason == "missing_edu_exp_skill":
        return bool(direction) and len(clean_text) >= 180 and has_structure_marker and (responsibilities or requirements)
    if repair_reason == "clean_text_too_short":
        return bool(direction) and 100 <= len(clean_text) < 180 and (has_experience or has_skills or has_education)
    return False


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    labels = row.get("labels") or {}
    original_direction = _normalize_text(labels.get("岗位方向"))
    repaired_direction = get_effective_direction(row)
    repair_reason = classify_rejection(row)
    if not original_direction and repaired_direction:
        repair_reason = "missing_direction"
    return {
        "id": row["id"],
        "source": row.get("source", ""),
        "job_title": row.get("job_title", ""),
        "company": row.get("company", ""),
        "location": row.get("location", ""),
        "salary": row.get("salary", ""),
        "raw_text": row.get("clean_text") or row.get("raw_text") or "",
        "meta": {
            **dict(row.get("meta") or {}),
            "language": row.get("language", ""),
            "sft_ready": row.get("sft_ready", False),
            "pool_origin": "repairable_candidate",
            "repair_reason": repair_reason,
            "repaired_direction": repaired_direction,
            "repair_has_education": bool(_normalize_text(labels.get("学历要求"))),
            "repair_has_experience": bool(_normalize_text(labels.get("经验要求"))),
            "repair_has_skills": bool(labels.get("必备技能")),
        },
    }


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        key = hashlib.sha1(
            (
                f"{_normalize_text(row.get('job_title'))}\n"
                f"{_normalize_text(row.get('company'))}\n"
                f"{_normalize_text(row.get('location'))}\n"
                f"{_normalize_text(row.get('raw_text'))[:500]}"
            ).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    built = [convert_row(row) for row in rows if is_repairable_row(row)]
    return deduplicate_rows(built)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/jd_clean_dedup.jsonl")
    parser.add_argument("--out", default="data/eval/jd_train_pool_repairable.jsonl")
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    built = build_rows(rows)
    write_jsonl(args.out, built)
    print(f"repairable={len(built)}")


if __name__ == "__main__":
    main()
