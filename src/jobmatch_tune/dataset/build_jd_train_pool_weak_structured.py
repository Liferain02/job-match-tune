from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import (
    STRONG_TITLE_EXCLUDE_KEYWORDS,
    STRONG_TITLE_INCLUDE_KEYWORDS,
    WEAK_TECH_SOURCES,
    is_external_source_training_allowed,
)
from jobmatch_tune.preprocess.jd_field_rules import extract_education_requirement
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def is_usable_weak_structured_row(row: dict[str, Any]) -> bool:
    if not is_external_source_training_allowed(row):
        return False
    if row.get("source") not in WEAK_TECH_SOURCES:
        return False
    if row.get("language") != "zh":
        return False

    labels = row.get("labels") or {}
    direction = _normalize_text(labels.get("岗位方向"))
    if not direction:
        return False

    title = _normalize_text(row.get("job_title"))
    if len(title) < 2:
        return False
    if _contains_any(title, STRONG_TITLE_EXCLUDE_KEYWORDS):
        return False
    if not _contains_any(title, STRONG_TITLE_INCLUDE_KEYWORDS) and direction not in {
        "算法工程",
        "后端开发",
        "前端开发",
        "客户端开发",
        "测试开发",
        "运维开发",
        "数据开发",
        "嵌入式开发",
        "硬件研发",
        "安全工程",
        "网络与基础设施",
        "AI Infra",
        "高性能计算",
        "汽车软件/智驾研发",
    }:
        return False

    clean_text = _normalize_text(row.get("clean_text"))
    if len(clean_text) < 140:
        return False

    sections = row.get("sections") or {}
    has_responsibilities = bool(_normalize_text(sections.get("responsibilities")))
    has_requirements = bool(_normalize_text(sections.get("requirements")))
    has_bonus = bool(_normalize_text(sections.get("bonus")))
    has_education = bool(_normalize_text(labels.get("学历要求")) or extract_education_requirement(clean_text))
    has_skills = bool(labels.get("必备技能"))
    structure_markers = any(
        marker in clean_text
        for marker in ("岗位职责", "工作职责", "职位描述", "工作内容", "任职要求", "岗位要求", "技能要求", "任职资格")
    )

    if not has_education:
        return False
    if not (has_requirements or has_responsibilities or has_skills):
        return False
    if not structure_markers and not (has_responsibilities and has_requirements):
        return False
    if len(clean_text) < 220 and not has_skills:
        return False
    if not (has_requirements or has_responsibilities or has_bonus):
        return False
    return True


def _dedup_key(row: dict[str, Any]) -> str:
    title = _normalize_text(row.get("job_title"))
    company = _normalize_text(row.get("company"))
    location = _normalize_text(row.get("location"))
    raw_text = _normalize_text(row.get("raw_text"))
    return hashlib.sha1(
        f"{title}\n{company}\n{location}\n{raw_text[:500]}".encode("utf-8")
    ).hexdigest()


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        key = _dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    built = []
    seen = set()
    for row in rows:
        if not is_usable_weak_structured_row(row):
            continue
        candidate = {
            "id": row["id"],
            "source": row.get("source", ""),
            "job_title": row.get("job_title", ""),
            "company": row.get("company", ""),
            "location": row.get("location", ""),
            "salary": row.get("salary", ""),
            "raw_text": row.get("clean_text") or row.get("raw_text") or "",
            "meta": {
                **(row.get("meta") or {}),
                "language": row.get("language", ""),
                "sft_ready": row.get("sft_ready", False),
                "pool_origin": "weak_structured_candidate",
            },
        }
        key = _dedup_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        built.append(candidate)
    return built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/jd_clean_dedup.jsonl")
    parser.add_argument("--out", default="data/eval/jd_train_pool_weak_structured.jsonl")
    args = parser.parse_args()

    input_count = 0

    def counted_rows():
        nonlocal input_count
        for row in read_jsonl(args.input):
            input_count += 1
            yield row

    built = build_rows(counted_rows())
    write_jsonl(args.out, built)
    print(f"input={input_count} weak_structured={len(built)}")


if __name__ == "__main__":
    main()
