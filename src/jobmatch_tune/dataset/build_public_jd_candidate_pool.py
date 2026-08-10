from __future__ import annotations

import argparse
import hashlib
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import (
    STRONG_TITLE_EXCLUDE_KEYWORDS,
    STRONG_TITLE_INCLUDE_KEYWORDS,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def is_usable_public_jd_row(row: dict[str, Any]) -> bool:
    meta = row.get("meta") or {}
    if meta.get("training_eligible") is not True:
        return False
    intended_usage = str(meta.get("intended_usage") or "").lower()
    if intended_usage not in {"training", "training_and_evaluation", "weak_supervision_only"}:
        return False
    language = str(meta.get("language") or "").lower()
    if language and not language.startswith("zh"):
        return False
    title = _normalize_text(row.get("job_title"))
    raw_text = _normalize_text(row.get("raw_text"))
    if len(title) < 2 or len(raw_text) < 120:
        return False
    if not _contains_any(title, STRONG_TITLE_INCLUDE_KEYWORDS):
        return False
    if _contains_any(title, STRONG_TITLE_EXCLUDE_KEYWORDS):
        return False
    signals = 0
    if row.get("salary") or "薪资范围：" in raw_text:
        signals += 1
    if meta.get("education") or "学历要求：" in raw_text:
        signals += 1
    if meta.get("experience") or meta.get("work_year") or "经验要求：" in raw_text:
        signals += 1
    if "岗位职责：" in raw_text or "职位描述：" in raw_text:
        signals += 1
    return signals >= 2


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        title = _normalize_text(row.get("job_title"))
        company = _normalize_text(row.get("company"))
        location = _normalize_text(row.get("location"))
        raw_text = _normalize_text(row.get("raw_text"))
        key = hashlib.sha1(
            f"{title}\n{company}\n{location}\n{raw_text[:500]}".encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        if not is_usable_public_jd_row(row):
            continue
        candidates.append(
            {
                "id": row["id"],
                "source": row.get("source", ""),
                "job_title": row.get("job_title", ""),
                "company": row.get("company", ""),
                "location": row.get("location", ""),
                "salary": row.get("salary", ""),
                "raw_text": row.get("raw_text", ""),
                "meta": row.get("meta") or {},
            }
        )
    return deduplicate_rows(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/public_job_datasets_raw.jsonl")
    parser.add_argument("--out", default="data/eval/public_jd_candidate_pool.jsonl")
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    candidates = build_candidate_rows(rows)
    write_jsonl(args.out, candidates)
    print(f"input={len(rows)} candidates={len(candidates)}")


if __name__ == "__main__":
    main()
