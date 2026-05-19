from __future__ import annotations

import argparse
import hashlib
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import (
    HIGH_TRUST_SOURCES,
    is_high_confidence_weak_tech_row,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


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


def build_supplemental_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        if row.get("source") in HIGH_TRUST_SOURCES:
            continue
        if not is_high_confidence_weak_tech_row(row):
            continue
        labels = row.get("labels") or {}
        if not str(labels.get("岗位方向") or "").strip():
            continue
        converted.append(
            {
                "id": row["id"],
                "source": row.get("source", ""),
                "job_title": row.get("job_title", ""),
                "company": row.get("company", ""),
                "location": row.get("location", ""),
                "salary": row.get("salary", ""),
                "raw_text": row.get("clean_text") or row.get("raw_text") or "",
                "meta": {
                    "language": row.get("language", ""),
                    "sft_ready": row.get("sft_ready", False),
                    "pool_origin": "supplemental_weak_high_conf",
                },
            }
        )
    return deduplicate_rows(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/jd_clean_dedup.jsonl")
    parser.add_argument("--out", default="data/eval/jd_train_pool_supplemental.jsonl")
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    supplemental = build_supplemental_rows(rows)
    write_jsonl(args.out, supplemental)
    print(f"input={len(rows)} supplemental={len(supplemental)}")


if __name__ == "__main__":
    main()
