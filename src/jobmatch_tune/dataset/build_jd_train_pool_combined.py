from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import is_high_trust_strong_row
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def build_manual_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        if not is_high_trust_strong_row(row):
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
                    **(row.get("meta") or {}),
                    "language": row.get("language", ""),
                    "sft_ready": row.get("sft_ready", False),
                    "pool_origin": "strict_manual",
                },
            }
        )
    return converted


def _dedup_key(row: dict[str, Any]) -> str:
    title = _normalize_text(row.get("job_title"))
    company = _normalize_text(row.get("company"))
    location = _normalize_text(row.get("location"))
    raw_text = _normalize_text(row.get("raw_text"))
    return hashlib.sha1(
        f"{title}\n{company}\n{location}\n{raw_text[:500]}".encode("utf-8")
    ).hexdigest()


def deduplicate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        key = _dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_combined_rows(
    manual_rows: Iterable[dict[str, Any]],
    public_rows: Iterable[dict[str, Any]],
    supplemental_rows: Iterable[dict[str, Any]] | None = None,
    weak_structured_rows: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    seen = set()
    combined = []

    def append_unique(row: dict[str, Any]) -> None:
        key = _dedup_key(row)
        if key not in seen:
            seen.add(key)
            combined.append(row)

    for row in build_manual_rows(manual_rows):
        append_unique(row)
    for row in supplemental_rows or []:
        copied = dict(row)
        meta = dict(copied.get("meta") or {})
        meta["pool_origin"] = meta.get("pool_origin") or "supplemental_candidate"
        copied["meta"] = meta
        append_unique(copied)
    for row in weak_structured_rows or []:
        copied = dict(row)
        meta = dict(copied.get("meta") or {})
        meta["pool_origin"] = meta.get("pool_origin") or "weak_structured_candidate"
        copied["meta"] = meta
        append_unique(copied)
    for row in public_rows:
        copied = dict(row)
        meta = dict(copied.get("meta") or {})
        meta["pool_origin"] = "public_candidate"
        copied["meta"] = meta
        append_unique(copied)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-input", default="data/interim/jd_clean_dedup.jsonl")
    parser.add_argument("--public-input", default="data/eval/public_jd_candidate_pool.jsonl")
    parser.add_argument("--supplemental-input", default="data/eval/jd_train_pool_supplemental.jsonl")
    parser.add_argument("--weak-structured-input", default="data/eval/jd_train_pool_weak_structured.jsonl")
    parser.add_argument("--out", default="data/eval/jd_train_pool_combined.jsonl")
    args = parser.parse_args()

    counts = {"manual": 0, "public": 0, "supplemental": 0, "weak_structured": 0}

    def counted_rows(path: str, key: str):
        if not Path(path).exists():
            return
        for row in read_jsonl(path):
            counts[key] += 1
            yield row

    manual_rows = counted_rows(args.manual_input, "manual")
    public_rows = counted_rows(args.public_input, "public")
    supplemental_rows = counted_rows(args.supplemental_input, "supplemental")
    weak_structured_rows = counted_rows(args.weak_structured_input, "weak_structured")
    combined = build_combined_rows(manual_rows, public_rows, supplemental_rows, weak_structured_rows)
    write_jsonl(args.out, combined)
    print(
        "manual="
        f"{counts['manual']} public={counts['public']} supplemental={counts['supplemental']} "
        f"weak_structured={counts['weak_structured']} combined={len(combined)}"
    )


if __name__ == "__main__":
    main()
