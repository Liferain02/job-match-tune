from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import is_high_trust_strong_row
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def build_manual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    "language": row.get("language", ""),
                    "sft_ready": row.get("sft_ready", False),
                    "pool_origin": "strict_manual",
                },
            }
        )
    return converted


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


def build_combined_rows(
    manual_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
    supplemental_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    combined = build_manual_rows(manual_rows)
    for row in supplemental_rows or []:
        copied = dict(row)
        meta = dict(copied.get("meta") or {})
        meta["pool_origin"] = meta.get("pool_origin") or "supplemental_candidate"
        copied["meta"] = meta
        combined.append(copied)
    for row in public_rows:
        copied = dict(row)
        meta = dict(copied.get("meta") or {})
        meta["pool_origin"] = "public_candidate"
        copied["meta"] = meta
        combined.append(copied)
    return deduplicate_rows(combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-input", default="data/interim/jd_clean_dedup.jsonl")
    parser.add_argument("--public-input", default="data/eval/public_jd_candidate_pool.jsonl")
    parser.add_argument("--supplemental-input", default="data/eval/jd_train_pool_supplemental.jsonl")
    parser.add_argument("--out", default="data/eval/jd_train_pool_combined.jsonl")
    args = parser.parse_args()

    manual_rows = list(read_jsonl(args.manual_input))
    public_rows = list(read_jsonl(args.public_input)) if Path(args.public_input).exists() else []
    supplemental_rows = (
        list(read_jsonl(args.supplemental_input)) if Path(args.supplemental_input).exists() else []
    )
    combined = build_combined_rows(manual_rows, public_rows, supplemental_rows)
    write_jsonl(args.out, combined)
    print(
        f"manual={len(manual_rows)} public={len(public_rows)} supplemental={len(supplemental_rows)} combined={len(combined)}"
    )


if __name__ == "__main__":
    main()
