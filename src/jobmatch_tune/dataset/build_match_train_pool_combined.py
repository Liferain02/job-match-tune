from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def is_usable_public_match_row(row: dict[str, Any]) -> bool:
    if str(row.get("task") or "") != "match":
        return False
    meta = row.get("meta") or {}
    language = str(meta.get("language") or "").lower()
    if language not in {"", "zh", "zh-cn", "en"}:
        return False
    jd_text = _normalize_text(row.get("jd_text"))
    resume_text = _normalize_text(row.get("resume_text"))
    if len(jd_text) < 80 or len(resume_text) < 80:
        return False
    label = row.get("label") or {}
    raw_label = _normalize_text(label.get("raw_label"))
    raw_score = label.get("raw_score")
    return bool(raw_label or raw_score not in ("", None))


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        jd_text = _normalize_text(row.get("jd_text"))
        resume_text = _normalize_text(row.get("resume_text"))
        key = hashlib.sha1(f"{jd_text}\n---\n{resume_text}".encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_combined_rows(
    manual_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    combined = list(manual_rows)
    for row in synthetic_rows or []:
        combined.append(row)
    for row in public_rows:
        if is_usable_public_match_row(row):
            combined.append(
                {
                    "id": row["id"],
                    "task": "match",
                    "source_type": row.get("source_type", "public_pair"),
                    "jd_text": row["jd_text"],
                    "resume_text": row["resume_text"],
                    "label": row.get("label") or {},
                    "meta": row.get("meta") or {},
                }
            )
    return deduplicate_rows(combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-input", default="data/eval/match_manual_train_pool.jsonl")
    parser.add_argument("--public-input", default="data/external/public_match_imports.jsonl")
    parser.add_argument("--synthetic-input", default="data/eval/match_train_pool_synthetic.jsonl")
    parser.add_argument("--out", default="data/eval/match_train_pool_combined.jsonl")
    args = parser.parse_args()

    manual_rows = list(read_jsonl(args.manual_input))
    public_rows = list(read_jsonl(args.public_input)) if Path(args.public_input).exists() else []
    synthetic_rows = list(read_jsonl(args.synthetic_input)) if Path(args.synthetic_input).exists() else []
    combined = build_combined_rows(manual_rows, public_rows, synthetic_rows)
    write_jsonl(args.out, combined)
    print(
        f"manual={len(manual_rows)} public={len(public_rows)} synthetic={len(synthetic_rows)} combined={len(combined)}"
    )


if __name__ == "__main__":
    main()
