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
    if str(meta.get("license_status") or "").lower() != "confirmed":
        return False
    if str(meta.get("intended_usage") or "").lower() not in {
        "training",
        "sft_training",
        "training_and_evaluation",
    }:
        return False
    if str(meta.get("provenance_status") or "").lower() in {"", "undocumented", "unknown"}:
        return False
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


def cap_educational_source_rows(
    rows: list[dict[str, Any]], max_rate: float = 0.4
) -> list[dict[str, Any]]:
    if not 0.0 <= max_rate <= 1.0:
        raise ValueError("max_rate must be between 0 and 1")
    if max_rate == 1.0:
        return rows
    educational = [
        row
        for row in rows
        if "synthetic_match_hf_job_educational_" in str(row.get("id") or "")
    ]
    other_count = len(rows) - len(educational)
    max_educational = int(max_rate * other_count / (1.0 - max_rate))
    if len(educational) <= max_educational:
        return rows
    keep_ids = {id(row) for row in educational[:max_educational]}
    return [
        row
        for row in rows
        if "synthetic_match_hf_job_educational_" not in str(row.get("id") or "")
        or id(row) in keep_ids
    ]


def build_combined_rows(
    manual_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
    max_educational_source_rate: float = 0.4,
) -> list[dict[str, Any]]:
    combined = list(manual_rows)
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
    return cap_educational_source_rows(
        deduplicate_rows(combined), max_rate=max_educational_source_rate
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-input",
        default="data/eval/match_curated_train_pool.jsonl",
        help="Optional independently annotated training pairs; never point this at a Gold/eval file.",
    )
    parser.add_argument("--public-input", default="data/external/public_match_imports.jsonl")
    parser.add_argument("--max-educational-source-rate", type=float, default=0.4)
    parser.add_argument("--out", default="data/eval/match_train_pool_combined.jsonl")
    args = parser.parse_args()

    manual_rows = (
        list(read_jsonl(args.manual_input))
        if args.manual_input and Path(args.manual_input).exists()
        else []
    )
    public_rows = list(read_jsonl(args.public_input)) if Path(args.public_input).exists() else []
    combined = build_combined_rows(
        manual_rows,
        public_rows,
        max_educational_source_rate=args.max_educational_source_rate,
    )
    write_jsonl(args.out, combined)
    print(
        f"manual={len(manual_rows)} public={len(public_rows)} combined={len(combined)}"
    )


if __name__ == "__main__":
    main()
