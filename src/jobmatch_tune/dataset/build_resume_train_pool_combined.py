from __future__ import annotations

import argparse
import hashlib
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def is_usable_public_resume_row(row: dict[str, Any]) -> bool:
    if str(row.get("task") or "") != "resume_parse":
        return False
    meta = row.get("meta") or {}
    language = str(meta.get("language") or "zh").lower()
    if language and not language.startswith("zh"):
        return False
    text = _normalize_text(row.get("text"))
    if len(text) < 80:
        return False
    label = row.get("label") or {}
    signals = 0
    if _normalize_text(label.get("目标岗位")):
        signals += 1
    if label.get("教育背景"):
        signals += 1
    if label.get("核心技能"):
        signals += 1
    if label.get("项目经历"):
        signals += 1
    if label.get("实习经历"):
        signals += 1
    return signals >= 3


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        text = _normalize_text(row.get("text"))
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_combined_rows(
    manual_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined = list(manual_rows)
    for row in public_rows:
        if is_usable_public_resume_row(row):
            combined.append(
                {
                    "id": row["id"],
                    "task": "resume_parse",
                    "source_type": row.get("source_type", "public_text"),
                    "text": row["text"],
                    "label": row.get("label") or {},
                }
            )
    return deduplicate_rows(combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-input", default="data/eval/resume_manual_train_pool.jsonl")
    parser.add_argument("--public-input", default="data/external/public_resume_imports.jsonl")
    parser.add_argument("--out", default="data/eval/resume_train_pool_combined.jsonl")
    args = parser.parse_args()

    manual_rows = list(read_jsonl(args.manual_input))
    public_rows = list(read_jsonl(args.public_input))
    combined = build_combined_rows(manual_rows, public_rows)
    write_jsonl(args.out, combined)
    print(f"manual={len(manual_rows)} public={len(public_rows)} combined={len(combined)}")


if __name__ == "__main__":
    main()
