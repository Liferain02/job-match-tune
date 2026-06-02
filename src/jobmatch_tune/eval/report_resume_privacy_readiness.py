from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jobmatch_tune.resume.privacy import detect_resume_pii
from jobmatch_tune.utils.io import read_jsonl, write_text


def _iter_row_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_iter_row_texts(item))
        return texts
    if isinstance(value, dict):
        texts: list[str] = []
        for key, item in value.items():
            if key in {"raw_text", "clean_text", "normalized_text", "text", "content", "sections"}:
                texts.extend(_iter_row_texts(item))
            elif key == "messages":
                texts.extend(_iter_row_texts(item))
        return texts
    return []


def build_resume_privacy_readiness_report(
    *,
    paths: list[str],
    max_pii_row_rate: float = 0.0,
) -> dict[str, Any]:
    row_count = 0
    rows_with_pii = 0
    pii_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_pii_rows: Counter[str] = Counter()
    examples = []

    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        split = file_path.stem
        for row in read_jsonl(file_path):
            row_count += 1
            split_counts[split] += 1
            text = "\n".join(_iter_row_texts(row))
            findings = detect_resume_pii(text)
            if not findings:
                continue
            rows_with_pii += 1
            split_pii_rows[split] += 1
            row_counts = Counter(finding.kind for finding in findings)
            pii_counts.update(row_counts)
            if len(examples) < 20:
                examples.append(
                    {
                        "path": str(file_path),
                        "id": row.get("id", ""),
                        "source_group": row.get("source_group", ""),
                        "pii_counts": dict(row_counts),
                    }
                )

    pii_row_rate = rows_with_pii / row_count if row_count else 0.0
    return {
        "ready_for_resume_training": row_count > 0 and pii_row_rate <= max_pii_row_rate,
        "row_count": row_count,
        "rows_with_pii": rows_with_pii,
        "pii_row_rate": pii_row_rate,
        "max_pii_row_rate": max_pii_row_rate,
        "pii_counts": dict(pii_counts),
        "split_counts": dict(split_counts),
        "split_pii_rows": dict(split_pii_rows),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/sft_resume/train.jsonl",
            "data/sft_resume/valid.jsonl",
            "data/sft_resume/test.jsonl",
        ],
    )
    parser.add_argument("--max-pii-row-rate", type=float, default=0.0)
    parser.add_argument("--out", default="outputs/eval_reports/resume_privacy_readiness_report.json")
    args = parser.parse_args()

    report = build_resume_privacy_readiness_report(
        paths=args.inputs,
        max_pii_row_rate=args.max_pii_row_rate,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
