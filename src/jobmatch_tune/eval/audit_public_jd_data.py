from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import mean
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_text


def compute_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counter = Counter()
    language_counter = Counter()
    sft_ready_counter = Counter()
    title_lengths: list[int] = []
    text_lengths: list[int] = []
    education_count = 0
    experience_count = 0
    salary_count = 0

    for row in rows:
        source_counter[str(row.get("source") or "")] += 1
        meta = row.get("meta") or {}
        language_counter[str(meta.get("language") or "")] += 1
        sft_ready_counter[str(bool(meta.get("sft_ready")))] += 1
        title = str(row.get("job_title") or "")
        raw_text = str(row.get("raw_text") or "")
        if title:
            title_lengths.append(len(title))
        if raw_text:
            text_lengths.append(len(raw_text))
        if meta.get("education") or "学历要求：" in raw_text:
            education_count += 1
        if meta.get("experience") or meta.get("work_year") or "经验要求：" in raw_text:
            experience_count += 1
        if row.get("salary") or "薪资范围：" in raw_text:
            salary_count += 1

    total = len(rows)
    return {
        "total_rows": total,
        "source_distribution_top30": source_counter.most_common(30),
        "language_distribution": language_counter.most_common(),
        "sft_ready_distribution": sft_ready_counter.most_common(),
        "avg_title_length": mean(title_lengths) if title_lengths else 0.0,
        "avg_raw_text_length": mean(text_lengths) if text_lengths else 0.0,
        "education_coverage": education_count / total if total else 0.0,
        "experience_coverage": experience_count / total if total else 0.0,
        "salary_coverage": salary_count / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    report = compute_report(rows)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
