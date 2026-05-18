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
    source_type_counter = Counter()
    raw_label_counter = Counter()
    jd_lengths: list[int] = []
    resume_lengths: list[int] = []
    score_present = 0

    for row in rows:
        meta = row.get("meta") or {}
        source_counter[str(meta.get("source_name") or "")] += 1
        language_counter[str(meta.get("language") or "")] += 1
        source_type_counter[str(row.get("source_type") or "")] += 1
        jd_text = str(row.get("jd_text") or "")
        resume_text = str(row.get("resume_text") or "")
        if jd_text:
            jd_lengths.append(len(jd_text))
        if resume_text:
            resume_lengths.append(len(resume_text))
        label = row.get("label") or {}
        raw_label = str(label.get("raw_label") or "")
        if raw_label:
            raw_label_counter[raw_label] += 1
        if label.get("raw_score") not in ("", None):
            score_present += 1

    total = len(rows)
    return {
        "total_rows": total,
        "source_distribution": source_counter.most_common(),
        "language_distribution": language_counter.most_common(),
        "source_type_distribution": source_type_counter.most_common(),
        "raw_label_distribution_top20": raw_label_counter.most_common(20),
        "avg_jd_length": mean(jd_lengths) if jd_lengths else 0.0,
        "avg_resume_length": mean(resume_lengths) if resume_lengths else 0.0,
        "score_coverage": score_present / total if total else 0.0,
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
