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
    intended_usage_counter = Counter()
    content_rights_counter = Counter()
    annotation_type_counter = Counter()
    split_role_counter = Counter()
    title_lengths: list[int] = []
    text_lengths: list[int] = []
    education_count = 0
    experience_count = 0
    salary_count = 0
    training_eligible_count = 0
    holdout_eligible_count = 0
    pinned_artifact_count = 0

    for row in rows:
        source_counter[str(row.get("source") or "")] += 1
        meta = row.get("meta") or {}
        language_counter[str(meta.get("language") or "")] += 1
        sft_ready_counter[str(bool(meta.get("sft_ready")))] += 1
        intended_usage_counter[str(meta.get("intended_usage") or "unspecified")] += 1
        content_rights_counter[str(meta.get("content_rights_status") or "unspecified")] += 1
        annotation_type_counter[str(meta.get("annotation_type") or "unspecified")] += 1
        split_role_counter[str(meta.get("split_role") or "unspecified")] += 1
        training_eligible_count += int(meta.get("training_eligible") is True)
        holdout_eligible_count += int(meta.get("holdout_eligible") is True)
        pinned_artifact_count += int(bool(meta.get("source_revision") and meta.get("artifact_sha256")))
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
        "intended_usage_distribution": intended_usage_counter.most_common(),
        "content_rights_distribution": content_rights_counter.most_common(),
        "annotation_type_distribution": annotation_type_counter.most_common(),
        "split_role_distribution": split_role_counter.most_common(),
        "training_eligible_rows": training_eligible_count,
        "holdout_eligible_rows": holdout_eligible_count,
        "pinned_artifact_rows": pinned_artifact_count,
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
