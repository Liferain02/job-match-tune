from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import mean
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_text


def compute_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_counter = Counter()
    source_counter = Counter()
    language_counter = Counter()
    source_type_counter = Counter()
    text_lengths: list[int] = []
    target_job_count = 0
    education_count = 0
    skill_count = 0
    internship_count = 0
    project_count = 0
    ner_tag_set = set()
    ner_tag_counter = Counter()
    sensitive_ner_rows = 0
    sensitive_entity_types = {"CONT", "LOC", "NAME", "ORG", "RACE"}
    resume_parse_rows = 0

    for row in rows:
        task = str(row.get("task") or "")
        task_counter[task] += 1
        meta = row.get("meta") or {}
        source_counter[str(meta.get("source_name") or "")] += 1
        language_counter[str(meta.get("language") or "")] += 1
        source_type_counter[str(row.get("source_type") or "")] += 1
        text = str(row.get("text") or "")
        if text:
            text_lengths.append(len(text))
        if task == "resume_parse":
            resume_parse_rows += 1
            label = row.get("label") or {}
            if label.get("目标岗位"):
                target_job_count += 1
            if label.get("教育背景"):
                education_count += 1
            if label.get("核心技能"):
                skill_count += 1
            if label.get("实习经历"):
                internship_count += 1
            if label.get("项目经历"):
                project_count += 1
        elif task == "resume_ner":
            row_tags = [str(tag) for tag in (row.get("ner_tags") or [])]
            ner_tag_set.update(row_tags)
            ner_tag_counter.update(row_tags)
            if any(tag.split("-", 1)[-1] in sensitive_entity_types for tag in row_tags):
                sensitive_ner_rows += 1

    total = len(rows)
    return {
        "total_rows": total,
        "task_distribution": task_counter.most_common(),
        "source_distribution": source_counter.most_common(),
        "language_distribution": language_counter.most_common(),
        "source_type_distribution": source_type_counter.most_common(),
        "avg_text_length": mean(text_lengths) if text_lengths else 0.0,
        "resume_parse_label_coverage": {
            "target_job": target_job_count / resume_parse_rows if resume_parse_rows else 0.0,
            "education": education_count / resume_parse_rows if resume_parse_rows else 0.0,
            "skills": skill_count / resume_parse_rows if resume_parse_rows else 0.0,
            "internships": internship_count / resume_parse_rows if resume_parse_rows else 0.0,
            "projects": project_count / resume_parse_rows if resume_parse_rows else 0.0,
        },
        "resume_ner_tag_count": len(ner_tag_set),
        "resume_ner_tags": sorted(ner_tag_set),
        "resume_ner_tag_distribution": ner_tag_counter.most_common(),
        "resume_ner_rows_with_sensitive_entities": sensitive_ner_rows,
        "resume_ner_training_ready": sensitive_ner_rows == 0,
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
