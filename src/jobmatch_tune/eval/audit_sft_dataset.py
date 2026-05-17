from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import mean
from typing import Any

from jobmatch_tune.utils.io import read_jsonl


def compute_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counter = Counter()
    direction_counter = Counter()
    responsibility_lengths: list[int] = []
    skill_counts: list[int] = []
    education_count = 0
    experience_count = 0
    valid_json_count = 0

    for row in rows:
        raw_id = str(row.get("id") or "").replace("_jd_parse", "")
        source_key = raw_id.split("_")[0] if "_" in raw_id else raw_id
        source_counter[source_key] += 1
        messages = row.get("messages") or []
        if len(messages) < 3:
            continue
        try:
            assistant = json.loads(messages[-1]["content"])
            valid_json_count += 1
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        direction_counter[str(assistant.get("岗位方向") or "")] += 1
        responsibility_lengths.append(len(assistant.get("核心职责") or []))
        skill_counts.append(len(assistant.get("必备技能") or []))
        if assistant.get("学历要求"):
            education_count += 1
        if assistant.get("经验要求"):
            experience_count += 1

    total = len(rows)
    return {
        "total_samples": total,
        "json_valid_rate": valid_json_count / total if total else 0.0,
        "source_distribution_top20": source_counter.most_common(20),
        "direction_distribution": direction_counter.most_common(),
        "avg_responsibility_count": mean(responsibility_lengths) if responsibility_lengths else 0.0,
        "avg_skill_count": mean(skill_counts) if skill_counts else 0.0,
        "education_coverage": education_count / total if total else 0.0,
        "experience_coverage": experience_count / total if total else 0.0,
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
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")


if __name__ == "__main__":
    main()
