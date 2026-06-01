from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_text


def build_resume_sft_profile(paths: list[str]) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    source_group_counts: Counter[str] = Counter()
    total = 0
    bootstrap_samples = 0
    bootstrap_source_groups: set[str] = set()
    for path in paths:
        split = Path(path).stem
        if not Path(path).exists():
            continue
        for row in read_jsonl(path):
            total += 1
            split_counts[split] += 1
            source_group = str(row.get("source_group") or row.get("id") or "")
            source_group_counts[source_group] += 1
            row_id = str(row.get("id") or "")
            prefix = f"{source_group}_"
            variant = row_id[len(prefix) :] if row_id.startswith(prefix) else "unknown"
            variant_counts[variant] += 1
            if "bootstrap" in source_group:
                bootstrap_samples += 1
                bootstrap_source_groups.add(source_group)

    unique_source_groups = len(source_group_counts)
    expansion_ratio = round(total / unique_source_groups, 4) if unique_source_groups else 0.0
    max_source_group_size = max(source_group_counts.values(), default=0)
    max_variant_rate = round(max(variant_counts.values(), default=0) / total, 4) if total else 0.0
    bootstrap_source_group_rate = (
        round(len(bootstrap_source_groups) / unique_source_groups, 4) if unique_source_groups else 0.0
    )
    return {
        "total": total,
        "split_counts": dict(split_counts),
        "unique_source_groups": unique_source_groups,
        "expansion_ratio": expansion_ratio,
        "max_source_group_size": max_source_group_size,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_rate": round(bootstrap_samples / total, 4) if total else 0.0,
        "bootstrap_source_groups": len(bootstrap_source_groups),
        "bootstrap_source_group_rate": bootstrap_source_group_rate,
        "variant_counts": dict(variant_counts.most_common()),
        "max_variant_rate": max_variant_rate,
        "profile_ready": (
            unique_source_groups >= 3000
            and expansion_ratio <= 20.0
            and max_source_group_size <= 20
            and max_variant_rate <= 0.1
            and bootstrap_source_group_rate <= 0.8
        ),
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
    parser.add_argument("--out", default="outputs/eval_reports/resume_sft_profile.json")
    args = parser.parse_args()

    report = build_resume_sft_profile(args.inputs)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
