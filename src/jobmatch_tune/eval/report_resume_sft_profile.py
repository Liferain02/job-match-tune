from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.build_resume_train_pool_combined import is_technical_direction
from jobmatch_tune.utils.io import read_jsonl, write_text


REAL_RESUME_ORIGINS = {
    "public_real_anonymized",
    "public_real_self_published_anonymized",
    "private_real_anonymized",
}
MIN_REAL_RESUME_SOURCE_GROUPS = 100


def build_resume_sft_profile(paths: list[str]) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    source_group_counts: Counter[str] = Counter()
    total = 0
    bootstrap_samples = 0
    bootstrap_source_groups: set[str] = set()
    split_source_groups: dict[str, set[str]] = {}
    source_category_groups: dict[str, set[str]] = {}
    split_source_category_groups: dict[str, dict[str, set[str]]] = {}
    real_source_groups: set[str] = set()
    language_counts: Counter[str] = Counter()
    non_technical_rows = 0

    def source_category(source_group: str) -> str:
        if source_group.startswith("resume_bootstrap_"):
            return "bootstrap_from_jd"
        if source_group.startswith("resume_eval_"):
            return "repository_curated_seed"
        return source_group.split("_", 1)[0] or "unknown"

    for path in paths:
        split = Path(path).stem
        if not Path(path).exists():
            continue
        for row in read_jsonl(path):
            total += 1
            split_counts[split] += 1
            source_group = str(row.get("source_group") or row.get("id") or "")
            source_group_counts[source_group] += 1
            split_source_groups.setdefault(split, set()).add(source_group)
            data_origin = str((row.get("meta") or {}).get("data_origin") or "")
            language = str((row.get("meta") or {}).get("language") or "unknown").lower()
            language_counts[language] += 1
            messages = row.get("messages") or []
            try:
                label = json.loads(str(messages[-1].get("content") or "{}")) if messages else {}
            except (TypeError, json.JSONDecodeError):
                label = {}
            non_technical_rows += int(not is_technical_direction(label.get("目标岗位")))
            category = data_origin or source_category(source_group)
            if data_origin in REAL_RESUME_ORIGINS:
                real_source_groups.add(source_group)
            source_category_groups.setdefault(category, set()).add(source_group)
            split_source_category_groups.setdefault(split, {}).setdefault(category, set()).add(
                source_group
            )
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
    chinese_rows = sum(
        count for language, count in language_counts.items() if language.startswith("zh")
    )
    scope_ready = bool(total and chinese_rows == total and non_technical_rows == 0)
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
        "real_resume_source_groups": len(real_source_groups),
        "minimum_real_resume_source_groups": MIN_REAL_RESUME_SOURCE_GROUPS,
        "supports_real_resume_quality_claim": (
            len(real_source_groups) >= MIN_REAL_RESUME_SOURCE_GROUPS
        ),
        "language_counts": dict(language_counts),
        "chinese_rows": chinese_rows,
        "chinese_rate": round(chinese_rows / total, 4) if total else 0.0,
        "non_technical_rows": non_technical_rows,
        "scope_ready": scope_ready,
        "source_group_counts_by_category": {
            category: len(groups) for category, groups in sorted(source_category_groups.items())
        },
        "split_unique_source_groups": {
            split: len(groups) for split, groups in sorted(split_source_groups.items())
        },
        "split_source_group_counts_by_category": {
            split: {
                category: len(groups)
                for category, groups in sorted(categories.items())
            }
            for split, categories in sorted(split_source_category_groups.items())
        },
        "variant_counts": dict(variant_counts.most_common()),
        "max_variant_rate": max_variant_rate,
        "profile_ready": (
            unique_source_groups >= 3000
            and expansion_ratio <= 20.0
            # Six deliberately distinct renderings cover the core input shapes
            # without inflating every source into twenty hand-written templates.
            and max_source_group_size <= 7
            # Original-only real data does not need artificial render variants.
            and (expansion_ratio <= 1.1 or max_variant_rate <= 0.25)
            and bootstrap_source_group_rate <= 0.8
            and len(real_source_groups) >= MIN_REAL_RESUME_SOURCE_GROUPS
            and scope_ready
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
