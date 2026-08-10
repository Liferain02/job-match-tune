from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from jobmatch_tune.eval.audit_jd_direction_conflicts import extract_title
from jobmatch_tune.preprocess.jd_field_rules import extract_experience_requirement
from jobmatch_tune.utils.io import read_jsonl, write_text


def audit_experience_gaps(rows: list[dict[str, Any]], *, sample_limit: int = 100) -> dict[str, Any]:
    empty_rows = 0
    gaps: list[dict[str, str]] = []
    tier_counts: Counter[str] = Counter()
    for row in rows:
        messages = row.get("messages") or []
        try:
            actual = str(json.loads(messages[-1]["content"]).get("经验要求") or "").strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            continue
        if actual:
            continue
        empty_rows += 1
        user_text = str(messages[1].get("content") or "") if len(messages) > 1 else ""
        recoverable = extract_experience_requirement(user_text)
        if not recoverable:
            continue
        meta = row.get("meta") or {}
        tier = str(meta.get("quality_tier") or "unknown")
        tier_counts[tier] += 1
        gaps.append(
            {
                "id": str(row.get("id") or ""),
                "title": extract_title(row),
                "recoverable_experience": recoverable,
                "quality_tier": tier,
                "source": str(meta.get("source") or "unknown"),
            }
        )
    return {
        "total_rows": len(rows),
        "empty_experience_rows": empty_rows,
        "recoverable_empty_rows": len(gaps),
        "recoverable_empty_rate": round(len(gaps) / empty_rows, 4) if empty_rows else 0.0,
        "interpretation": "confirmed_rule_extraction_gaps",
        "tier_counts": dict(tier_counts),
        "samples": gaps[:sample_limit],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 JD 正文明确经验信号与空标签")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/sft_jd_quality/train.jsonl",
            "data/sft_jd_quality/valid.jsonl",
            "data/sft_jd_quality/test.jsonl",
        ],
    )
    parser.add_argument("--out", default="outputs/eval_reports/jd_experience_gaps.json")
    parser.add_argument("--sample-limit", type=int, default=100)
    args = parser.parse_args()

    rows = [row for path in args.inputs for row in read_jsonl(path)]
    report = audit_experience_gaps(rows, sample_limit=args.sample_limit)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
