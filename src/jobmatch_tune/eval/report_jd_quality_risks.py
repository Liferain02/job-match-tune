from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.jd_quality_risk import (
    HIGH_RISK_THRESHOLD,
    RISK_WEIGHTS,
    extract_title,
    infer_source,
    prompt_text,
    risk_reasons,
    risk_score,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl, write_text


def build_risk_report(rows: list[dict[str, Any]], *, sample_limit: int = 100) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reason_counter = Counter()
    tier_counter = Counter()
    source_counter = Counter()
    risk_score_counter = Counter()
    samples = []

    for row in rows:
        reasons = risk_reasons(row)
        meta = row.get("meta") or {}
        tier = str(meta.get("quality_tier") or "unknown")
        source = infer_source(row)
        score = risk_score(reasons)
        tier_counter[tier] += 1
        source_counter[source] += 1
        risk_score_counter[score] += 1
        reason_counter.update(reasons)
        if score and len(samples) < sample_limit:
            samples.append(
                {
                    "id": row.get("id"),
                    "quality_tier": tier,
                    "quality_reason": meta.get("quality_reason", ""),
                    "source": source,
                    "risk_score": score,
                    "risk_reasons": reasons,
                    "title": extract_title(prompt_text(row)),
                    "assistant": row.get("messages", [])[-1].get("content", ""),
                }
            )

    total = len(rows)
    risky = sum(count for score, count in risk_score_counter.items() if score > 0)
    high_risk = sum(count for score, count in risk_score_counter.items() if score >= HIGH_RISK_THRESHOLD)
    report = {
        "total": total,
        "risky_samples": risky,
        "risky_rate": round(risky / total, 4) if total else 0.0,
        "high_risk_samples": high_risk,
        "high_risk_rate": round(high_risk / total, 4) if total else 0.0,
        "tier_counts": dict(tier_counter),
        "risk_score_counts": {str(key): value for key, value in sorted(risk_score_counter.items())},
        "risk_reason_counts": dict(reason_counter),
        "risk_weights": RISK_WEIGHTS,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "top_sources": source_counter.most_common(20),
        "sample_limit": sample_limit,
    }
    return report, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/sft_jd_quality/train.jsonl",
            "data/sft_jd_quality/valid.jsonl",
            "data/sft_jd_quality/test.jsonl",
        ],
    )
    parser.add_argument("--out", default="outputs/eval_reports/jd_quality_risk_report.json")
    parser.add_argument("--samples-out", default="outputs/eval_reports/jd_quality_risk_samples.jsonl")
    parser.add_argument("--sample-limit", type=int, default=200)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        if Path(path).exists():
            rows.extend(read_jsonl(path))
    report, samples = build_risk_report(rows, sample_limit=args.sample_limit)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")
    write_jsonl(args.samples_out, samples)


if __name__ == "__main__":
    main()
