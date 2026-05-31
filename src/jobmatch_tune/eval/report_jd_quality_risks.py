from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_jsonl, write_text


SUSPICIOUS_TITLE_KEYWORDS = [
    "教师",
    "老师",
    "讲师",
    "助教",
    "培训师",
    "编导",
    "编剧",
    "摄制",
    "新闻编辑",
    "校园招聘",
    "管培生",
    "实习",
]

WEAK_SOURCE_PREFIXES = ("hf_", "github_")

RISK_WEIGHTS = {
    "invalid_assistant_json": 5,
    "suspicious_title_keyword": 4,
    "quality_weak_missing_core_field": 3,
    "empty_responsibilities": 2,
    "oversized_single_responsibility": 2,
    "empty_skills": 1,
    "empty_education": 1,
    "empty_experience": 1,
    "quality_weak_tier": 0,
    "weak_public_source": 0,
}


def _assistant_json(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return json.loads(row["messages"][-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def _prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return ""
    return str(messages[1].get("content") or "")


def extract_title(prompt: str) -> str:
    match = re.search(r"岗位名称[：:]\s*([^\n]+)", prompt)
    return match.group(1).strip() if match else ""


def infer_source(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    source = str(meta.get("source") or "")
    if source:
        return source
    row_id = str(row.get("id") or "")
    if row_id.startswith("hf_job_educational_train"):
        return "hf_job_educational_train_2026_05_17"
    if row_id.startswith("hf_job_educational_validation"):
        return "hf_job_educational_validation_2026_05_17"
    if row_id.startswith("github_workaggregation_test"):
        return "github_workaggregation_test"
    return row_id.replace("_jd_parse", "").split("_")[0]


def risk_reasons(row: dict[str, Any]) -> list[str]:
    assistant = _assistant_json(row)
    if assistant is None:
        return ["invalid_assistant_json"]

    reasons = []
    meta = row.get("meta") or {}
    tier = str(meta.get("quality_tier") or "")
    source = infer_source(row)
    prompt = _prompt(row)
    title = extract_title(prompt)

    if tier == "quality_weak":
        reasons.append("quality_weak_tier")
    if source.startswith(WEAK_SOURCE_PREFIXES):
        reasons.append("weak_public_source")
    if any(keyword in title for keyword in SUSPICIOUS_TITLE_KEYWORDS):
        reasons.append("suspicious_title_keyword")
    if not assistant.get("核心职责"):
        reasons.append("empty_responsibilities")
    if not assistant.get("必备技能"):
        reasons.append("empty_skills")
    if not assistant.get("学历要求"):
        reasons.append("empty_education")
    if not assistant.get("经验要求"):
        reasons.append("empty_experience")
    if len(assistant.get("核心职责") or []) == 1:
        first = str((assistant.get("核心职责") or [""])[0])
        if len(first) > 500:
            reasons.append("oversized_single_responsibility")
    if tier == "quality_weak" and (not assistant.get("学历要求") or not assistant.get("必备技能")):
        reasons.append("quality_weak_missing_core_field")
    return reasons


def risk_score(reasons: list[str]) -> int:
    return sum(RISK_WEIGHTS.get(reason, 1) for reason in reasons)


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
                    "title": extract_title(_prompt(row)),
                    "assistant": row.get("messages", [])[-1].get("content", ""),
                }
            )

    total = len(rows)
    risky = sum(count for score, count in risk_score_counter.items() if score > 0)
    high_risk = sum(count for score, count in risk_score_counter.items() if score >= 4)
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
        "high_risk_threshold": 4,
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
