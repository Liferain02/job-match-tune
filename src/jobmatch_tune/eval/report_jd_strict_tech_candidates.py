from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import HIGH_TRUST_SOURCES
from jobmatch_tune.eval.report_jd_strict_rejections import classify_rejection
from jobmatch_tune.utils.io import read_jsonl, write_jsonl, write_text


TECH_LIKE_TITLE_PATTERNS = [
    r"工程师",
    r"开发",
    r"算法",
    r"测试",
    r"研发",
    r"后端",
    r"前端",
    r"客户端",
    r"服务端",
    r"运维",
    r"\bsre\b",
    r"架构",
    r"\bsdk\b",
    r"编译",
    r"网络",
    r"内核",
    r"数据库",
    r"固件",
    r"驱动",
    r"模型",
    r"智能体",
]


def is_tech_like_title(title: str) -> bool:
    normalized = str(title or "").strip().lower()
    return any(re.search(pattern, normalized, flags=re.I) for pattern in TECH_LIKE_TITLE_PATTERNS)


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    filtered = []
    for row in rows:
        if row.get("source") not in HIGH_TRUST_SOURCES:
            continue
        if not is_tech_like_title(row.get("job_title") or ""):
            continue
        reason = classify_rejection(row)
        if reason == "accepted":
            continue
        filtered.append((reason, row))

    reason_counter: Counter[str] = Counter(reason for reason, _ in filtered)
    source_counter: Counter[str] = Counter(str(row.get("source") or "") for _, row in filtered)
    title_counter: Counter[str] = Counter(str(row.get("job_title") or "") for _, row in filtered)
    direction_counter: Counter[str] = Counter(str((row.get("labels") or {}).get("岗位方向") or "<empty>") for _, row in filtered)

    return {
        "total_tech_like_rejected": len(filtered),
        "top_reasons": [{"name": name, "count": count} for name, count in reason_counter.most_common(20)],
        "top_sources": [{"name": name, "count": count} for name, count in source_counter.most_common(20)],
        "top_titles": [{"name": name, "count": count} for name, count in title_counter.most_common(30)],
        "top_directions": [{"name": name, "count": count} for name, count in direction_counter.most_common(20)],
    }


def build_samples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        if row.get("source") not in HIGH_TRUST_SOURCES:
            continue
        if not is_tech_like_title(row.get("job_title") or ""):
            continue
        reason = classify_rejection(row)
        if reason == "accepted":
            continue
        samples.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "job_title": row.get("job_title"),
                "reason": reason,
                "direction": (row.get("labels") or {}).get("岗位方向", ""),
                "experience": (row.get("labels") or {}).get("经验要求", ""),
                "education": (row.get("labels") or {}).get("学历要求", ""),
                "skills": (row.get("labels") or {}).get("必备技能", []),
                "clean_text_preview": str(row.get("clean_text") or "")[:400],
            }
        )
        if len(samples) >= limit:
            break
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/jd_clean_dedup.jsonl")
    parser.add_argument("--out", default="outputs/eval_reports/jd_strict_tech_candidate_report.json")
    parser.add_argument("--sample-out", default="outputs/eval_reports/jd_strict_tech_candidate_samples.jsonl")
    parser.add_argument("--sample-limit", type=int, default=200)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    report = build_report(rows)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")
    write_jsonl(args.sample_out, build_samples(rows, args.sample_limit))


if __name__ == "__main__":
    main()
