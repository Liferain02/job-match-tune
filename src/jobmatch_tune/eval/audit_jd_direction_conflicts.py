from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_text


TITLE_SIGNALS = (
    ("前端开发", re.compile(r"前端|web前端|frontend", re.I)),
    ("后端开发", re.compile(r"后端|后台开发|服务端|backend|java开发|golang开发|python开发", re.I)),
    ("测试开发", re.compile(r"测试|\bqa\b", re.I)),
    ("运维开发", re.compile(r"运维|devops|\bsre\b", re.I)),
    ("嵌入式开发", re.compile(r"嵌入式|固件|bsp|驱动开发", re.I)),
    ("安全工程", re.compile(r"安全工程|网络安全|信息安全|渗透测试", re.I)),
    ("数据开发", re.compile(r"数据开发|大数据开发|数仓|etl", re.I)),
    ("客户端开发", re.compile(r"客户端|android|ios", re.I)),
    ("算法工程", re.compile(r"算法|机器学习|深度学习|自然语言处理|计算机视觉", re.I)),
)


def extract_title(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    user_text = str(messages[1].get("content") or "") if len(messages) > 1 else ""
    match = re.search(r"岗位名称[：:]\s*([^\n]+)", user_text)
    return match.group(1).strip() if match else ""


def expected_direction(title: str) -> str:
    directions = {direction for direction, pattern in TITLE_SIGNALS if pattern.search(title)}
    return next(iter(directions)) if len(directions) == 1 else ""


def audit_direction_conflicts(rows: list[dict[str, Any]], *, sample_limit: int = 100) -> dict[str, Any]:
    signaled_rows = 0
    conflicts: list[dict[str, Any]] = []
    distribution: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        title = extract_title(row)
        expected = expected_direction(title)
        if not expected:
            continue
        signaled_rows += 1
        try:
            actual = str(json.loads(row["messages"][-1]["content"]).get("岗位方向") or "")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            actual = "<invalid>"
        if actual == expected:
            continue
        meta = row.get("meta") or {}
        item = {
            "id": str(row.get("id") or ""),
            "title": title,
            "expected_from_title": expected,
            "actual": actual,
            "quality_tier": str(meta.get("quality_tier") or "unknown"),
            "source": str(meta.get("source") or "unknown"),
        }
        conflicts.append(item)
        distribution[(expected, actual, item["quality_tier"], item["source"])] += 1
    return {
        "total_rows": len(rows),
        "single_strong_title_signal_rows": signaled_rows,
        "conflict_rows": len(conflicts),
        "conflict_rate": round(len(conflicts) / signaled_rows, 4) if signaled_rows else 0.0,
        "interpretation": "review_candidates_not_automatic_errors",
        "distribution": [
            {
                "expected_from_title": key[0],
                "actual": key[1],
                "quality_tier": key[2],
                "source": key[3],
                "count": count,
            }
            for key, count in distribution.most_common()
        ],
        "samples": conflicts[:sample_limit],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 JD 标题强信号与方向标签冲突")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/sft_jd_quality/train.jsonl",
            "data/sft_jd_quality/valid.jsonl",
            "data/sft_jd_quality/test.jsonl",
        ],
    )
    parser.add_argument("--out", default="outputs/eval_reports/jd_direction_conflicts.json")
    parser.add_argument("--sample-limit", type=int, default=100)
    args = parser.parse_args()

    rows = [row for path in args.inputs for row in read_jsonl(path)]
    report = audit_direction_conflicts(rows, sample_limit=args.sample_limit)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
