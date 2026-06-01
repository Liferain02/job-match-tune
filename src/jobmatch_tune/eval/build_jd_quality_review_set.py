from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def collect_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def build_review_rows(
    rows: list[dict[str, Any]],
    *,
    per_tier: int,
    seed: int,
    strategy: str = "balanced",
) -> list[dict[str, Any]]:
    if strategy not in {"balanced", "lowest-score"}:
        raise ValueError(f"Unsupported review sampling strategy: {strategy}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tier = str((row.get("meta") or {}).get("quality_tier") or "unknown")
        grouped[tier].append(row)

    rng = random.Random(seed)
    review_rows = []
    for tier in sorted(grouped):
        candidates = grouped[tier][:]
        if strategy == "lowest-score":
            candidates.sort(
                key=lambda row: (
                    int((row.get("meta") or {}).get("quality_score") or 0),
                    -int((row.get("meta") or {}).get("quality_risk_score") or 0),
                    str(row.get("id") or ""),
                )
            )
        else:
            rng.shuffle(candidates)
        for row in candidates[:per_tier]:
            meta = row.get("meta") or {}
            review_rows.append(
                {
                    "id": row["id"],
                    "quality_tier": tier,
                    "quality_reason": meta.get("quality_reason", ""),
                    "quality_score": meta.get("quality_score"),
                    "quality_risk_score": meta.get("quality_risk_score"),
                    "quality_risk_reasons": meta.get("quality_risk_reasons", []),
                    "task_type": row.get("task_type", ""),
                    "prompt": row["messages"][1]["content"],
                    "assistant": row["messages"][-1]["content"],
                    "review": {
                        "岗位方向": "",
                        "核心职责": "",
                        "必备技能": "",
                        "学历要求": "",
                        "经验要求": "",
                        "notes": "",
                    },
                }
            )
    return review_rows


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
    parser.add_argument("--out", default="data/eval/jd_quality_review_seed.jsonl")
    parser.add_argument("--per-tier", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=["balanced", "lowest-score"],
        default="balanced",
        help="balanced samples randomly per tier; lowest-score prioritizes low quality_score rows per tier.",
    )
    args = parser.parse_args()

    rows = collect_rows(args.inputs)
    review_rows = build_review_rows(rows, per_tier=args.per_tier, seed=args.seed, strategy=args.strategy)
    write_jsonl(args.out, review_rows)
    print(f"wrote {len(review_rows)} review rows to {args.out}")


if __name__ == "__main__":
    main()
