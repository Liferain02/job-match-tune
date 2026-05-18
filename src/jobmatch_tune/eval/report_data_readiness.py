from __future__ import annotations

import argparse
import json
from pathlib import Path

from jobmatch_tune.utils.io import read_jsonl, write_text


READINESS_THRESHOLDS = {
    "jd": {"train": 5000, "valid": 500, "test": 500, "pool": 8000},
    "resume": {"train": 2000, "valid": 200, "test": 200, "pool": 3000},
    "match": {"train": 500, "valid": 100, "test": 100, "pool": 1000},
}


def count_jsonl(path: str) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    return sum(1 for _ in read_jsonl(file_path))


def build_task_report(
    task_name: str,
    train_path: str,
    valid_path: str,
    test_path: str,
    pool_path: str,
) -> dict[str, object]:
    thresholds = READINESS_THRESHOLDS[task_name]
    train_count = count_jsonl(train_path)
    valid_count = count_jsonl(valid_path)
    test_count = count_jsonl(test_path)
    pool_count = count_jsonl(pool_path)
    ready = (
        train_count >= thresholds["train"]
        and valid_count >= thresholds["valid"]
        and test_count >= thresholds["test"]
        and pool_count >= thresholds["pool"]
    )
    return {
        "task": task_name,
        "counts": {
            "train": train_count,
            "valid": valid_count,
            "test": test_count,
            "combined_pool": pool_count,
        },
        "thresholds": thresholds,
        "ready_for_sft": ready,
    }


def build_report() -> dict[str, object]:
    tasks = {
        "jd": build_task_report(
            "jd",
            "data/sft/train.jsonl",
            "data/sft/valid.jsonl",
            "data/sft/test.jsonl",
            "data/eval/jd_train_pool_combined.jsonl",
        ),
        "resume": build_task_report(
            "resume",
            "data/sft_resume/train.jsonl",
            "data/sft_resume/valid.jsonl",
            "data/sft_resume/test.jsonl",
            "data/eval/resume_train_pool_combined.jsonl",
        ),
        "match": build_task_report(
            "match",
            "data/sft_match/train.jsonl",
            "data/sft_match/valid.jsonl",
            "data/sft_match/test.jsonl",
            "data/eval/match_train_pool_combined.jsonl",
        ),
    }
    return {
        "summary": {
            "all_ready_for_training": all(task["ready_for_sft"] for task in tasks.values()),
            "not_ready_tasks": [name for name, task in tasks.items() if not task["ready_for_sft"]],
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/eval_reports/data_readiness_report.json",
    )
    args = parser.parse_args()

    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
