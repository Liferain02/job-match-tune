from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_text


READINESS_THRESHOLDS = {
    "jd": {"train": 4000, "valid": 500, "test": 500, "pool": 8000},
    "resume": {"train": 10000, "valid": 1000, "test": 1000, "pool": 3000},
    "match": {"train": 1500, "valid": 200, "test": 200, "pool": 2000},
    "multitask": {"train": 8000, "valid": 1000, "test": 0, "pool": 9000},
}

REQUIRED_FIELDS = {
    "jd": ["岗位方向", "核心职责", "必备技能", "学历要求", "经验要求"],
    "resume": ["目标岗位", "教育背景", "核心技能", "实习经历", "项目经历", "优势标签"],
    "match": ["匹配结论", "匹配优势", "主要短板", "简历优化建议", "推荐投递岗位方向"],
    "multitask": [],
}

MAX_EMPTY_RATE = {
    "jd": {
        "岗位方向": 0.0,
        "核心职责": 0.08,
        "必备技能": 0.30,
        "学历要求": 0.35,
        "经验要求": 0.55,
    },
    "resume": {
        "目标岗位": 0.0,
        "教育背景": 0.0,
        "核心技能": 0.0,
        "实习经历": 0.0,
        "项目经历": 0.0,
        "优势标签": 0.0,
    },
    "match": {
        "匹配结论": 0.0,
        "匹配优势": 0.0,
        "主要短板": 0.0,
        "简历优化建议": 0.0,
        "推荐投递岗位方向": 0.0,
    },
    "multitask": {},
}


def count_jsonl(path: str) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    return sum(1 for _ in read_jsonl(file_path))


def _empty(value: Any) -> bool:
    return value in (None, "", [], {})


def audit_sft_files(task_name: str, paths: list[str]) -> dict[str, Any]:
    total = 0
    invalid_json = 0
    duplicate_ids = 0
    ids: set[str] = set()
    content_seen: dict[str, str] = {}
    cross_split_duplicate_hashes = 0
    split_counts: dict[str, int] = {}
    empty_counts = {field: 0 for field in REQUIRED_FIELDS[task_name]}
    task_types: dict[str, int] = {}

    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        split_name = file_path.stem
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                split_counts[split_name] = split_counts.get(split_name, 0) + 1
                try:
                    row = json.loads(line)
                    row_id = str(row.get("id") or "")
                    if row_id in ids:
                        duplicate_ids += 1
                    ids.add(row_id)
                    task_type = str(row.get("task_type") or "")
                    task_types[task_type] = task_types.get(task_type, 0) + 1
                    messages = row["messages"]
                    user_text = str(messages[1].get("content") or "")
                    assistant = json.loads(row["messages"][-1]["content"])
                    assistant_text = json.dumps(assistant, ensure_ascii=False, sort_keys=True)
                    content_hash = hashlib.sha1(f"{user_text}\n---\n{assistant_text}".encode("utf-8")).hexdigest()
                    previous_split = content_seen.get(content_hash)
                    if previous_split and previous_split != split_name:
                        cross_split_duplicate_hashes += 1
                    content_seen.setdefault(content_hash, split_name)
                except Exception:
                    invalid_json += 1
                    continue
                for field in REQUIRED_FIELDS[task_name]:
                    if _empty(assistant.get(field)):
                        empty_counts[field] += 1

    empty_rates = {
        field: (round(count / total, 4) if total else 1.0)
        for field, count in empty_counts.items()
    }
    field_quality_ok = all(
        empty_rates[field] <= MAX_EMPTY_RATE[task_name][field]
        for field in REQUIRED_FIELDS[task_name]
    )
    return {
        "total": total,
        "invalid_json": invalid_json,
        "duplicate_ids": duplicate_ids,
        "cross_split_duplicate_hashes": cross_split_duplicate_hashes,
        "split_counts": split_counts,
        "task_types": task_types,
        "empty_counts": empty_counts,
        "empty_rates": empty_rates,
        "max_empty_rates": MAX_EMPTY_RATE[task_name],
        "field_quality_ok": field_quality_ok,
    }


def build_multitask_report(train_path: str, valid_path: str) -> dict[str, Any]:
    thresholds = READINESS_THRESHOLDS["multitask"]
    train_count = count_jsonl(train_path)
    valid_count = count_jsonl(valid_path)
    audit = audit_sft_files("multitask", [train_path, valid_path])
    task_mix = {}
    for path in [train_path, valid_path]:
        split = Path(path).stem
        for row in read_jsonl(path):
            task = str((row.get("meta") or {}).get("dataset_task") or row.get("task_type") or "")
            task_mix.setdefault(split, {})
            task_mix[split][task] = task_mix[split].get(task, 0) + 1
    required_tasks = {"jd", "resume", "match"}
    has_required_mix = all(required_tasks.issubset(set(task_mix.get(split, {}))) for split in ("train", "valid"))
    count_ready = train_count >= thresholds["train"] and valid_count >= thresholds["valid"]
    format_ready = (
        audit["invalid_json"] == 0
        and audit["duplicate_ids"] == 0
        and audit["cross_split_duplicate_hashes"] == 0
    )
    ready = count_ready and format_ready and has_required_mix
    return {
        "task": "multitask",
        "counts": {"train": train_count, "valid": valid_count, "test": 0, "combined_pool": train_count + valid_count},
        "thresholds": thresholds,
        "count_ready": count_ready,
        "format_ready": format_ready,
        "task_mix": task_mix,
        "has_required_mix": has_required_mix,
        "quality_audit": audit,
        "ready_for_sft": ready,
    }


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
    audit = audit_sft_files(task_name, [train_path, valid_path, test_path])
    count_ready = (
        train_count >= thresholds["train"]
        and valid_count >= thresholds["valid"]
        and test_count >= thresholds["test"]
        and pool_count >= thresholds["pool"]
    )
    format_ready = (
        audit["invalid_json"] == 0
        and audit["duplicate_ids"] == 0
        and audit["cross_split_duplicate_hashes"] == 0
    )
    ready = count_ready and format_ready and bool(audit["field_quality_ok"])
    return {
        "task": task_name,
        "counts": {
            "train": train_count,
            "valid": valid_count,
            "test": test_count,
            "combined_pool": pool_count,
        },
        "thresholds": thresholds,
        "count_ready": count_ready,
        "format_ready": format_ready,
        "quality_audit": audit,
        "ready_for_sft": ready,
    }


def build_report() -> dict[str, object]:
    tasks = {
        "jd": build_task_report(
            "jd",
            "data/sft_jd_quality/train.jsonl",
            "data/sft_jd_quality/valid.jsonl",
            "data/sft_jd_quality/test.jsonl",
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
        "multitask": build_multitask_report(
            "data/sft_multitask/train.jsonl",
            "data/sft_multitask/valid.jsonl",
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
