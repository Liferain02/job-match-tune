from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.grouped_split import normalized_input_hash
from jobmatch_tune.dataset.pipeline_freshness import build_pipeline_freshness_report
from jobmatch_tune.utils.io import read_jsonl, write_text


READINESS_THRESHOLDS = {
    "jd": {"train": 4240, "valid": 530, "test": 530, "pool": 8000},
    "resume": {"train": 10000, "valid": 1000, "test": 1000, "pool": 3000},
    "match": {"train": 3000, "valid": 400, "test": 400, "pool": 4500},
    "multitask": {"train": 9540, "valid": 1180, "test": 0, "pool": 10720},
}

MAX_JD_HIGH_RISK_RATE = 0.05
MIN_MULTITASK_SOURCE_GROUP_RATIO = {"jd": 0.95, "resume": 0.8, "match": 0.8}

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
        # Many official JDs omit experience requirements entirely. Keep empty values
        # instead of fabricating labels, while still tracking the rate explicitly.
        "经验要求": 0.56,
    },
    "resume": {
        "目标岗位": 0.05,
        "教育背景": 0.10,
        "核心技能": 0.10,
        "实习经历": 0.50,
        "项目经历": 0.20,
        "优势标签": 0.40,
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


def read_json_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _normalized_source_id(row: dict[str, Any]) -> str:
    return str(row.get("source_id") or row.get("id") or "").removesuffix("_jd_parse")


def count_holdout_overlap(paths: list[str], holdout_path: str) -> int:
    file_path = Path(holdout_path)
    if not file_path.exists():
        return 0
    holdout_ids = {_normalized_source_id(row) for row in read_jsonl(file_path)}
    overlap = 0
    for path in paths:
        if not Path(path).exists():
            continue
        overlap += sum(1 for row in read_jsonl(path) if _normalized_source_id(row) in holdout_ids)
    return overlap


def audit_sft_files(task_name: str, paths: list[str]) -> dict[str, Any]:
    total = 0
    invalid_json = 0
    duplicate_ids = 0
    ids: set[str] = set()
    content_seen: dict[str, str] = {}
    normalized_input_seen: dict[str, str] = {}
    cross_split_duplicate_hashes = 0
    cross_split_normalized_input_hashes = 0
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
                    input_hash = normalized_input_hash(user_text)
                    previous_input_split = normalized_input_seen.get(input_hash)
                    if previous_input_split and previous_input_split != split_name:
                        cross_split_normalized_input_hashes += 1
                    normalized_input_seen.setdefault(input_hash, split_name)
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
        "cross_split_normalized_input_hashes": cross_split_normalized_input_hashes,
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
    source_groups: dict[str, dict[str, set[str]]] = {}
    group_splits: dict[str, dict[str, set[str]]] = {}
    for path in [train_path, valid_path]:
        split = Path(path).stem
        for row in read_jsonl(path):
            task = str((row.get("meta") or {}).get("dataset_task") or row.get("task_type") or "")
            task_mix.setdefault(split, {})
            task_mix[split][task] = task_mix[split].get(task, 0) + 1
            source_group = str(
                row.get("source_group") or row.get("source_id") or row.get("id") or ""
            )
            source_groups.setdefault(split, {}).setdefault(task, set()).add(source_group)
            group_splits.setdefault(task, {}).setdefault(source_group, set()).add(split)
    required_tasks = {"jd", "resume", "match"}
    has_required_mix = all(required_tasks.issubset(set(task_mix.get(split, {}))) for split in ("train", "valid"))
    source_diversity = {}
    for split, task_counts in task_mix.items():
        source_diversity[split] = {}
        for task, row_count in task_counts.items():
            unique_groups = len(source_groups.get(split, {}).get(task, set()))
            source_diversity[split][task] = {
                "rows": row_count,
                "unique_source_groups": unique_groups,
                "source_group_ratio": round(unique_groups / row_count, 4) if row_count else 0.0,
                "minimum_ratio": MIN_MULTITASK_SOURCE_GROUP_RATIO.get(task, 0.0),
            }
    source_diversity_ready = all(
        source_diversity.get(split, {}).get(task, {}).get("source_group_ratio", 0.0)
        >= MIN_MULTITASK_SOURCE_GROUP_RATIO[task]
        for split in ("train", "valid")
        for task in required_tasks
    )
    cross_split_source_groups = sum(
        1
        for task_groups in group_splits.values()
        for splits in task_groups.values()
        if len(splits) > 1
    )
    count_ready = train_count >= thresholds["train"] and valid_count >= thresholds["valid"]
    format_ready = (
        audit["invalid_json"] == 0
        and audit["duplicate_ids"] == 0
        and audit["cross_split_duplicate_hashes"] == 0
        and audit.get("cross_split_normalized_input_hashes", 0) == 0
        and cross_split_source_groups == 0
    )
    ready = count_ready and format_ready and has_required_mix and source_diversity_ready
    return {
        "task": "multitask",
        "counts": {"train": train_count, "valid": valid_count, "test": 0, "combined_pool": train_count + valid_count},
        "thresholds": thresholds,
        "count_ready": count_ready,
        "format_ready": format_ready,
        "task_mix": task_mix,
        "has_required_mix": has_required_mix,
        "source_diversity": source_diversity,
        "source_diversity_ready": source_diversity_ready,
        "cross_split_source_groups": cross_split_source_groups,
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
        and audit["cross_split_duplicate_hashes"] == 0
        and audit.get("cross_split_normalized_input_hashes", 0) == 0
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
    pipeline_freshness = build_pipeline_freshness_report()
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
    jd_quality_profile = read_json_file("outputs/eval_reports/jd_quality_profile.json")
    if jd_quality_profile:
        tasks["jd"]["quality_profile"] = jd_quality_profile
    jd_risk_report = read_json_file("outputs/eval_reports/jd_quality_risk_report.json")
    if jd_risk_report:
        high_risk_rate = _float_or_default(jd_risk_report.get("high_risk_rate"), 1.0)
        tasks["jd"]["risk_report"] = jd_risk_report
        tasks["jd"]["risk_ready"] = high_risk_rate <= MAX_JD_HIGH_RISK_RATE
        tasks["jd"]["ready_for_sft"] = bool(tasks["jd"]["ready_for_sft"] and tasks["jd"]["risk_ready"])
    jd_direction_conflicts = read_json_file("outputs/eval_reports/jd_direction_conflicts.json")
    if jd_direction_conflicts:
        tasks["jd"]["direction_conflict_audit"] = jd_direction_conflicts
    jd_experience_gaps = read_json_file("outputs/eval_reports/jd_experience_gaps.json")
    if jd_experience_gaps:
        tasks["jd"]["experience_gap_audit"] = jd_experience_gaps
    jd_holdout_overlap = count_holdout_overlap(
        [
            "data/sft_jd_quality/train.jsonl",
            "data/sft_jd_quality/valid.jsonl",
            "data/sft_jd_quality/test.jsonl",
        ],
        "data/eval/jd_manual_eval_50.jsonl",
    )
    tasks["jd"]["holdout_overlap"] = jd_holdout_overlap
    tasks["jd"]["holdout_ready"] = jd_holdout_overlap == 0
    tasks["jd"]["ready_for_sft"] = bool(tasks["jd"]["ready_for_sft"] and tasks["jd"]["holdout_ready"])
    resume_sft_profile = read_json_file("outputs/eval_reports/resume_sft_profile.json")
    if resume_sft_profile:
        tasks["resume"]["sft_profile"] = resume_sft_profile
        tasks["resume"]["profile_ready"] = bool(resume_sft_profile.get("profile_ready"))
        tasks["resume"]["ready_for_sft"] = bool(tasks["resume"]["ready_for_sft"] and tasks["resume"]["profile_ready"])
    resume_privacy_report = read_json_file("outputs/eval_reports/resume_privacy_readiness_report.json")
    if resume_privacy_report:
        tasks["resume"]["privacy_report"] = resume_privacy_report
        tasks["resume"]["privacy_ready"] = bool(resume_privacy_report.get("ready_for_resume_training"))
        tasks["resume"]["ready_for_sft"] = bool(tasks["resume"]["ready_for_sft"] and tasks["resume"]["privacy_ready"])
    preference_report = read_json_file("outputs/eval_reports/preference_readiness_report.json")
    product_preference_report = read_json_file(
        "outputs/eval_reports/preference_product_bootstrap_readiness_report.json"
    )
    all_ready_for_sft = all(task["ready_for_sft"] for task in tasks.values())
    ready_for_dpo = bool(preference_report.get("ready_for_dpo"))
    ready_for_dpo_smoke = bool(preference_report.get("ready_for_dpo_smoke"))
    ready_for_product_dpo = bool(product_preference_report.get("ready_for_dpo"))
    ready_for_product_dpo_smoke = bool(product_preference_report.get("ready_for_dpo_smoke"))
    all_ready_for_training = (
        all_ready_for_sft
        and ready_for_dpo
        and ready_for_product_dpo
        and bool(pipeline_freshness["fresh"])
    )
    return {
        "summary": {
            "all_ready_for_training": all_ready_for_training,
            "all_ready_for_sft": all_ready_for_sft,
            "ready_for_dpo_smoke": ready_for_dpo_smoke,
            "ready_for_dpo": ready_for_dpo,
            "ready_for_product_dpo_smoke": ready_for_product_dpo_smoke,
            "ready_for_product_dpo": ready_for_product_dpo,
            "not_ready_tasks": [name for name, task in tasks.items() if not task["ready_for_sft"]],
            "pipeline_fresh": pipeline_freshness["fresh"],
        },
        "pipeline_freshness": pipeline_freshness,
        "tasks": tasks,
        "preference": preference_report,
        "product_preference": product_preference_report,
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
