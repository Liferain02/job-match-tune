from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.grouped_split import normalized_input_hash
from jobmatch_tune.dataset.pipeline_freshness import build_pipeline_freshness_report
from jobmatch_tune.dataset.templates import resume_parse_prompt
from jobmatch_tune.match.rule_engine import compute_match_rule_result
from jobmatch_tune.match.rule_engine import (
    _extract_required_education_rank,
    _extract_required_years,
)
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
)
from jobmatch_tune.resume.privacy import detect_resume_pii, sanitize_resume_text_for_training
from jobmatch_tune.utils.io import read_jsonl, write_text


READINESS_THRESHOLDS = {
    "jd": {"train": 2560, "valid": 320, "test": 320, "pool": 8000},
    "resume": {"train": 10000, "valid": 1000, "test": 1000, "pool": 3000},
    "match": {"train": 3000, "valid": 400, "test": 400, "pool": 4500},
    "multitask": {"train": 9540, "valid": 1180, "test": 0, "pool": 10720},
}

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
        # Real official postings often omit education entirely. Empty is a
        # valid extraction target and must not be replaced with a guessed degree.
        "学历要求": 0.50,
        # Many official JDs omit experience requirements entirely. Keep empty values
        # instead of fabricating labels, while still tracking the rate explicitly.
        "经验要求": 0.56,
    },
    "resume": {
        "目标岗位": 0.05,
        "教育背景": 0.10,
        "核心技能": 0.10,
        # Experienced candidates commonly have work history but no internship.
        # Keep the field in the schema without fabricating internship content.
        "实习经历": 0.80,
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


def read_match_evaluation_readiness() -> dict[str, Any]:
    pair_report = read_json_file("outputs/eval_reports/match_gold_audit.json")
    ranking_report = read_json_file("outputs/eval_reports/djinni_real_ranking_bm25.json")
    if not ranking_report:
        return pair_report
    expert_ready = bool(ranking_report.get("expert_regression_ready"))
    formal_ready = bool(ranking_report.get("formal_evaluation_ready"))
    return {
        **pair_report,
        "ranking_ready": expert_ready or formal_ready,
        "ranking_formal_ready": formal_ready,
        "ranking_expert_regression_ready": expert_ready,
        "ranking_blockers": ranking_report.get("readiness_blockers") or [],
        "ranking_data_profile": ranking_report.get("data_profile") or {},
        "ranking_metrics": ranking_report.get("metrics") or {},
    }


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


def count_resume_evaluation_overlap(paths: list[str], evaluation_path: str) -> int:
    file_path = Path(evaluation_path)
    if not file_path.exists():
        return 0
    evaluation_hashes = {
        normalized_input_hash(
            resume_parse_prompt(
                sanitize_resume_text_for_training(str(row.get("text") or ""))
            )
        )
        for row in read_jsonl(file_path)
    }
    overlapping_hashes: set[str] = set()
    for path in paths:
        if not Path(path).exists():
            continue
        for row in read_jsonl(path):
            messages = row.get("messages") or []
            if len(messages) < 2:
                continue
            prompt_hash = normalized_input_hash(str(messages[1].get("content") or ""))
            if prompt_hash in evaluation_hashes:
                overlapping_hashes.add(prompt_hash)
    return len(overlapping_hashes)


def profile_match_training_sources(path: str) -> dict[str, Any]:
    source_types: Counter[str] = Counter()
    generators: Counter[str] = Counter()
    total = 0
    synthetic_rows = 0
    curated_fictional_rows = 0
    verified_non_synthetic_rows = 0
    educational_source_rows = 0
    pair_type_counts: Counter[str] = Counter()
    file_path = Path(path)
    if file_path.exists():
        for row in read_jsonl(file_path):
            total += 1
            educational_source_rows += int(
                "synthetic_match_hf_job_educational_" in str(row.get("id") or "")
            )
            source_type = str(row.get("source_type") or "missing")
            generator = str((row.get("meta") or {}).get("generator") or "missing")
            source_types[source_type] += 1
            generators[generator] += 1
            if source_type.startswith("synthetic") or generator.startswith("synthetic"):
                synthetic_rows += 1
                pair_type_counts["synthetic_rule_pair"] += 1
            elif source_type == "curated_fictional_pair":
                curated_fictional_rows += 1
                pair_type_counts["curated_fictional_pair"] += 1
            elif str((row.get("meta") or {}).get("annotation_status") or "") == "human_verified":
                verified_non_synthetic_rows += 1
                pair_type_counts["human_reviewed_pair"] += 1
            elif str((row.get("meta") or {}).get("pair_type") or "") == "real_observed_pair":
                pair_type_counts["real_observed_pair"] += 1
            else:
                pair_type_counts["unknown_non_synthetic_pair"] += 1
    non_synthetic_rows = total - synthetic_rows - curated_fictional_rows
    if not total:
        origin = "empty"
    elif synthetic_rows == total:
        origin = "synthetic_only"
    elif curated_fictional_rows == total:
        origin = "curated_fictional_only"
    elif synthetic_rows:
        origin = "mixed"
    else:
        origin = "non_synthetic_only"
    return {
        "total_rows": total,
        "source_type_counts": dict(source_types),
        "generator_counts": dict(generators),
        "synthetic_rows": synthetic_rows,
        "curated_fictional_rows": curated_fictional_rows,
        "non_synthetic_rows": non_synthetic_rows,
        "human_verified_non_synthetic_rows": verified_non_synthetic_rows,
        "pair_type_counts": {
            "synthetic_rule_pair": pair_type_counts.get("synthetic_rule_pair", 0),
            "human_reviewed_pair": pair_type_counts.get("human_reviewed_pair", 0),
            "real_observed_pair": pair_type_counts.get("real_observed_pair", 0),
            "unknown_non_synthetic_pair": pair_type_counts.get(
                "unknown_non_synthetic_pair", 0
            ),
        },
        "synthetic_rate": round(synthetic_rows / total, 4) if total else 0.0,
        "curated_fictional_pair_ratio": (
            round(curated_fictional_rows / total, 4) if total else 0.0
        ),
        "human_reviewed_pair_ratio": (
            round(pair_type_counts.get("human_reviewed_pair", 0) / total, 4)
            if total
            else 0.0
        ),
        "real_pair_ratio": (
            round(pair_type_counts.get("real_observed_pair", 0) / total, 4)
            if total
            else 0.0
        ),
        "educational_source_rows": educational_source_rows,
        "educational_source_rate": (
            round(educational_source_rows / total, 4) if total else 0.0
        ),
        "max_educational_source_rate": 0.4,
        "source_concentration_ready": bool(
            total and educational_source_rows / total <= 0.4
        ),
        "training_origin": origin,
        "supports_real_pair_quality_claim": verified_non_synthetic_rows > 0,
    }


def profile_match_training_privacy(path: str) -> dict[str, Any]:
    total = 0
    rows_with_findings = 0
    finding_counts: Counter[str] = Counter()
    file_path = Path(path)
    if file_path.exists():
        for row in read_jsonl(file_path):
            total += 1
            findings = detect_resume_pii(str(row.get("resume_text") or ""))
            if findings:
                rows_with_findings += 1
                finding_counts.update(finding.kind for finding in findings)
    return {
        "total_rows": total,
        "rows_with_privacy_findings": rows_with_findings,
        "privacy_finding_rate": round(rows_with_findings / total, 4) if total else 0.0,
        "finding_counts": dict(finding_counts),
        "privacy_ready": total > 0 and rows_with_findings == 0,
    }


def profile_match_education_consistency(path: str) -> dict[str, Any]:
    total = 0
    disagreement_rows = 0
    examples = []
    file_path = Path(path)
    if file_path.exists():
        for row in read_jsonl(file_path):
            total += 1
            requirement = extract_education_requirement(str(row.get("jd_text") or ""))
            result = compute_match_rule_result(
                {"学历要求": requirement},
                {"教育背景": [str(row.get("resume_text") or "")]},
                resume_text=str(row.get("resume_text") or ""),
            )
            actual = bool((row.get("label") or {}).get("学历匹配"))
            expected = bool(result["学历匹配"])
            if actual != expected:
                disagreement_rows += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "id": row.get("id"),
                            "requirement": requirement,
                            "label": actual,
                            "expected": expected,
                        }
                    )
    return {
        "total_rows": total,
        "disagreement_rows": disagreement_rows,
        "examples": examples,
        "education_consistency_ready": total > 0 and disagreement_rows == 0,
    }


def profile_match_education_distribution(path: str) -> dict[str, Any]:
    total = 0
    required_rows = 0
    matched_rows = 0
    requirement_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = {}
    rank_names = {1: "中专", 2: "大专", 3: "本科", 4: "硕士/研究生", 5: "博士"}
    file_path = Path(path)
    if file_path.exists():
        for row in read_jsonl(file_path):
            total += 1
            requirement = extract_education_requirement(str(row.get("jd_text") or ""))
            rank = _extract_required_education_rank(requirement)
            if rank <= 0:
                continue
            required_rows += 1
            requirement_counts[rank_names.get(rank, f"rank_{rank}")] += 1
            matched = bool((row.get("label") or {}).get("学历匹配"))
            matched_rows += int(matched)
            split = str((row.get("meta") or {}).get("entity_split") or "missing")
            counts = split_counts.setdefault(split, Counter())
            counts["required"] += 1
            counts["matched"] += int(matched)
            counts["unmatched"] += int(not matched)
    return {
        "total_rows": total,
        "explicit_education_requirement_count": required_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": required_rows - matched_rows,
        "matched_rate": round(matched_rows / required_rows, 4) if required_rows else 0.0,
        "requirement_level_counts": dict(requirement_counts),
        "split_counts": {
            split: dict(counts) for split, counts in split_counts.items()
        },
    }


def profile_match_experience_distribution(path: str) -> dict[str, Any]:
    total = 0
    required_rows = 0
    matched_rows = 0
    split_counts: dict[str, Counter[str]] = {}
    threshold_counts: Counter[str] = Counter()
    file_path = Path(path)
    if file_path.exists():
        for row in read_jsonl(file_path):
            total += 1
            requirement = extract_experience_requirement(str(row.get("jd_text") or ""))
            required_years = _extract_required_years(requirement)
            if required_years <= 0:
                continue
            required_rows += 1
            threshold_counts[f"{required_years}年"] += 1
            matched = bool((row.get("label") or {}).get("经验匹配"))
            matched_rows += int(matched)
            split = str((row.get("meta") or {}).get("entity_split") or "missing")
            counts = split_counts.setdefault(split, Counter())
            counts["required"] += 1
            counts["matched"] += int(matched)
            counts["unmatched"] += int(not matched)
    rendered_splits = {split: dict(counts) for split, counts in split_counts.items()}
    covered_splits = bool(rendered_splits) and all(
        counts.get("matched", 0) > 0 and counts.get("unmatched", 0) > 0
        for counts in rendered_splits.values()
    )
    return {
        "total_rows": total,
        "explicit_experience_requirement_count": required_rows,
        "required_rows": required_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": required_rows - matched_rows,
        "positive_rate": round(matched_rows / required_rows, 4) if required_rows else 0.0,
        "matched_rate": round(matched_rows / required_rows, 4) if required_rows else 0.0,
        "threshold_counts": dict(threshold_counts),
        "split_counts": rendered_splits,
        "experience_distribution_ready": required_rows > 0 and covered_splits,
    }


def audit_sft_files(task_name: str, paths: list[str]) -> dict[str, Any]:
    total = 0
    invalid_json = 0
    duplicate_ids = 0
    ids: set[str] = set()
    content_seen: dict[str, str] = {}
    normalized_input_seen: dict[str, str] = {}
    linked_group_seen: dict[str, str] = {}
    cross_split_linked_groups: set[str] = set()
    cross_split_duplicate_hashes = 0
    cross_split_normalized_input_hashes = 0
    split_counts: dict[str, int] = {}
    empty_counts = {field: 0 for field in REQUIRED_FIELDS[task_name]}
    task_types: dict[str, int] = {}
    placeholder_match_recommendation_rows = 0
    match_rows_missing_project_evidence_field = 0
    placeholder_recommendations = {
        "当前 JD 对应方向",
        "同方向相近岗位",
        "技能相近岗位",
        "技能相近或低门槛过渡岗位",
    }

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
                    if task_type == "match" and any(
                        str(role) in placeholder_recommendations
                        for role in assistant.get("推荐投递岗位方向") or []
                    ):
                        placeholder_match_recommendation_rows += 1
                    if task_type == "match" and '"命中项目"' not in user_text:
                        match_rows_missing_project_evidence_field += 1
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
                    for linked_group in row.get("linked_source_groups") or []:
                        linked_group = str(linked_group)
                        previous_linked_split = linked_group_seen.get(linked_group)
                        if previous_linked_split and previous_linked_split != split_name:
                            cross_split_linked_groups.add(linked_group)
                        linked_group_seen.setdefault(linked_group, split_name)
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
        "cross_split_linked_source_groups": len(cross_split_linked_groups),
        "cross_split_linked_source_group_types": dict(
            Counter(group.split(":", 1)[0] for group in cross_split_linked_groups)
        ),
        "split_counts": split_counts,
        "task_types": task_types,
        "placeholder_match_recommendation_rows": placeholder_match_recommendation_rows,
        "match_rows_missing_project_evidence_field": match_rows_missing_project_evidence_field,
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
        and audit.get("cross_split_linked_source_groups", 0) == 0
        and audit.get("placeholder_match_recommendation_rows", 0) == 0
        and audit.get("match_rows_missing_project_evidence_field", 0) == 0
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
        and audit.get("cross_split_linked_source_groups", 0) == 0
        and audit.get("placeholder_match_recommendation_rows", 0) == 0
        and audit.get("match_rows_missing_project_evidence_field", 0) == 0
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
    match_evaluation = read_match_evaluation_readiness()
    tasks = {
        "jd": build_task_report(
            "jd",
            "data/sft_jd_strict_plus/train.jsonl",
            "data/sft_jd_strict_plus/valid.jsonl",
            "data/sft_jd_strict_plus/test.jsonl",
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
    match_source_profile = profile_match_training_sources(
        "data/eval/match_train_pool_combined.jsonl"
    )
    tasks["match"]["source_profile"] = match_source_profile
    tasks["match"]["real_pair_quality_evidence_ready"] = bool(
        match_source_profile["supports_real_pair_quality_claim"]
    )
    tasks["match"]["source_concentration_ready"] = bool(
        match_source_profile["source_concentration_ready"]
    )
    match_privacy_profile = profile_match_training_privacy(
        "data/eval/match_train_pool_combined.jsonl"
    )
    tasks["match"]["privacy_profile"] = match_privacy_profile
    tasks["match"]["privacy_ready"] = bool(
        match_privacy_profile["privacy_ready"] or match_privacy_profile["total_rows"] == 0
    )
    match_education_profile = profile_match_education_consistency(
        "data/eval/match_train_pool_combined.jsonl"
    )
    tasks["match"]["education_consistency_profile"] = match_education_profile
    tasks["match"]["education_consistency_ready"] = bool(
        match_education_profile["education_consistency_ready"]
        or match_education_profile["total_rows"] == 0
    )
    tasks["match"]["education_distribution_profile"] = (
        profile_match_education_distribution(
            "data/eval/match_train_pool_combined.jsonl"
        )
    )
    match_experience_profile = profile_match_experience_distribution(
        "data/eval/match_train_pool_combined.jsonl"
    )
    tasks["match"]["experience_distribution_profile"] = match_experience_profile
    tasks["match"]["experience_distribution_ready"] = bool(
        match_experience_profile["experience_distribution_ready"]
    )
    tasks["match"]["ready_for_sft"] = bool(
        tasks["match"]["ready_for_sft"]
        and tasks["match"]["privacy_ready"]
        and tasks["match"]["source_concentration_ready"]
        and tasks["match"]["education_consistency_ready"]
        and tasks["match"]["experience_distribution_ready"]
        and tasks["match"]["real_pair_quality_evidence_ready"]
    )
    jd_holdout_overlap = count_holdout_overlap(
        [
            "data/sft_jd_strict_plus/train.jsonl",
            "data/sft_jd_strict_plus/valid.jsonl",
            "data/sft_jd_strict_plus/test.jsonl",
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
    resume_evaluation_overlap = count_resume_evaluation_overlap(
        [
            "data/sft_resume/train.jsonl",
            "data/sft_resume/valid.jsonl",
            "data/sft_resume/test.jsonl",
        ],
        "data/eval/resume_manual_eval_text_seed.jsonl",
    )
    tasks["resume"]["evaluation_overlap_count"] = resume_evaluation_overlap
    tasks["resume"]["evaluation_overlap_ready"] = resume_evaluation_overlap == 0
    tasks["resume"]["ready_for_sft"] = bool(
        tasks["resume"]["ready_for_sft"]
        and tasks["resume"]["evaluation_overlap_ready"]
    )

    def split_leakage_ready(task: dict[str, Any]) -> bool:
        audit = task.get("quality_audit") or {}
        return all(
            int(audit.get(key, 0)) == 0
            for key in (
                "cross_split_duplicate_hashes",
                "cross_split_normalized_input_hashes",
                "cross_split_linked_source_groups",
            )
        )

    # License/provenance describe the rows that actually entered the current
    # training pools. Candidate public resume sources remain outside those pools
    # until source admission explicitly marks them training_allowed.
    tasks["resume"]["readiness_dimensions"] = {
        "format_ready": bool(tasks["resume"]["format_ready"]),
        "privacy_ready": bool(tasks["resume"].get("privacy_ready", False)),
        "license_ready": True,
        "provenance_ready": True,
        "source_diversity_ready": bool(tasks["resume"].get("profile_ready", False)),
        "condition_distribution_ready": None,
        "real_pair_quality_evidence_ready": None,
        "split_leakage_ready": split_leakage_ready(tasks["resume"]),
        "evaluation_overlap_ready": bool(tasks["resume"]["evaluation_overlap_ready"]),
    }
    tasks["match"]["readiness_dimensions"] = {
        "format_ready": bool(tasks["match"]["format_ready"]),
        "privacy_ready": bool(tasks["match"]["privacy_ready"]),
        "license_ready": True,
        "provenance_ready": True,
        "source_diversity_ready": bool(tasks["match"]["source_concentration_ready"]),
        "condition_distribution_ready": bool(
            tasks["match"]["education_consistency_ready"]
            and tasks["match"]["experience_distribution_ready"]
        ),
        "real_pair_quality_evidence_ready": bool(
            tasks["match"]["real_pair_quality_evidence_ready"]
        ),
        "split_leakage_ready": split_leakage_ready(tasks["match"]),
    }
    tasks["multitask"]["readiness_dimensions"] = {
        "format_ready": bool(tasks["multitask"].get("format_ready", True)),
        "privacy_ready": bool(
            tasks["resume"].get("privacy_ready", False)
            and tasks["match"]["privacy_ready"]
        ),
        "license_ready": True,
        "provenance_ready": True,
        "source_diversity_ready": bool(
            tasks["multitask"].get("source_diversity_ready", True)
        ),
        "condition_distribution_ready": bool(
            tasks["match"]["education_consistency_ready"]
            and tasks["match"]["experience_distribution_ready"]
        ),
        "real_pair_quality_evidence_ready": bool(
            tasks["match"]["real_pair_quality_evidence_ready"]
        ),
        "split_leakage_ready": split_leakage_ready(tasks["multitask"]),
    }
    preference_report = read_json_file("outputs/eval_reports/preference_readiness_report.json")
    sft_pipeline_fresh = bool(pipeline_freshness.get("sft_fresh", pipeline_freshness["fresh"]))
    dpo_pipeline_fresh = bool(pipeline_freshness.get("dpo_fresh", pipeline_freshness["fresh"]))
    all_ready_for_sft = bool(
        all(task["ready_for_sft"] for task in tasks.values()) and sft_pipeline_fresh
    )
    ready_for_sft_experiment = bool(
        sft_pipeline_fresh
        and all(
            task.get("format_ready", True)
            and int((task.get("counts") or {}).get("train", 0)) > 0
            and int((task.get("counts") or {}).get("valid", 0)) > 0
            for task in tasks.values()
        )
        and tasks["resume"].get("privacy_ready", False)
        and tasks["resume"].get("evaluation_overlap_ready", False)
        and tasks["match"].get("privacy_ready", False)
        and tasks["jd"].get("holdout_ready", False)
        and tasks["multitask"].get("has_required_mix", True)
    )
    ready_for_dpo = bool(preference_report.get("ready_for_dpo"))
    ready_for_dpo_smoke = bool(preference_report.get("ready_for_dpo_smoke"))
    ready_for_dpo_experiment = bool(preference_report.get("ready_for_dpo_experiment"))
    dpo_paused = os.environ.get("JOBMATCH_ALLOW_DPO", "0") != "1"
    all_ready_for_training = (
        all_ready_for_sft
        and ready_for_dpo
        and bool(pipeline_freshness["fresh"])
    )
    return {
        "summary": {
            "all_ready_for_training": all_ready_for_training,
            "all_ready_for_sft": all_ready_for_sft,
            "ready_for_sft_experiment": ready_for_sft_experiment,
            "ready_for_dpo_smoke": ready_for_dpo_smoke,
            "ready_for_dpo_experiment": ready_for_dpo_experiment,
            "ready_for_dpo": ready_for_dpo,
            "dpo_paused_by_quality_goal": dpo_paused,
            "dpo_execution_ready": bool(
                not dpo_paused and ready_for_dpo
            ),
            "not_ready_tasks": [name for name, task in tasks.items() if not task["ready_for_sft"]],
            "pipeline_fresh": pipeline_freshness["fresh"],
            "sft_pipeline_fresh": sft_pipeline_fresh,
            "dpo_pipeline_fresh": dpo_pipeline_fresh,
            "match_real_pair_quality_evidence_ready": tasks["match"][
                "real_pair_quality_evidence_ready"
            ],
            "match_regression_evaluation_ready": bool(
                match_evaluation.get("regression_ready")
            ),
            "match_blind_evaluation_ready": bool(match_evaluation.get("blind_ready")),
            "match_level_decision_ready": bool(match_evaluation.get("decision_ready")),
            "match_ranking_evaluation_ready": bool(match_evaluation.get("ranking_ready")),
            "match_ranking_formal_evaluation_ready": bool(
                match_evaluation.get("ranking_formal_ready")
            ),
        },
        "pipeline_freshness": pipeline_freshness,
        "tasks": tasks,
        "preference": preference_report,
        "match_evaluation": match_evaluation,
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
