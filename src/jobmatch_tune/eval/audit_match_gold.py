from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jobmatch_tune.dataset.grouped_split import normalized_input_hash
from jobmatch_tune.utils.io import read_jsonl, write_text


DIFFICULTY_TAGS = {
    "技能同义词",
    "可迁移技能",
    "相近岗位方向",
    "年限不足但项目强",
    "学历不完全满足",
    "技能仅在项目中",
    "AI算法后端交叉",
    "OCR噪声",
    "单项硬门槛不满足",
}
MATCH_LEVELS = {"高匹配", "较匹配", "基本匹配", "低匹配"}
MATCH_LEVEL_ORDER = ("低匹配", "基本匹配", "较匹配", "高匹配")
BOOL_FIELDS = ("岗位方向匹配", "学历匹配", "经验匹配")
LIST_FIELDS = ("命中技能", "缺失技能")


def pair_hash(row: dict[str, Any]) -> str:
    return normalized_input_hash(
        f"{str(row.get('jd_text') or '')}\n---\n{str(row.get('resume_text') or '')}"
    )


def jd_hash(row: dict[str, Any]) -> str:
    return normalized_input_hash(str(row.get("jd_text") or ""))


def resume_hash(row: dict[str, Any]) -> str:
    return normalized_input_hash(str(row.get("resume_text") or ""))


def _label_errors(label: Any) -> list[str]:
    if not isinstance(label, dict):
        return ["label_not_object"]
    errors = []
    if label.get("匹配等级") not in MATCH_LEVELS:
        errors.append("invalid_match_level")
    for field in BOOL_FIELDS:
        if not isinstance(label.get(field), bool):
            errors.append(f"invalid_bool:{field}")
    for field in LIST_FIELDS:
        value = label.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            errors.append(f"invalid_list:{field}")
    matched = set(label.get("命中技能") or [])
    missing = set(label.get("缺失技能") or [])
    if matched & missing:
        errors.append("skill_in_both_matched_and_missing")
    return errors


def audit_match_gold(
    gold_rows: Iterable[dict[str, Any]],
    training_rows: Iterable[dict[str, Any]],
    *,
    jd_training_rows: Iterable[dict[str, Any]] = (),
    resume_training_rows: Iterable[dict[str, Any]] = (),
    min_rows: int = 20,
) -> dict[str, Any]:
    training_rows = list(training_rows)
    training_hashes = {pair_hash(row) for row in training_rows}
    training_jd_hashes = {jd_hash(row) for row in training_rows}
    training_resume_hashes = {resume_hash(row) for row in training_rows}
    jd_task_training_hashes = {
        normalized_input_hash(str(row.get("raw_text") or row.get("clean_text") or ""))
        for row in jd_training_rows
    }
    resume_task_training_hashes = {
        normalized_input_hash(str(row.get("text") or row.get("clean_text") or ""))
        for row in resume_training_rows
    }
    rows = list(gold_rows)
    ids = Counter(str(row.get("id") or "") for row in rows)
    source_groups = Counter(str(row.get("source_group") or "") for row in rows)
    pair_hashes = Counter(pair_hash(row) for row in rows)
    status_counts: Counter[str] = Counter()
    evaluation_role_counts: Counter[str] = Counter()
    inspection_status_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    invalid_rows = []
    verified_rows = 0
    training_overlap_ids = []
    training_jd_overlap_ids = []
    training_resume_overlap_ids = []
    jd_task_training_overlap_ids = []
    resume_task_training_overlap_ids = []
    level_counts: Counter[str] = Counter()
    jd_candidate_counts: Counter[str] = Counter()

    for row in rows:
        row_id = str(row.get("id") or "")
        meta = row.get("meta") or {}
        status = str(meta.get("annotation_status") or "missing")
        status_counts[status] += 1
        evaluation_role_counts[str(meta.get("evaluation_role") or "unspecified")] += 1
        inspection_status_counts[str(meta.get("inspection_status") or "unspecified")] += 1
        tags = [str(tag) for tag in meta.get("difficulty_tags") or []]
        difficulty_counts.update(tag for tag in tags if tag in DIFFICULTY_TAGS)
        errors = _label_errors(row.get("label"))
        level_counts[str((row.get("label") or {}).get("匹配等级") or "missing")] += 1
        jd_candidate_counts[jd_hash(row)] += 1
        if not row_id:
            errors.append("missing_id")
        if not str(row.get("source_group") or ""):
            errors.append("missing_source_group")
        if not str(row.get("jd_text") or "").strip():
            errors.append("missing_jd_text")
        if not str(row.get("resume_text") or "").strip():
            errors.append("missing_resume_text")
        if status != "human_verified":
            errors.append("not_human_verified")
        else:
            verified_rows += 1
            for field in ("annotator_id", "reviewed_at", "rationale"):
                if not str(meta.get(field) or "").strip():
                    errors.append(f"missing_annotation_field:{field}")
        if not tags:
            errors.append("missing_difficulty_tags")
        elif any(tag not in DIFFICULTY_TAGS for tag in tags):
            errors.append("unknown_difficulty_tag")
        if pair_hash(row) in training_hashes:
            training_overlap_ids.append(row_id)
        if jd_hash(row) in training_jd_hashes:
            training_jd_overlap_ids.append(row_id)
        if resume_hash(row) in training_resume_hashes:
            training_resume_overlap_ids.append(row_id)
        if jd_hash(row) in jd_task_training_hashes:
            jd_task_training_overlap_ids.append(row_id)
        if resume_hash(row) in resume_task_training_hashes:
            resume_task_training_overlap_ids.append(row_id)
        if errors:
            invalid_rows.append({"id": row_id, "errors": errors})

    duplicate_ids = sorted(key for key, count in ids.items() if key and count > 1)
    duplicate_source_groups = sorted(
        key for key, count in source_groups.items() if key and count > 1
    )
    duplicate_pairs = sum(count - 1 for count in pair_hashes.values() if count > 1)
    missing_difficulty_tags = sorted(DIFFICULTY_TAGS - set(difficulty_counts))
    ready = bool(
        len(rows) >= min_rows
        and verified_rows == len(rows)
        and not invalid_rows
        and not duplicate_ids
        and not duplicate_source_groups
        and duplicate_pairs == 0
        and not training_overlap_ids
        and not training_jd_overlap_ids
        and not training_resume_overlap_ids
        and not jd_task_training_overlap_ids
        and not resume_task_training_overlap_ids
        and not missing_difficulty_tags
    )
    blind_declared = evaluation_role_counts == {"blind_holdout": len(rows)}
    blind_unseen = inspection_status_counts == {"unseen": len(rows)}
    blind_ready = ready and blind_declared and blind_unseen
    blind_blockers = []
    if not ready:
        blind_blockers.append("regression_quality_or_independence_gate_failed")
    if not blind_declared:
        blind_blockers.append("evaluation_role_not_blind_holdout")
    if not blind_unseen:
        blind_blockers.append("inspection_status_not_unseen")
    minimum_level_support = min(
        (level_counts.get(level, 0) for level in MATCH_LEVEL_ORDER),
        default=0,
    )
    decision_ready = ready and minimum_level_support >= 5
    decision_blockers = []
    if not ready:
        decision_blockers.append("regression_quality_or_independence_gate_failed")
    if minimum_level_support < 5:
        decision_blockers.append("match_level_support_below_5")
    multi_candidate_queries = sum(count >= 5 for count in jd_candidate_counts.values())
    ranking_ready = ready and multi_candidate_queries >= 20
    ranking_blockers = []
    if not ready:
        ranking_blockers.append("regression_quality_or_independence_gate_failed")
    if multi_candidate_queries < 20:
        ranking_blockers.append("fewer_than_20_queries_with_5_candidates")
    return {
        "gold_ready": ready,
        "regression_ready": ready,
        "blind_ready": blind_ready,
        "blind_blockers": blind_blockers,
        "decision_ready": decision_ready,
        "decision_blockers": decision_blockers,
        "ranking_ready": ranking_ready,
        "ranking_blockers": ranking_blockers,
        "evaluation_scope": {
            "pair_regression": ready,
            "match_level_decision": decision_ready,
            "candidate_ranking": ranking_ready,
        },
        "total_rows": len(rows),
        "minimum_rows": min_rows,
        "human_verified_rows": verified_rows,
        "annotation_status_counts": dict(status_counts),
        "evaluation_role_counts": dict(evaluation_role_counts),
        "inspection_status_counts": dict(inspection_status_counts),
        "difficulty_counts": dict(difficulty_counts),
        "match_level_counts": {
            level: level_counts.get(level, 0) for level in MATCH_LEVEL_ORDER
        },
        "minimum_match_level_support": minimum_level_support,
        "queries_with_at_least_5_candidates": multi_candidate_queries,
        "missing_difficulty_tags": missing_difficulty_tags,
        "invalid_rows": invalid_rows,
        "duplicate_ids": duplicate_ids,
        "duplicate_source_groups": duplicate_source_groups,
        "duplicate_pair_count": duplicate_pairs,
        "training_overlap_count": len(training_overlap_ids),
        "training_overlap_ids": training_overlap_ids,
        "training_jd_overlap_count": len(training_jd_overlap_ids),
        "training_jd_overlap_ids": training_jd_overlap_ids,
        "training_resume_overlap_count": len(training_resume_overlap_ids),
        "training_resume_overlap_ids": training_resume_overlap_ids,
        "jd_task_training_overlap_count": len(jd_task_training_overlap_ids),
        "jd_task_training_overlap_ids": jd_task_training_overlap_ids,
        "resume_task_training_overlap_count": len(resume_task_training_overlap_ids),
        "resume_task_training_overlap_ids": resume_task_training_overlap_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/private/match_gold.jsonl")
    parser.add_argument("--training", default="data/eval/match_train_pool_combined.jsonl")
    parser.add_argument("--jd-training", default="data/eval/jd_train_pool_combined.jsonl")
    parser.add_argument("--resume-training", default="data/eval/resume_train_pool_combined.jsonl")
    parser.add_argument("--minimum-rows", type=int, default=20)
    parser.add_argument("--out", default="outputs/eval_reports/match_gold_audit.json")
    args = parser.parse_args()

    report = audit_match_gold(
        read_jsonl(args.gold),
        read_jsonl(args.training) if Path(args.training).exists() else [],
        jd_training_rows=(
            read_jsonl(args.jd_training) if Path(args.jd_training).exists() else []
        ),
        resume_training_rows=(
            read_jsonl(args.resume_training) if Path(args.resume_training).exists() else []
        ),
        min_rows=args.minimum_rows,
    )
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
