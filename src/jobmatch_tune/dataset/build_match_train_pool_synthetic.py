from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.match.rule_engine import compute_match_rule_result
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
    extract_skills_from_text,
    infer_job_direction,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def load_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_jd_structured(row: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    title = str(row.get("job_title") or "")
    text = str(row.get("raw_text") or "")
    return {
        "岗位方向": infer_job_direction(title, text, schema),
        "学历要求": extract_education_requirement(text),
        "经验要求": extract_experience_requirement(text),
        "必备技能": extract_skills_from_text(text, schema),
    }


def build_resume_structured(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("label") or {}


def can_use_jd(row: dict[str, Any], structured: dict[str, Any]) -> bool:
    return bool(
        str(row.get("job_title") or "").strip()
        and str(row.get("raw_text") or "").strip()
        and str(structured.get("岗位方向") or "").strip()
    )


def can_use_resume(row: dict[str, Any]) -> bool:
    label = row.get("label") or {}
    return bool(
        str(label.get("目标岗位") or "").strip()
        and label.get("核心技能")
        and str(row.get("text") or "").strip()
    )


def pick_negative_resume(
    current_direction: str,
    all_resumes: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in all_resumes
        if str((row.get("label") or {}).get("目标岗位") or "").strip() != current_direction
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def make_row(
    jd_row: dict[str, Any],
    jd_structured: dict[str, Any],
    resume_row: dict[str, Any],
    idx: int,
) -> dict[str, Any]:
    resume_structured = build_resume_structured(resume_row)
    label = compute_match_rule_result(
        jd_structured,
        resume_structured,
        jd_text=str(jd_row.get("raw_text") or ""),
        resume_text=str(resume_row.get("text") or ""),
    )
    return {
        "id": f"synthetic_match_{jd_row['id']}_{resume_row['id']}_{idx}",
        "task": "match",
        "source_type": "synthetic_text",
        "jd_text": str(jd_row.get("raw_text") or ""),
        "resume_text": str(resume_row.get("text") or ""),
        "label": {
            "匹配等级": label["匹配等级"],
            "岗位方向匹配": label["岗位方向匹配"],
            "学历匹配": label["学历匹配"],
            "经验匹配": label["经验匹配"],
            "命中技能": label["命中技能"],
            "缺失技能": label["缺失技能"],
            "命中项目": label["命中项目"],
            "raw_score": label["匹配分数"],
        },
        "meta": {
            "language": "zh",
            "generator": "jd_resume_rule_pairing_v1",
            "jd_direction": jd_structured["岗位方向"],
            "resume_direction": resume_structured.get("目标岗位", ""),
        },
    }


def build_rows(
    jd_rows: list[dict[str, Any]],
    resume_rows: list[dict[str, Any]],
    schema: dict[str, Any],
    *,
    seed: int = 42,
    positive_per_jd: int = 2,
    negatives_per_jd: int = 1,
    max_jd_rows: int = 300,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    usable_resumes = [row for row in resume_rows if can_use_resume(row)]
    resumes_by_direction: dict[str, list[dict[str, Any]]] = {}
    for row in usable_resumes:
        direction = str((row.get("label") or {}).get("目标岗位") or "").strip()
        resumes_by_direction.setdefault(direction, []).append(row)

    usable_jd = []
    for row in jd_rows:
        structured = build_jd_structured(row, schema)
        if can_use_jd(row, structured):
            usable_jd.append((row, structured))

    rng.shuffle(usable_jd)
    rows: list[dict[str, Any]] = []
    for jd_row, jd_structured in usable_jd[:max_jd_rows]:
        direction = str(jd_structured["岗位方向"])
        positives = resumes_by_direction.get(direction) or []
        if positives:
            sample_count = min(positive_per_jd, len(positives))
            for idx, resume_row in enumerate(rng.sample(positives, sample_count), start=1):
                rows.append(make_row(jd_row, jd_structured, resume_row, idx))
        for extra_idx in range(negatives_per_jd):
            negative = pick_negative_resume(direction, usable_resumes, rng)
            if negative is None:
                continue
            rows.append(make_row(jd_row, jd_structured, negative, positive_per_jd + extra_idx + 1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd-input", default="data/eval/jd_train_pool_combined.jsonl")
    parser.add_argument("--resume-input", default="data/eval/resume_train_pool_combined.jsonl")
    parser.add_argument("--schema", default="configs/label_schema.yaml")
    parser.add_argument("--out", default="data/eval/match_train_pool_synthetic.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-per-jd", type=int, default=2)
    parser.add_argument("--negatives-per-jd", type=int, default=1)
    parser.add_argument("--max-jd-rows", type=int, default=360)
    args = parser.parse_args()

    jd_rows = list(read_jsonl(args.jd_input))
    resume_rows = list(read_jsonl(args.resume_input))
    schema = load_schema(args.schema)
    rows = build_rows(
        jd_rows,
        resume_rows,
        schema,
        seed=args.seed,
        positive_per_jd=args.positive_per_jd,
        negatives_per_jd=args.negatives_per_jd,
        max_jd_rows=args.max_jd_rows,
    )
    write_jsonl(args.out, rows)
    print(f"synthetic={len(rows)}")


if __name__ == "__main__":
    main()
