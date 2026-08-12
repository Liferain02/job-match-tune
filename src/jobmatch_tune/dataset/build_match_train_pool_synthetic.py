from __future__ import annotations

import argparse
import hashlib
import random
import unicodedata
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
from jobmatch_tune.resume.privacy import sanitize_resume_text_for_training
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def load_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sanitize_jd_text_for_matching(text: str) -> str:
    excluded_prefixes = (
        "任务类型：从岗位中提取学历",
        "任务类型:从岗位中提取学历",
        "学历提示：",
        "学历提示:",
    )
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith(excluded_prefixes)
    ).strip()


def build_jd_structured(row: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    title = str(row.get("job_title") or "")
    text = sanitize_jd_text_for_matching(str(row.get("raw_text") or ""))
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


def select_jds_with_educational_source_cap(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    max_rows: int,
    max_educational_source_rate: float,
    rng: random.Random,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    # Use one seed-derived salt and rank each entity independently. Removing an
    # unselected candidate must not reshuffle the whole review/training pool.
    salt = rng.getrandbits(128)
    ranked = sorted(
        rows,
        key=lambda item: hashlib.sha1(
            (
                f"{salt}\n{item[0].get('id', '')}\n"
                f"{sanitize_jd_text_for_matching(str(item[0].get('raw_text') or ''))}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    educational_limit = int(max_rows * max_educational_source_rate)
    educational_count = 0
    selected = []
    for item in ranked:
        row_id = str(item[0].get("id") or "")
        is_educational = row_id.startswith("hf_job_educational_")
        if is_educational and educational_count >= educational_limit:
            continue
        selected.append(item)
        if is_educational:
            educational_count += 1
        if len(selected) >= max_rows:
            break
    return selected


def _entity_text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    compact = "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] in {"L", "N"}
    )
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()


def _partition_grouped_rows(
    rows: list[Any],
    *,
    group_key: Any,
    seed: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    for row in rows:
        groups.setdefault(str(group_key(row)), []).append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 3:
        return {"train": [row for key in keys for row in groups[key]], "valid": [], "test": []}
    valid_count = max(1, int(len(keys) * valid_ratio))
    test_count = max(1, len(keys) - int(len(keys) * train_ratio) - valid_count)
    train_count = len(keys) - valid_count - test_count
    split_keys = {
        "train": keys[:train_count],
        "valid": keys[train_count : train_count + valid_count],
        "test": keys[train_count + valid_count :],
    }
    return {
        split: [row for key in selected_keys for row in groups[key]]
        for split, selected_keys in split_keys.items()
    }


def partition_match_entities(
    selected_jds: list[tuple[dict[str, Any], dict[str, Any]]],
    resumes: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[
    dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    dict[str, list[dict[str, Any]]],
]:
    jd_splits = {"train": [], "valid": [], "test": []}
    for family, family_rows in {
        "educational": [
            item for item in selected_jds if str(item[0].get("id") or "").startswith("hf_job_educational_")
        ],
        "other": [
            item for item in selected_jds if not str(item[0].get("id") or "").startswith("hf_job_educational_")
        ],
    }.items():
        partitions = _partition_grouped_rows(
            family_rows,
            group_key=lambda item: _entity_text_hash(
                sanitize_jd_text_for_matching(str(item[0].get("raw_text") or ""))
            ),
            seed=f"{seed}:jd:{family}",
        )
        for split in jd_splits:
            jd_splits[split].extend(partitions[split])

    resume_splits = {"train": [], "valid": [], "test": []}
    resumes_by_direction: dict[str, list[dict[str, Any]]] = {}
    for row in resumes:
        direction = str((row.get("label") or {}).get("目标岗位") or "")
        resumes_by_direction.setdefault(direction, []).append(row)
    for direction, direction_rows in resumes_by_direction.items():
        partitions = _partition_grouped_rows(
            direction_rows,
            group_key=lambda row: _entity_text_hash(
                sanitize_resume_text_for_training(str(row.get("text") or ""))
            ),
            seed=f"{seed}:resume:{direction}",
        )
        for split in resume_splits:
            resume_splits[split].extend(partitions[split])
    return jd_splits, resume_splits


def make_row(
    jd_row: dict[str, Any],
    jd_structured: dict[str, Any],
    resume_row: dict[str, Any],
    idx: int,
    *,
    entity_split: str = "train",
) -> dict[str, Any]:
    resume_structured = build_resume_structured(resume_row)
    resume_text = sanitize_resume_text_for_training(str(resume_row.get("text") or ""))
    jd_text = sanitize_jd_text_for_matching(str(jd_row.get("raw_text") or ""))
    label = compute_match_rule_result(
        jd_structured,
        resume_structured,
        jd_text=jd_text,
        resume_text=resume_text,
    )
    return {
        "id": f"synthetic_match_{jd_row['id']}_{resume_row['id']}_{idx}",
        "task": "match",
        "source_type": "synthetic_text",
        "jd_text": jd_text,
        "resume_text": resume_text,
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
            "entity_split": entity_split,
            "jd_entity_hash": _entity_text_hash(jd_text),
            "resume_entity_hash": _entity_text_hash(resume_text),
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
    max_educational_source_rate: float = 0.4,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    usable_resumes = [row for row in resume_rows if can_use_resume(row)]
    usable_jd = []
    for row in jd_rows:
        structured = build_jd_structured(row, schema)
        if can_use_jd(row, structured):
            usable_jd.append((row, structured))

    selected_jds = select_jds_with_educational_source_cap(
        usable_jd,
        max_rows=max_jd_rows,
        max_educational_source_rate=max_educational_source_rate,
        rng=rng,
    )
    jd_splits, resume_splits = partition_match_entities(selected_jds, usable_resumes, seed=seed)
    rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        split_resumes = resume_splits[split]
        resumes_by_direction: dict[str, list[dict[str, Any]]] = {}
        for row in split_resumes:
            direction = str((row.get("label") or {}).get("目标岗位") or "").strip()
            resumes_by_direction.setdefault(direction, []).append(row)
        for jd_row, jd_structured in jd_splits[split]:
            direction = str(jd_structured["岗位方向"])
            positives = resumes_by_direction.get(direction) or []
            if positives:
                sample_count = min(positive_per_jd, len(positives))
                for idx, resume_row in enumerate(rng.sample(positives, sample_count), start=1):
                    rows.append(
                        make_row(
                            jd_row,
                            jd_structured,
                            resume_row,
                            idx,
                            entity_split=split,
                        )
                    )
            for extra_idx in range(negatives_per_jd):
                negative = pick_negative_resume(direction, split_resumes, rng)
                if negative is None:
                    continue
                rows.append(
                    make_row(
                        jd_row,
                        jd_structured,
                        negative,
                        positive_per_jd + extra_idx + 1,
                        entity_split=split,
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd-input", default="data/eval/jd_train_pool_combined.jsonl")
    parser.add_argument("--resume-input", default="data/eval/resume_train_pool_combined.jsonl")
    parser.add_argument("--schema", default="configs/label_schema.yaml")
    parser.add_argument("--out", default="data/eval/match_train_pool_synthetic.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-per-jd", type=int, default=2)
    parser.add_argument("--negatives-per-jd", type=int, default=2)
    parser.add_argument("--max-jd-rows", type=int, default=1200)
    parser.add_argument("--max-educational-source-rate", type=float, default=0.4)
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
        max_educational_source_rate=args.max_educational_source_rate,
    )
    write_jsonl(args.out, rows)
    print(f"synthetic={len(rows)}")


if __name__ == "__main__":
    main()
