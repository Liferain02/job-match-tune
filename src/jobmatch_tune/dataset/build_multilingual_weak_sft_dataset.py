from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import _split_lines, compose_jd_input_text
from jobmatch_tune.dataset.templates import SYSTEM_PROMPT, jd_parse_prompt
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
    extract_skills_from_text,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


EN_DIRECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "前端开发",
        re.compile(
            r"\b(front[ -]?end|web front|react|vue|javascript|typescript)\b",
            re.I,
        ),
    ),
    (
        "后端开发",
        re.compile(
            r"\b(back[ -]?end|server[- ]side|server engineer|java developer|"
            r"golang|go developer|spring|microservice|distributed systems?|"
            r"platform engineer|devops engineer|sre|cybersecurity engineer|"
            r"devsecops engineer|embedded software engineer)\b",
            re.I,
        ),
    ),
    (
        "数据开发",
        re.compile(
            r"\b(data engineer|etl|spark|flink|hadoop|warehouse|analytics engineer)\b",
            re.I,
        ),
    ),
    (
        "测试开发",
        re.compile(
            r"\b(qa|quality assurance|test engineer|software qa|sdet|automation test)\b",
            re.I,
        ),
    ),
    (
        "算法工程",
        re.compile(
            r"\b(machine learning|deep learning|applied scientist|research scientist|"
            r"research engineer|nlp|computer vision|cv engineer|inference|training|"
            r"multimodal|data scientist|ml engineer|ai scientist)\b",
            re.I,
        ),
    ),
    (
        "AI应用开发",
        re.compile(
            r"\b(agent|rag|copilot|prompt|ai application|ai enablement|"
            r"solutions architect|llm engineer|llm application|genai|generative ai)\b",
            re.I,
        ),
    ),
    (
        "客户端开发",
        re.compile(
            r"\b(android|ios|mobile developer|react native|flutter)\b",
            re.I,
        ),
    ),
]

EN_EDUCATION_PATTERNS = [
    re.compile(r"\b(ph\.?d\.?|doctor(?:ate)?)(?:\s+degree)?\b", re.I),
    re.compile(r"\b(master(?:'s)?)(?:\s+degree)?\b", re.I),
    re.compile(r"\b(bachelor(?:'s)?)(?:\s+degree)?\b", re.I),
]

EN_EXPERIENCE_PATTERNS = [
    re.compile(r"\b(\d{1,2}\+?\s*(?:years?|yrs?)\s+of\s+experience)\b", re.I),
    re.compile(r"\b(minimum\s+\d{1,2}\+?\s*(?:years?|yrs?)\b[^.;\n]*)", re.I),
    re.compile(r"\b(\d{1,2}\+?\s*(?:years?|yrs?)\s+in\b[^.;\n]*)", re.I),
]


def infer_high_confidence_en_direction(title: str, text: str) -> tuple[str, bool]:
    haystack = f"{title}\n{text[:1200]}"
    hits = [direction for direction, pattern in EN_DIRECTION_PATTERNS if pattern.search(haystack)]
    unique_hits = list(dict.fromkeys(hits))
    if len(unique_hits) != 1:
        return "", False
    return unique_hits[0], True


def extract_en_education_requirement(text: str) -> str:
    for pattern in EN_EDUCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).lower()
            if value.startswith("ph"):
                return "博士"
            if value.startswith("master"):
                return "硕士"
            if value.startswith("bachelor"):
                return "本科"
    return ""


def extract_en_experience_requirement(text: str) -> str:
    for pattern in EN_EXPERIENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def select_core_responsibilities(row: dict[str, Any]) -> list[str]:
    sections = row.get("sections", {}) or {}
    candidates = _split_lines(sections.get("responsibilities", ""))
    if candidates:
        return candidates[:6]
    return _split_lines(row.get("clean_text", ""))[:6]


def build_multilingual_weak_sample(
    row: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any] | None:
    language = row.get("language") or ""
    if language == "en":
        direction, ok = infer_high_confidence_en_direction(
            str(row.get("job_title") or ""),
            str(row.get("clean_text") or ""),
        )
        if not ok:
            return None
        responsibilities = select_core_responsibilities(row)
        if len(responsibilities) < 2:
            return None
        text = str(row.get("clean_text") or "")
        assistant = {
            "岗位方向": direction,
            "核心职责": responsibilities,
            "必备技能": extract_skills_from_text(text, schema),
            "加分项": [],
            "经验要求": extract_en_experience_requirement(text),
            "学历要求": extract_en_education_requirement(text),
        }
    else:
        labels = row.get("labels", {}) or {}
        sections = row.get("sections", {}) or {}
        assistant = {
            "岗位方向": labels.get("岗位方向", ""),
            "核心职责": _split_lines(sections.get("responsibilities", ""))[:6],
            "必备技能": labels.get("必备技能", []),
            "加分项": _split_lines(sections.get("bonus", ""))[:6],
            "经验要求": labels.get("经验要求") or extract_experience_requirement(row.get("clean_text", "")),
            "学历要求": labels.get("学历要求") or extract_education_requirement(row.get("clean_text", "")),
        }
        if not assistant["岗位方向"] or len(assistant["核心职责"]) < 1:
            return None
    return {
        "id": f"{row['id']}_jd_parse",
        "task_type": "jd_parse",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": jd_parse_prompt(compose_jd_input_text(row))},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
    }


def split_samples(
    samples: list[dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    valid_count = max(1, int(n * valid_ratio))
    test_count = max(1, n - int(n * train_ratio) - valid_count)
    train_count = max(1, n - valid_count - test_count)
    train_end = train_count
    valid_end = train_end + valid_count
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def load_schema(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_candidate_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        language = row.get("language") or ""
        if language == "en":
            yield row
            continue
        if row.get("sft_ready", True):
            yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", default="data/interim/jd_clean_dedup.jsonl")
    parser.add_argument("--schema", default="configs/label_schema.yaml")
    parser.add_argument("--out-dir", default="data/sft_multilingual_weak")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    samples = []
    for row in iter_candidate_rows(read_jsonl(args.jd)):
        sample = build_multilingual_weak_sample(row, schema)
        if sample is not None:
            samples.append(sample)
    splits = split_samples(samples, args.train_ratio, args.valid_ratio, args.seed)
    for split, rows in splits.items():
        write_jsonl(f"{args.out_dir}/{split}.jsonl", rows)
        print(f"wrote {len(rows)} {split} samples")


if __name__ == "__main__":
    main()
