from __future__ import annotations

import argparse
import random
from typing import Any

import yaml

from jobmatch_tune.dataset.build_sft_dataset import build_jd_parse_sample, split_samples
from jobmatch_tune.preprocess.normalize_jd import normalize_jd_row
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


WEAK_BOOTSTRAP_SOURCES = {
    "hf_job_educational_train_2026_05_17",
    "hf_job_educational_validation_2026_05_17",
    "hf_job_educational_test_2026_05_17",
    "github_workaggregation_test",
    "github_jhcoco_bosszp",
}

LOW_SIGNAL_TITLE_KEYWORDS = [
    "实习",
    "应届",
    "校招",
    "培训生",
]
LOW_SIGNAL_TEXT_PATTERNS = [
    "任务类型：从岗位中提取学历",
    "毕业生",
    "管培生",
    "实习单位",
    "2025届",
    "2026届",
    "2027届",
]


def load_schema(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_bootstrap_rows(rows: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    built = []
    for row in rows:
        normalized = normalize_jd_row(
            {
                "id": row["id"],
                "job_title": row.get("job_title", ""),
                "source": row.get("source", ""),
                "company": row.get("company", ""),
                "location": row.get("location", ""),
                "raw_text": row.get("raw_text", ""),
                "meta": row.get("meta") or {},
            },
            schema,
        )
        labels = normalized.get("labels") or {}
        if not str(labels.get("岗位方向") or "").strip():
            continue
        if not (
            labels.get("必备技能")
            or labels.get("学历要求")
            or labels.get("经验要求")
            or normalized.get("sections")
        ):
            continue
        if not is_usable_bootstrap_row(normalized):
            continue
        built.append(normalized)
    return built


def is_usable_bootstrap_row(row: dict[str, Any]) -> bool:
    title = str(row.get("job_title") or "").strip().lower()
    source = str(row.get("source") or "")
    labels = row.get("labels") or {}
    sections = row.get("sections") or {}
    clean_text = str(row.get("clean_text") or "").strip()

    responsibilities = str(sections.get("responsibilities") or "").strip()
    requirements = str(sections.get("requirements") or "").strip()
    bonus = str(sections.get("bonus") or "").strip()
    skill_count = len(labels.get("必备技能") or [])
    has_education = bool(str(labels.get("学历要求") or "").strip())
    has_experience = bool(str(labels.get("经验要求") or "").strip())
    responsibility_lines = [line for line in responsibilities.splitlines() if line.strip()]

    if not has_education:
        return False

    if source in WEAK_BOOTSTRAP_SOURCES:
        if any(keyword in title for keyword in LOW_SIGNAL_TITLE_KEYWORDS):
            return False
        if any(pattern in clean_text for pattern in LOW_SIGNAL_TEXT_PATTERNS) and not has_experience:
            return False
        if skill_count < 1:
            return False
        if not responsibilities and not requirements:
            return False
        if len(responsibility_lines) == 0 and len(responsibilities) < 60:
            return False
        if not has_experience and skill_count < 2:
            return False
        if not has_experience and len(clean_text) < 220:
            return False
        return True

    strong_signals = sum(
        bool(flag)
        for flag in (
            responsibilities,
            requirements,
            bonus,
            labels.get("必备技能"),
            labels.get("经验要求"),
        )
    )
    return strong_signals >= 2


def sample_rows(rows: list[dict[str, Any]], max_rows: int | None, seed: int) -> list[dict[str, Any]]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    chosen = rows[:]
    rng.shuffle(chosen)
    return chosen[:max_rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/jd_train_pool_combined.jsonl")
    parser.add_argument("--schema", default="configs/label_schema.yaml")
    parser.add_argument("--out-dir", default="data/sft_jd_bootstrap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=12000)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    rows = list(read_jsonl(args.input))
    bootstrap_rows = build_bootstrap_rows(rows, schema)
    bootstrap_rows = sample_rows(bootstrap_rows, args.max_rows, args.seed)
    samples = [build_jd_parse_sample(row) for row in bootstrap_rows]
    splits = split_samples(samples, args.train_ratio, args.valid_ratio, args.seed)
    for split, split_rows in splits.items():
        write_jsonl(f"{args.out_dir}/{split}.jsonl", split_rows)
        print(f"wrote {len(split_rows)} {split} samples")


if __name__ == "__main__":
    main()
