from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.dataset.build_sft_dataset import build_jd_parse_sample, split_samples
from jobmatch_tune.preprocess.normalize_jd import normalize_jd_row
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


LOW_SIGNAL_TITLE_KEYWORDS = [
    "实习",
    "应届",
    "校招",
    "校园招聘",
    "培训生",
    "教师",
    "老师",
    "讲师",
    "助教",
    "培训师",
    "培训讲师",
    "编导",
    "编剧",
    "摄制",
    "新闻编辑",
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


def load_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_strict_plus_row(row: dict[str, Any]) -> bool:
    title = str(row.get("job_title") or "").strip().lower()
    source = str(row.get("source") or "")
    clean_text = str(row.get("clean_text") or "")
    labels = row.get("labels") or {}
    sections = row.get("sections") or {}

    direction = str(labels.get("岗位方向") or "").strip()
    education = str(labels.get("学历要求") or "").strip()
    experience = str(labels.get("经验要求") or "").strip()
    skills = labels.get("必备技能") or []
    responsibilities = str(sections.get("responsibilities") or "").strip()
    requirements = str(sections.get("requirements") or "").strip()

    if not direction or not education:
        return False
    if any(keyword in title for keyword in LOW_SIGNAL_TITLE_KEYWORDS):
        return False
    if any(pattern in clean_text for pattern in LOW_SIGNAL_TEXT_PATTERNS) and not experience:
        return False

    resp_len = len(responsibilities)
    req_len = len(requirements)
    skill_count = len(skills)

    if source.startswith("hf_") or source.startswith("github_"):
        return skill_count >= 1 and (resp_len >= 40 or req_len >= 40) and (bool(experience) or skill_count >= 2)

    return (resp_len >= 40 or req_len >= 40) and (bool(experience) or skill_count >= 1)


def build_rows(rows: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
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
        if is_strict_plus_row(normalized):
            built.append(normalized)
    return built


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
    parser.add_argument("--out-dir", default="data/sft_jd_strict_plus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    rows = list(read_jsonl(args.input))
    strict_plus_rows = build_rows(rows, schema)
    strict_plus_rows = sample_rows(strict_plus_rows, args.max_rows, args.seed)
    samples = [build_jd_parse_sample(row) for row in strict_plus_rows]
    splits = split_samples(samples, args.train_ratio, args.valid_ratio, args.seed)
    for split, split_rows in splits.items():
        write_jsonl(f"{args.out_dir}/{split}.jsonl", split_rows)
        print(f"wrote {len(split_rows)} {split} samples")


if __name__ == "__main__":
    main()
