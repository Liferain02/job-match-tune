from __future__ import annotations

import argparse
import json
import random
from typing import Any

from jobmatch_tune.dataset.templates import SYSTEM_PROMPT, jd_parse_prompt
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def build_jd_parse_sample(row: dict[str, Any]) -> dict[str, Any]:
    labels = row.get("labels", {})
    sections = row.get("sections", {})
    source_text = row.get("clean_text", "")
    assistant = {
        "岗位方向": labels.get("岗位方向", ""),
        "核心职责": _split_lines(sections.get("responsibilities", ""))[:6],
        "必备技能": labels.get("必备技能", []),
        "加分项": _split_lines(sections.get("bonus", ""))[:6],
        "经验要求": labels.get("经验要求") or extract_experience_requirement(source_text),
        "学历要求": labels.get("学历要求") or extract_education_requirement(source_text),
    }
    return {
        "id": f"{row['id']}_jd_parse",
        "task_type": "jd_parse",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": jd_parse_prompt(compose_jd_input_text(row))},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
    }


def _split_lines(text: str) -> list[str]:
    lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip(" -•\t")]
    return [
        line
        for line in lines
        if not line.startswith("经验要求")
        and not line.startswith("学历要求")
        and not line.startswith("任职要求")
        and not line.startswith("岗位要求")
    ]


def compose_jd_input_text(row: dict[str, Any]) -> str:
    clean_text = row.get("clean_text", "").strip()
    title = str(row.get("job_title") or "").strip()
    company = str(row.get("company") or "").strip()
    location = str(row.get("location") or "").strip()
    header = []
    if title and "岗位名称" not in clean_text[:80]:
        header.append(f"岗位名称：{title}")
    if company and "公司" not in clean_text[:120]:
        header.append(f"公司：{company}")
    if location and "工作地点" not in clean_text[:120]:
        header.append(f"工作地点：{location}")
    return "\n".join(header + ([clean_text] if clean_text else []))


def split_samples(
    samples: list[dict[str, Any]], train_ratio: float, valid_ratio: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n < 3:
        return {"train": shuffled, "valid": shuffled[:1], "test": shuffled[:1]}
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", default="data/interim/jd_clean.jsonl")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    args = parser.parse_args()

    samples = [build_jd_parse_sample(row) for row in read_jsonl(args.jd)]
    splits = split_samples(samples, args.train_ratio, args.valid_ratio, args.seed)
    for split, rows in splits.items():
        write_jsonl(f"{args.out_dir}/{split}.jsonl", rows)
        print(f"wrote {len(rows)} {split} samples")


if __name__ == "__main__":
    main()
