from __future__ import annotations

import argparse
import json
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def extract_resume_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        marker = "简历：\n"
        if marker in content:
            return content.split(marker, 1)[1].strip()
    return ""


def extract_label(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        return json.loads(content)
    return {}


def build_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    built = []
    for row in rows:
        messages = row.get("messages") or []
        text = extract_resume_text(messages)
        label = extract_label(messages)
        if not text or not label:
            continue
        built.append(
            {
                "id": row.get("id", ""),
                "task": "resume_parse",
                "source_type": "sft_materialized",
                "text": text,
                "label": label,
                "meta": {"language": "zh", "generator": "resume_sft_projection_v1"},
            }
        )
    return built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/sft_resume/train.jsonl",
            "data/sft_resume/valid.jsonl",
            "data/sft_resume/test.jsonl",
        ],
    )
    parser.add_argument("--out", default="data/eval/resume_train_pool_from_sft.jsonl")
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for path in args.inputs:
        all_rows.extend(list(read_jsonl(path)))
    built = build_rows(all_rows)
    write_jsonl(args.out, built)
    print(f"materialized={len(built)}")


if __name__ == "__main__":
    main()
