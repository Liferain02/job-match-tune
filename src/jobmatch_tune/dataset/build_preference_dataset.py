from __future__ import annotations

import argparse
import json
import random
from typing import Any
from pathlib import Path

from jobmatch_tune.dataset.templates import SYSTEM_PROMPT, jd_parse_prompt, resume_parse_prompt
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def build_prompt_text(task: str, text: str) -> str:
    if task == "jd_parse":
        user_text = jd_parse_prompt(text)
    elif task == "resume_parse":
        user_text = resume_parse_prompt(text)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return f"{SYSTEM_PROMPT}\n\n{user_text}"


def build_preference_row(row: dict[str, Any]) -> dict[str, Any] | None:
    label = row.get("label") or {}
    parsed = row.get("parsed")
    raw_prediction = row.get("prediction") or ""
    task = row.get("task", "jd_parse")
    text = row.get("text", "")

    chosen = json.dumps(label, ensure_ascii=False, sort_keys=True)
    if parsed:
        rejected = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    else:
        rejected = str(raw_prediction).strip()

    if not rejected or chosen == rejected:
        return None

    return {
        "id": row["id"],
        "task_type": task,
        "prompt": build_prompt_text(task, text),
        "chosen": chosen,
        "rejected": rejected,
    }


def load_prediction_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        candidate = Path(item)
        if any(token in item for token in "*?[]"):
            paths.extend(sorted(Path().glob(item)))
        elif candidate.is_dir():
            paths.extend(sorted(candidate.glob("*predictions.jsonl")))
        else:
            paths.append(candidate)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def split_rows(rows: list[dict[str, Any]], valid_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, shuffled[:]
    valid_count = max(1, int(len(shuffled) * valid_ratio))
    train_count = max(1, len(shuffled) - valid_count)
    return shuffled[:train_count], shuffled[train_count:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        action="append",
        default=[],
        help="Prediction JSONL path, glob, or directory. Can be passed multiple times.",
    )
    parser.add_argument("--train-out", default="data/preference/train.jsonl")
    parser.add_argument("--valid-out", default="data/preference/valid.jsonl")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prediction_inputs = args.predictions or ["outputs/eval_reports/*predictions.jsonl"]
    prediction_paths = load_prediction_paths(prediction_inputs)

    rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for prediction_path in prediction_paths:
        for row in read_jsonl(prediction_path):
            built = build_preference_row(row)
            if built is None:
                continue
            dedup_key = (built["id"], built["rejected"])
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)
            rows.append(built)

    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.valid_out, valid_rows)
    print(f"wrote {len(train_rows)} train preference rows to {args.train_out}")
    print(f"wrote {len(valid_rows)} valid preference rows to {args.valid_out}")


if __name__ == "__main__":
    main()
