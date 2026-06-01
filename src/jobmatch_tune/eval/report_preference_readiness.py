from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_text


SMOKE_THRESHOLDS = {"train": 50, "valid": 10}
FULL_THRESHOLDS = {"train": 1000, "valid": 100}


def _holdout_ids(path: str) -> set[str]:
    file_path = Path(path)
    if not file_path.exists():
        return set()
    return {str(row.get("id") or "") for row in read_jsonl(file_path)}


def _preference_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return ""


def _completion_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return str(value[0].get("content") or "").strip()
    return ""


def audit_preference_files(train_path: str, valid_path: str, holdout_path: str) -> dict[str, Any]:
    holdout_ids = _holdout_ids(holdout_path)
    split_counts: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    ids: set[str] = set()
    prompt_splits: dict[str, str] = {}
    invalid_rows = 0
    duplicate_ids = 0
    chosen_equals_rejected = 0
    cross_split_prompt_hashes = 0
    holdout_overlap = 0

    for path in (train_path, valid_path):
        split = Path(path).stem
        file_path = Path(path)
        if not file_path.exists():
            continue
        for row in read_jsonl(file_path):
            split_counts[split] += 1
            try:
                row_id = str(row["id"])
                source_id = str(row.get("source_id") or row_id)
                prompt = _preference_text(row["prompt"])
                chosen = _completion_content(row["chosen"])
                rejected = _completion_content(row["rejected"])
                if not prompt or not chosen or not rejected:
                    raise ValueError("missing preference text")
                json.loads(chosen)
                json.loads(rejected)
            except Exception:
                invalid_rows += 1
                continue
            if row_id in ids:
                duplicate_ids += 1
            ids.add(row_id)
            if chosen == rejected:
                chosen_equals_rejected += 1
            prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
            previous_split = prompt_splits.get(prompt_hash)
            if previous_split and previous_split != split:
                cross_split_prompt_hashes += 1
            prompt_splits.setdefault(prompt_hash, split)
            if source_id in holdout_ids:
                holdout_overlap += 1
            strategy = str((row.get("meta") or {}).get("rejection_strategy") or "prediction_mismatch")
            strategies[strategy] += 1

    counts = {"train": split_counts["train"], "valid": split_counts["valid"]}
    format_ready = (
        invalid_rows == 0
        and duplicate_ids == 0
        and chosen_equals_rejected == 0
        and cross_split_prompt_hashes == 0
        and holdout_overlap == 0
    )
    return {
        "counts": counts,
        "thresholds": {"smoke": SMOKE_THRESHOLDS, "full": FULL_THRESHOLDS},
        "invalid_rows": invalid_rows,
        "duplicate_ids": duplicate_ids,
        "chosen_equals_rejected": chosen_equals_rejected,
        "cross_split_prompt_hashes": cross_split_prompt_hashes,
        "holdout_overlap": holdout_overlap,
        "unique_prompts": len(prompt_splits),
        "strategy_counts": dict(strategies.most_common()),
        "format_ready": format_ready,
        "ready_for_dpo_smoke": format_ready
        and counts["train"] >= SMOKE_THRESHOLDS["train"]
        and counts["valid"] >= SMOKE_THRESHOLDS["valid"],
        "ready_for_dpo": format_ready
        and counts["train"] >= FULL_THRESHOLDS["train"]
        and counts["valid"] >= FULL_THRESHOLDS["valid"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/preference/train.jsonl")
    parser.add_argument("--valid", default="data/preference/valid.jsonl")
    parser.add_argument("--holdout", default="data/eval/jd_manual_eval_50.jsonl")
    parser.add_argument("--out", default="outputs/eval_reports/preference_readiness_report.json")
    args = parser.parse_args()

    report = audit_preference_files(args.train, args.valid, args.holdout)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
