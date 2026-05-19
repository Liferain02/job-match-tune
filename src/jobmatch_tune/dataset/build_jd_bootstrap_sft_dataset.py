from __future__ import annotations

import argparse
import random
from typing import Any

import yaml

from jobmatch_tune.dataset.build_sft_dataset import build_jd_parse_sample, split_samples
from jobmatch_tune.preprocess.normalize_jd import normalize_jd_row
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


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
