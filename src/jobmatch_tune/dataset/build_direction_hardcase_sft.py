from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import build_jd_parse_sample
from jobmatch_tune.eval.build_direction_hardcases import HARDCASE_DIRECTIONS
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


VALID_IDS = {
    "tencent_1987809143616069632",
    "tencent_2026648959719735296",
    "tencent_2020809468387946496",
    "tencent_2027658886831570944",
    "tencent_1985973381228552192",
    "tencent_2043950303379877888",
    "tencent_2029541840893669376",
    "tencent_2017094358532247552",
    "tencent_2010621139876995072",
    "tencent_1968944641277579264",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/jd_clean.jsonl")
    parser.add_argument("--train-out", default="data/sft/hardcases_train.jsonl")
    parser.add_argument("--valid-out", default="data/sft/hardcases_valid.jsonl")
    parser.add_argument("--train-repeat", type=int, default=2)
    args = parser.parse_args()

    rows = {row["id"]: row for row in read_jsonl(args.input)}
    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    for row_id, direction in HARDCASE_DIRECTIONS.items():
        row = dict(rows[row_id])
        row.setdefault("labels", {})
        row["labels"] = {**row["labels"], "岗位方向": direction}
        sample = build_jd_parse_sample(row)
        assistant = json.loads(sample["messages"][2]["content"])
        assistant["岗位方向"] = direction
        sample["messages"][2]["content"] = json.dumps(assistant, ensure_ascii=False)
        if row_id in VALID_IDS:
            valid_rows.append(sample)
        else:
            for repeat_idx in range(args.train_repeat):
                repeated = dict(sample)
                repeated["id"] = f"{sample['id']}_hardcase_r{repeat_idx+1}"
                repeated["messages"] = [dict(message) for message in sample["messages"]]
                train_rows.append(repeated)

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.valid_out, valid_rows)
    print(f"wrote {len(train_rows)} train hardcases to {args.train_out}")
    print(f"wrote {len(valid_rows)} valid hardcases to {args.valid_out}")


if __name__ == "__main__":
    main()
