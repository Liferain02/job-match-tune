from __future__ import annotations

import argparse

from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", default="data/sft/train.jsonl")
    parser.add_argument("--base-valid", default="data/sft/valid.jsonl")
    parser.add_argument("--hard-train", default="data/sft/hardcases_train.jsonl")
    parser.add_argument("--hard-valid", default="data/sft/hardcases_valid.jsonl")
    parser.add_argument("--train-out", default="data/sft/train_incremental.jsonl")
    parser.add_argument("--valid-out", default="data/sft/valid_incremental.jsonl")
    args = parser.parse_args()

    train_rows = list(read_jsonl(args.base_train)) + list(read_jsonl(args.hard_train))
    valid_rows = list(read_jsonl(args.base_valid)) + list(read_jsonl(args.hard_valid))
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.valid_out, valid_rows)
    print(f"wrote {len(train_rows)} train rows to {args.train_out}")
    print(f"wrote {len(valid_rows)} valid rows to {args.valid_out}")


if __name__ == "__main__":
    main()
