from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Iterable
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def deduplicate_rows(rows: Iterable[dict[str, Any]], text_key: str = "clean_text") -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        fp = fingerprint(str(row.get(text_key, "")))
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(row)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-key", default="clean_text")
    args = parser.parse_args()
    rows = deduplicate_rows(read_jsonl(args.input), args.text_key)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} unique rows to {args.out}")


if __name__ == "__main__":
    main()
