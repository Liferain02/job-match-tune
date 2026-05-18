from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def normalize_compact(text: str) -> str:
    return re.sub(r"\s+", "", text.lower()).strip()


def normalize_title(text: str) -> str:
    lowered = normalize_compact(text)
    lowered = re.sub(r"[（）()\[\]【】\-_/·,.，。:：;；]+", "", lowered)
    return lowered


def normalize_similarity_text(text: str) -> str:
    lowered = normalize_compact(text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", lowered)


def fingerprint(text: str) -> str:
    return hashlib.sha1(normalize_compact(text).encode("utf-8")).hexdigest()


def build_bucket_key(row: dict[str, Any]) -> str:
    title = normalize_title(str(row.get("job_title") or ""))
    company = normalize_title(str(row.get("company") or ""))
    location = normalize_title(str(row.get("location") or ""))
    source = normalize_title(str(row.get("source") or ""))
    text_prefix = normalize_similarity_text(str(row.get("clean_text") or ""))[:32]
    if company and title:
        return "|".join((source, company, title, location))
    if title:
        return "|".join((source, title, location, text_prefix))
    return "|".join((source, text_prefix))


def build_shingles(text: str, ngram_size: int = 5) -> set[str]:
    normalized = normalize_similarity_text(text)
    if not normalized:
        return set()
    if len(normalized) <= ngram_size:
        return {normalized}
    return {normalized[i : i + ngram_size] for i in range(len(normalized) - ngram_size + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def text_similarity(left: str, right: str, *, ngram_size: int = 5) -> float:
    left_normalized = normalize_similarity_text(left)
    right_normalized = normalize_similarity_text(right)
    jaccard = jaccard_similarity(
        build_shingles(left_normalized, ngram_size=ngram_size),
        build_shingles(right_normalized, ngram_size=ngram_size),
    )
    sequence_ratio = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    return max(jaccard, sequence_ratio)


def deduplicate_rows(
    rows: Iterable[dict[str, Any]],
    text_key: str = "clean_text",
    *,
    near_threshold: float = 0.9,
    ngram_size: int = 5,
) -> list[dict[str, Any]]:
    kept_rows: list[dict[str, Any]] = []
    bucket_fingerprints: dict[str, set[str]] = defaultdict(set)
    bucket_texts: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        text = str(row.get(text_key, ""))
        bucket_key = build_bucket_key(row)
        exact_fp = fingerprint(text)
        if exact_fp in bucket_fingerprints[bucket_key]:
            continue

        duplicate = False
        for existing_text in bucket_texts[bucket_key]:
            if text_similarity(text, existing_text, ngram_size=ngram_size) >= near_threshold:
                duplicate = True
                break
        if duplicate:
            continue

        bucket_fingerprints[bucket_key].add(exact_fp)
        bucket_texts[bucket_key].append(text)
        kept_rows.append(row)
    return kept_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-key", default="clean_text")
    parser.add_argument("--near-threshold", type=float, default=0.9)
    parser.add_argument("--ngram-size", type=int, default=5)
    args = parser.parse_args()
    rows = deduplicate_rows(
        read_jsonl(args.input),
        args.text_key,
        near_threshold=args.near_threshold,
        ngram_size=args.ngram_size,
    )
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} unique rows to {args.out}")


if __name__ == "__main__":
    main()
