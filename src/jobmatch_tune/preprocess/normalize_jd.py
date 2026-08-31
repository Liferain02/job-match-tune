from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.database import iter_table_batches, upsert_jd_clean
from jobmatch_tune.dataset.pipeline_freshness import NORMALIZATION_MANIFEST, write_normalization_manifest
from jobmatch_tune.preprocess.clean_text import clean_text
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
    extract_experience_requirement_from_meta,
    extract_skills_from_text,
    infer_job_direction,
)
from jobmatch_tune.preprocess.jd_sections import split_sections
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def load_label_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_jd_row(row: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        meta_json = row.get("meta_json")
        if isinstance(meta_json, str) and meta_json.strip():
            try:
                meta = json.loads(meta_json)
            except json.JSONDecodeError:
                meta = {}
        else:
            meta = {}
    raw_text = row.get("raw_text") or row.get("html") or ""
    cleaned = clean_text(raw_text, is_html=False)
    source = str(row.get("source") or meta.get("source") or "")
    sections = split_sections(cleaned, source=source)
    title = row.get("job_title") or ""
    existing_labels = row.get("labels") or {}
    experience = (
        extract_experience_requirement(cleaned)
        or extract_experience_requirement_from_meta(meta)
        or str(existing_labels.get("经验要求") or "").strip()
    )
    labels = {
        "岗位方向": infer_job_direction(title, cleaned, schema),
        "必备技能": extract_skills_from_text(cleaned, schema),
        "经验要求": experience,
        "学历要求": extract_education_requirement(cleaned),
    }
    return {
        "id": row["id"],
        "raw_id": row["id"],
        "job_title": title,
        "source": source,
        "company": row.get("company") or "",
        "location": row.get("location") or "",
        "clean_text": cleaned,
        "sections": sections,
        "labels": labels,
        "meta": meta,
        "language": meta.get("language", ""),
        "sft_ready": bool(meta.get("sft_ready", True)),
    }


def normalize_file(input_path: str, output_path: str, schema_path: str) -> list[dict[str, Any]]:
    schema = load_label_schema(schema_path)
    rows = [normalize_jd_row(row, schema) for row in read_jsonl(input_path)]
    write_jsonl(output_path, rows)
    return rows


def normalize_database(
    db_path: str,
    output_path: str,
    schema: dict[str, Any],
    *,
    batch_size: int = 1000,
    sync_clean_table: bool = True,
) -> int:
    count = 0

    def normalized_rows():
        nonlocal count
        for raw_batch in iter_table_batches(db_path, "jd_raw", batch_size=batch_size):
            clean_batch = [normalize_jd_row(row, schema) for row in raw_batch]
            if sync_clean_table:
                upsert_jd_clean(db_path, clean_batch)
            count += len(clean_batch)
            yield from clean_batch

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_jsonl(temporary, normalized_rows())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/jd_raw.jsonl")
    parser.add_argument("--out", default="data/interim/jd_clean.jsonl")
    parser.add_argument("--schema", default="configs/label_schema.yaml")
    parser.add_argument("--db", default=None)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--skip-clean-table-sync",
        action="store_true",
        help="Only rebuild the JSONL derivative; jd_clean is not used by the training pipeline.",
    )
    parser.add_argument("--manifest-out", default=NORMALIZATION_MANIFEST)
    args = parser.parse_args()

    schema = load_label_schema(args.schema)
    if args.db:
        row_count = normalize_database(
            args.db,
            args.out,
            schema,
            batch_size=args.batch_size,
            sync_clean_table=not args.skip_clean_table_sync,
        )
        write_normalization_manifest(args.db, args.out, args.manifest_out, args.schema)
    else:
        rows = normalize_file(args.input, args.out, args.schema)
        row_count = len(rows)
    print(f"wrote {row_count} cleaned JD rows to {args.out}")


if __name__ == "__main__":
    main()
