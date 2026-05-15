from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.database import fetch_table, init_db, upsert_jd_clean
from jobmatch_tune.preprocess.clean_text import clean_text
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
    extract_skills_from_text,
    infer_job_direction,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


SECTION_ALIASES = {
    "responsibilities": ["岗位职责", "工作职责", "职位描述", "工作内容", "职责描述"],
    "requirements": ["任职要求", "岗位要求", "职位要求", "任职资格", "能力要求"],
    "bonus": ["加分项", "优先", "优先考虑", "加分条件"],
}


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {key: [] for key in SECTION_ALIASES}
    current: str | None = None
    for line in text.splitlines():
        normalized = line.strip(" ：:")
        matched = None
        for key, aliases in SECTION_ALIASES.items():
            if any(alias in normalized for alias in aliases):
                matched = key
                break
        if matched:
            current = matched
            tail = re.sub(r"^.*?[：:]", "", line).strip()
            if tail:
                sections[current].append(tail)
            continue
        if current:
            sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items() if value}


def load_label_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_jd_row(row: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    raw_text = row.get("raw_text") or row.get("html") or ""
    cleaned = clean_text(raw_text, is_html=False)
    sections = split_sections(cleaned)
    title = row.get("job_title") or ""
    labels = {
        "岗位方向": infer_job_direction(title, cleaned, schema),
        "必备技能": extract_skills_from_text(cleaned, schema),
        "经验要求": extract_experience_requirement(cleaned),
        "学历要求": extract_education_requirement(cleaned),
    }
    return {
        "id": row["id"],
        "raw_id": row["id"],
        "job_title": title,
        "company": row.get("company") or "",
        "location": row.get("location") or "",
        "clean_text": cleaned,
        "sections": sections,
        "labels": labels,
    }


def normalize_file(input_path: str, output_path: str, schema_path: str) -> list[dict[str, Any]]:
    schema = load_label_schema(schema_path)
    rows = [normalize_jd_row(row, schema) for row in read_jsonl(input_path)]
    write_jsonl(output_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/jd_raw.jsonl")
    parser.add_argument("--out", default="data/interim/jd_clean.jsonl")
    parser.add_argument("--schema", default="configs/label_schema.yaml")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    schema = load_label_schema(args.schema)
    if args.db:
        init_db(args.db)
        raw_rows = fetch_table(args.db, "jd_raw")
        rows = [normalize_jd_row(row, schema) for row in raw_rows]
        upsert_jd_clean(args.db, rows)
        write_jsonl(args.out, rows)
    else:
        rows = normalize_file(args.input, args.out, args.schema)
    print(f"wrote {len(rows)} cleaned JD rows to {args.out}")


if __name__ == "__main__":
    main()
