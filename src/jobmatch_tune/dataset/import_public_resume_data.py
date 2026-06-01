from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from jobmatch_tune.utils.io import write_jsonl


def load_sources(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    sources = payload.get("sources") or []
    if not isinstance(sources, list):
        raise ValueError(f"Invalid source manifest: {path}")
    return sources


def get_path_value(row: dict[str, Any], path: str, default: Any = "") -> Any:
    current: Any = row
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return text if text and text.lower() != "none" else ""


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        parts = value
    else:
        text = normalize_text(value)
        if not text:
            return []
        if "、" in text:
            parts = text.split("、")
        elif "," in text:
            parts = text.split(",")
        elif "\n" in text:
            parts = text.splitlines()
        else:
            parts = [text]
    return [normalize_text(part) for part in parts if normalize_text(part)]


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        raise ValueError(f"Unsupported JSON payload in {path}: expected list")
    if suffix == ".csv":
        return pd.read_csv(file_path).fillna("").to_dict(orient="records")
    if suffix == ".parquet":
        return pd.read_parquet(file_path).fillna("").to_dict(orient="records")
    raise ValueError(f"Unsupported resume source file type: {path}")


def build_resume_parse_row(
    row: dict[str, Any],
    source: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    mapping = source.get("mapping") or {}
    text = normalize_text(get_path_value(row, mapping.get("text", "")))
    if not text:
        return None
    label = {
        "目标岗位": normalize_text(get_path_value(row, mapping.get("target_job", ""))),
        "教育背景": normalize_list(get_path_value(row, mapping.get("education", ""))),
        "核心技能": normalize_list(get_path_value(row, mapping.get("skills", ""))),
        "实习经历": normalize_list(get_path_value(row, mapping.get("internships", ""))),
        "项目经历": normalize_list(get_path_value(row, mapping.get("projects", ""))),
        "优势标签": normalize_list(get_path_value(row, mapping.get("strengths", ""))),
    }
    digest = hashlib.sha1(f"{source['name']}::{index}::{text}".encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"{source['name']}_{digest}",
        "task": "resume_parse",
        "source_type": normalize_text(source.get("source_type") or "public_text"),
        "text": text,
        "label": label,
        "meta": {
            "source_name": source["name"],
            "schema": source["schema"],
            "language": normalize_text(source.get("language") or "zh"),
            "source_path": normalize_text(source.get("path")),
        },
    }


def build_resume_ner_row(
    row: dict[str, Any],
    source: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    mapping = source.get("mapping") or {}
    tokens = get_path_value(row, mapping.get("tokens", ""))
    tags = get_path_value(row, mapping.get("tags", ""))
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if hasattr(tags, "tolist"):
        tags = tags.tolist()
    if not isinstance(tokens, list) or not tokens:
        return None
    if not isinstance(tags, list):
        tags = []
    text = "".join(str(token) for token in tokens)
    digest = hashlib.sha1(f"{source['name']}::{index}::{text}".encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"{source['name']}_{digest}",
        "task": "resume_ner",
        "source_type": normalize_text(source.get("source_type") or "public_text"),
        "text": text,
        "tokens": [str(token) for token in tokens],
        "ner_tags": [str(tag) for tag in tags],
        "meta": {
            "source_name": source["name"],
            "schema": source["schema"],
            "language": normalize_text(source.get("language") or "zh"),
            "tag_scheme": normalize_text(source.get("tag_scheme") or "BIO"),
            "source_path": normalize_text(source.get("path")),
        },
    }


def convert_rows(source: dict[str, Any], rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    schema = normalize_text(source.get("schema"))
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if schema == "resume_parse_rows":
            item = build_resume_parse_row(row, source, index)
        elif schema == "resume_ner_rows":
            item = build_resume_ner_row(row, source, index)
        else:
            raise ValueError(f"Unsupported resume schema: {schema}")
        if item:
            converted.append(item)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/public_resume_sources.yaml")
    parser.add_argument("--out", default="data/external/public_resume_imports.jsonl")
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for source in load_sources(args.manifest):
        if not Path(source["path"]).exists():
            print(f"{source['name']}: skipped missing file {source['path']}")
            continue
        rows = read_rows(source["path"])
        converted = convert_rows(source, rows)
        all_rows.extend(converted)
        print(f"{source['name']}: {len(converted)}")
    write_jsonl(args.out, all_rows)
    print(f"total: {len(all_rows)} -> {args.out}")


if __name__ == "__main__":
    main()
