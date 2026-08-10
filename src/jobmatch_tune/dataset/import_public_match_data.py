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
    raise ValueError(f"Unsupported match source file type: {path}")


def build_match_pair_row(
    row: dict[str, Any],
    source: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    mapping = source.get("mapping") or {}
    jd_text = normalize_text(get_path_value(row, mapping.get("jd_text", "")))
    resume_text = normalize_text(get_path_value(row, mapping.get("resume_text", "")))
    if not jd_text or not resume_text:
        return None
    label_value = get_path_value(row, mapping.get("label", "")) if mapping.get("label") else ""
    score_value = get_path_value(row, mapping.get("score", "")) if mapping.get("score") else ""
    digest = hashlib.sha1(
        f"{source['name']}::{index}::{jd_text}::{resume_text}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        "id": f"{source['name']}_{digest}",
        "task": "match",
        "source_type": normalize_text(source.get("source_type") or "public_pair"),
        "jd_text": jd_text,
        "resume_text": resume_text,
        "label": {
            "raw_label": label_value,
            "raw_score": score_value,
        },
        "meta": {
            "source_name": source["name"],
            "schema": source["schema"],
            "language": normalize_text(source.get("language") or "unknown"),
            "source_path": normalize_text(source.get("path")),
            "source_url": normalize_text(source.get("source_url")),
            "source_revision": normalize_text(source.get("source_revision")),
            "provenance_status": normalize_text(
                source.get("provenance_status") or "undocumented"
            ),
            "license_status": normalize_text(source.get("license_status") or "unconfirmed"),
            "intended_usage": normalize_text(source.get("intended_usage") or "audit_only"),
        },
    }


def convert_rows(source: dict[str, Any], rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    schema = normalize_text(source.get("schema"))
    if schema != "match_pair_rows":
        raise ValueError(f"Unsupported match schema: {schema}")
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = build_match_pair_row(row, source, index)
        if item:
            converted.append(item)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/public_match_sources.yaml")
    parser.add_argument("--out", default="data/external/public_match_imports.jsonl")
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for source in load_sources(args.manifest):
        if source.get("enabled") is False:
            print(f"{source['name']}: skipped (disabled)")
            continue
        rows = read_rows(source["path"])
        converted = convert_rows(source, rows)
        all_rows.extend(converted)
        print(f"{source['name']}: {len(converted)}")
    write_jsonl(args.out, all_rows)
    print(f"total: {len(all_rows)} -> {args.out}")


if __name__ == "__main__":
    main()
