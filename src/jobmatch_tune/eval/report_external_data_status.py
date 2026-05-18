from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.utils.io import write_text


def load_sources(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    sources = payload.get("sources") or []
    return sources if isinstance(sources, list) else []


def describe_manifest(name: str, manifest_path: str | Path) -> dict[str, Any]:
    sources = load_sources(manifest_path)
    items = []
    existing = 0
    for source in sources:
        source_path = Path(str(source.get("path") or ""))
        exists = source_path.exists()
        if exists:
            existing += 1
        items.append(
            {
                "name": str(source.get("name") or ""),
                "path": str(source_path),
                "exists": exists,
            }
        )
    return {
        "manifest": name,
        "manifest_path": str(manifest_path),
        "total_sources": len(items),
        "existing_sources": existing,
        "missing_sources": len(items) - existing,
        "sources": items,
    }


def build_report() -> dict[str, Any]:
    manifests = {
        "public_job_sources": "configs/public_job_sources.yaml",
        "public_resume_sources": "configs/public_resume_sources.yaml",
        "public_match_sources": "configs/public_match_sources.yaml",
    }
    sections = {name: describe_manifest(name, path) for name, path in manifests.items()}
    return {
        "summary": {
            "all_manifests_ready": all(
                section["total_sources"] > 0 and section["missing_sources"] == 0
                for section in sections.values()
            ),
            "ready_manifests": [
                name for name, section in sections.items() if section["missing_sources"] == 0
            ],
            "not_ready_manifests": [
                name for name, section in sections.items() if section["missing_sources"] > 0
            ],
        },
        "manifests": sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/eval_reports/external_data_status_report.json",
    )
    args = parser.parse_args()

    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
