from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_text


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_profile(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "exists": False, "rows": 0, "sha256": ""}
    rows = list(read_jsonl(file_path))
    task_counts: Counter[str] = Counter()
    source_groups: set[str] = set()
    provenance_counts: Counter[str] = Counter()
    quality_tier_counts: Counter[str] = Counter()
    for row in rows:
        meta = row.get("meta") or {}
        task = str(meta.get("dataset_task") or row.get("task_type") or "unknown")
        task_counts[task] += 1
        source_group = str(row.get("source_group") or row.get("source_id") or row.get("id") or "")
        if source_group:
            source_groups.add(source_group)
        if meta.get("provenance"):
            provenance_counts[str(meta["provenance"])] += 1
        if meta.get("quality_tier"):
            quality_tier_counts[str(meta["quality_tier"])] += 1
    row_count = len(rows)
    return {
        "path": str(file_path),
        "exists": True,
        "rows": row_count,
        "sha256": file_sha256(file_path),
        "task_counts": dict(task_counts),
        "unique_source_groups": len(source_groups),
        "source_group_ratio": round(len(source_groups) / row_count, 4) if row_count else 0.0,
        "provenance_counts": dict(provenance_counts),
        "quality_tier_counts": dict(quality_tier_counts),
    }


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_profile() -> dict[str, Any]:
    status = _git_value(["status", "--short"])
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def readiness_summary(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "exists": False}
    report = json.loads(file_path.read_text(encoding="utf-8"))
    return {
        "path": str(file_path),
        "exists": True,
        "sha256": file_sha256(file_path),
        "summary": report.get("summary", {}),
    }


def build_run_manifest(
    *,
    stage: str,
    config_path: str,
    output_dir: str,
    train_file: str,
    valid_file: str,
    readiness_report: str = "outputs/eval_reports/data_readiness_report.json",
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "git": git_profile(),
        "config": {
            "path": config_path,
            "exists": Path(config_path).exists(),
            "sha256": file_sha256(config_path),
        },
        "datasets": {
            "train": jsonl_profile(train_file),
            "valid": jsonl_profile(valid_file),
        },
        "readiness": readiness_summary(readiness_report),
        "output_dir": output_dir,
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        },
        "cli_args": cli_args or {},
    }


def write_run_manifest(
    *,
    stage: str,
    config_path: str,
    output_dir: str,
    train_file: str,
    valid_file: str,
    readiness_report: str = "outputs/eval_reports/data_readiness_report.json",
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = build_run_manifest(
        stage=stage,
        config_path=config_path,
        output_dir=output_dir,
        train_file=train_file,
        valid_file=valid_file,
        readiness_report=readiness_report,
        cli_args=cli_args,
    )
    write_text(Path(output_dir) / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
