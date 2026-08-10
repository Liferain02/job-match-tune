from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.utils.io import write_text


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_enabled_sources(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    sources = payload.get("sources") or []
    if not isinstance(sources, list):
        raise ValueError(f"Invalid source manifest: {path}")
    return [source for source in sources if bool(source.get("enabled", True))]


def _required_text(source: dict[str, Any], field: str) -> str:
    value = str(source.get(field) or "").strip()
    if not value:
        raise ValueError(f"Source {source.get('name') or '<unnamed>'} is missing {field}")
    return value


def verify_or_download_source(
    source: dict[str, Any],
    *,
    verify_only: bool = False,
    repair: bool = False,
) -> dict[str, Any]:
    name = _required_text(source, "name")
    target = Path(_required_text(source, "local_path"))
    url = _required_text(source, "source_url")
    expected = _required_text(source, "artifact_sha256").lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"Source {name} has invalid artifact_sha256")

    if target.exists():
        actual = sha256_file(target)
        if actual == expected:
            return {"name": name, "path": str(target), "status": "verified_cached", "sha256": actual}
        if verify_only or not repair:
            raise ValueError(
                f"Checksum mismatch for {name}: expected={expected} actual={actual} path={target}"
            )
    elif verify_only:
        raise FileNotFoundError(f"Missing artifact for {name}: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "JobMatchTune/verified-source"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        actual = sha256_file(temporary)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch after download for {name}: expected={expected} actual={actual}"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"name": name, "path": str(target), "status": "downloaded", "sha256": expected}


def process_manifest(
    manifest: str | Path,
    *,
    verify_only: bool = False,
    repair: bool = False,
) -> list[dict[str, Any]]:
    return [
        verify_or_download_source(source, verify_only=verify_only, repair=repair)
        for source in load_enabled_sources(manifest)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并校验来源清单中的固定版本文件")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--report-out")
    args = parser.parse_args()

    results = process_manifest(
        args.manifest,
        verify_only=args.verify_only,
        repair=args.repair,
    )
    rendered = json.dumps({"manifest": args.manifest, "artifacts": results}, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report_out:
        write_text(args.report_out, rendered + "\n")


if __name__ == "__main__":
    main()
