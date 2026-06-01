from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import write_jsonl


DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
SECTION_ALIASES = {
    "education": ("教育背景", "教育经历"),
    "skills": ("专业技能", "核心技能", "技能"),
    "internships": ("实习经历", "工作经历"),
    "projects": ("项目经验", "项目经历"),
    "strengths": ("其他亮点", "优势", "自我评价"),
}


def fetch_rows(
    dataset: str,
    split: str,
    limit: int,
    batch_size: int = 100,
    request_interval: float = 0.25,
    max_retries: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(0, limit, batch_size):
        length = min(batch_size, limit - offset)
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        request = urllib.request.Request(
            f"{DATASETS_SERVER}?{query}",
            headers={"User-Agent": "job-match-tune/1.0"},
        )
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt + 1 >= max_retries:
                    raise
                time.sleep(2 ** attempt)
            except urllib.error.URLError:
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(2 ** attempt)
        time.sleep(request_interval)
        batch = [item["row"] for item in payload.get("rows") or []]
        rows.extend(batch)
        if len(batch) < length:
            break
    return rows


def download_file(url: str, output_path: Path, max_retries: int = 5) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "job-match-tune/1.0"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, output_path.open("wb") as out:
                shutil.copyfileobj(response, out)
            return
        except (urllib.error.HTTPError, urllib.error.URLError):
            if attempt + 1 >= max_retries:
                raise
            time.sleep(2 ** attempt)


def _clean_markdown_line(line: str) -> str:
    return re.sub(r"^[\s#>*-]+", "", line).replace("**", "").strip()


def _extract_sections(text: str) -> dict[str, list[str]]:
    sections = {key: [] for key in SECTION_ALIASES}
    active = ""
    for raw_line in text.splitlines():
        line = _clean_markdown_line(raw_line)
        if not line or line == "---":
            continue
        matched = ""
        for key, aliases in SECTION_ALIASES.items():
            if any(alias in line for alias in aliases) and len(line) <= 24:
                matched = key
                break
        if matched:
            active = matched
            continue
        if active and len(line) >= 2:
            sections[active].append(line)
    return {key: values[:12] for key, values in sections.items()}


def convert_faircv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        text = str(row.get("content") or "").strip()
        if not text:
            continue
        sections = _extract_sections(text)
        converted.append(
            {
                "text": text,
                "label": {
                    "目标岗位": str(metadata.get("position") or "").strip(),
                    "教育背景": sections["education"],
                    "核心技能": sections["skills"],
                    "实习经历": sections["internships"],
                    "项目经历": sections["projects"],
                    "优势标签": sections["strengths"],
                },
                "meta": {
                    "source_name": "OhMyKing/FairCV",
                    "source_kind": "public_synthetic_resume",
                },
            }
        )
    return converted


def convert_resume_ner_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "tokens": row.get("tokens") or [],
            "ner_tags": row.get("ner_tags") or [],
        }
        for row in rows
        if row.get("tokens")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faircv-limit", type=int, default=1000)
    parser.add_argument("--resume-ner-limit", type=int, default=3821)
    parser.add_argument("--out-dir", default="data/external/public_resume_exports")
    parser.add_argument("--skip-faircv", action="store_true")
    parser.add_argument("--skip-resume-ner", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_faircv:
        faircv_rows = fetch_rows("OhMyKing/FairCV", "train", args.faircv_limit)
        faircv_out = out_dir / "faircv_resume_parse.jsonl"
        write_jsonl(faircv_out, convert_faircv_rows(faircv_rows))
        print(f"faircv: {len(faircv_rows)} -> {faircv_out}")

    if not args.skip_resume_ner:
        resume_ner_out = out_dir / "resume_ner_train.parquet"
        download_file(
            "https://huggingface.co/datasets/PassbyGrocer/resume-ner/resolve/main/data/train-00000-of-00001.parquet",
            resume_ner_out,
        )
        print(f"resume_ner parquet -> {resume_ner_out}")


if __name__ == "__main__":
    main()
