from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from jobmatch_tune.resume.privacy import sanitize_resume_text_for_training
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


TECHNICAL_DIRECTIONS = {
    "AI Infra",
    "AI基础设施",
    "AI应用",
    "AI应用开发",
    "Android客户端开发",
    "HPC",
    "iOS客户端开发",
    "前端",
    "前端开发",
    "后端",
    "后端开发",
    "大模型应用开发",
    "安全",
    "安全工程",
    "客户端",
    "客户端开发",
    "嵌入式",
    "嵌入式开发",
    "数据",
    "数据开发",
    "数据工程",
    "智驾",
    "汽车软件/智驾研发",
    "测试",
    "测试开发",
    "硬件",
    "硬件研发",
    "算法",
    "算法工程",
    "网络",
    "网络与基础设施",
    "运维",
    "运维开发",
    "高性能计算",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def is_technical_direction(value: Any) -> bool:
    return _normalize_text(value) in TECHNICAL_DIRECTIONS


def is_usable_public_resume_row(row: dict[str, Any]) -> bool:
    if str(row.get("task") or "") != "resume_parse":
        return False
    meta = row.get("meta") or {}
    if str(meta.get("license_status") or "").lower() in {
        "prohibited",
        "training_prohibited",
        "no_derivatives",
    }:
        return False
    if str(meta.get("intended_usage") or "").lower() not in {
        "training",
        "sft_training",
        "training_and_evaluation",
    }:
        return False
    language = str(meta.get("language") or "").lower()
    if not language.startswith("zh"):
        return False
    text = _normalize_text(row.get("text"))
    if len(text) < 80:
        return False
    label = row.get("label") or {}
    signals = 0
    if _normalize_text(label.get("目标岗位")):
        signals += 1
    if label.get("教育背景"):
        signals += 1
    if label.get("核心技能"):
        signals += 1
    if label.get("项目经历"):
        signals += 1
    if label.get("实习经历"):
        signals += 1
    return signals >= 3


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        row = dict(row)
        text = sanitize_resume_text_for_training(_normalize_text(row.get("text")))
        row["text"] = text
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_combined_rows(
    manual_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined = [
        row
        for row in manual_rows
        if is_technical_direction((row.get("label") or {}).get("目标岗位"))
    ]
    for row in public_rows:
        if (
            is_usable_public_resume_row(row)
            and is_technical_direction((row.get("label") or {}).get("目标岗位"))
        ):
            combined.append(
                {
                    "id": row["id"],
                    "task": "resume_parse",
                    "source_type": row.get("source_type", "public_text"),
                    "source_group": row.get("source_group", row["id"]),
                    "text": row["text"],
                    "label": row.get("label") or {},
                    "meta": row.get("meta") or {},
                }
            )
    return deduplicate_rows(combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-input", default="data/eval/resume_manual_train_pool.jsonl")
    parser.add_argument(
        "--public-input", default="data/external/public_resume_imports_zh_tech.jsonl"
    )
    parser.add_argument("--out", default="data/eval/resume_train_pool_combined.jsonl")
    args = parser.parse_args()

    manual_rows = list(read_jsonl(args.manual_input))
    public_rows = list(read_jsonl(args.public_input)) if Path(args.public_input).exists() else []
    combined = build_combined_rows(manual_rows, public_rows)
    write_jsonl(args.out, combined)
    print(
        "manual="
        f"{len(manual_rows)} public={len(public_rows)} combined={len(combined)}"
    )


if __name__ == "__main__":
    main()
