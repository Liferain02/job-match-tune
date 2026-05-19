from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_text


def top_counter(rows: list[dict[str, Any]], getter, limit: int = 10) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = getter(row)
        if not value:
            continue
        if isinstance(value, list):
            counter.update(str(item) for item in value if item)
        else:
            counter[str(value)] += 1
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return list(read_jsonl(file_path))


def profile_jd(path: str | Path) -> dict[str, Any]:
    rows = load_rows(path)
    return {
        "path": str(path),
        "count": len(rows),
        "top_sources": top_counter(rows, lambda r: r.get("source")),
        "top_titles": top_counter(rows, lambda r: r.get("job_title"), limit=15),
        "languages": top_counter(rows, lambda r: (r.get("meta") or {}).get("language")),
        "top_companies": top_counter(rows, lambda r: r.get("company"), limit=15),
    }


def profile_resume(path: str | Path) -> dict[str, Any]:
    rows = load_rows(path)
    return {
        "path": str(path),
        "count": len(rows),
        "top_source_types": top_counter(rows, lambda r: r.get("source_type")),
        "top_target_jobs": top_counter(rows, lambda r: (r.get("label") or {}).get("目标岗位")),
        "top_skills": top_counter(rows, lambda r: (r.get("label") or {}).get("核心技能"), limit=20),
    }


def profile_match(path: str | Path) -> dict[str, Any]:
    rows = load_rows(path)
    return {
        "path": str(path),
        "count": len(rows),
        "top_source_types": top_counter(rows, lambda r: r.get("source_type")),
        "top_match_levels": top_counter(rows, lambda r: (r.get("label") or {}).get("匹配等级")),
        "raw_label_presence": {
            "with_raw_label": sum(
                1 for r in rows if (r.get("label") or {}).get("raw_label") not in (None, "")
            ),
            "with_raw_score": sum(
                1 for r in rows if (r.get("label") or {}).get("raw_score") not in (None, "")
            ),
        },
    }


def build_report() -> dict[str, Any]:
    return {
        "jd": profile_jd("data/eval/jd_train_pool_combined.jsonl"),
        "resume": profile_resume("data/eval/resume_train_pool_combined.jsonl"),
        "match": profile_match("data/eval/match_train_pool_combined.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/eval_reports/current_pool_profiles.json")
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
