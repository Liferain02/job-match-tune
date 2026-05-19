from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jobmatch_tune.eval.audit_sft_dataset import compute_report
from jobmatch_tune.utils.io import read_jsonl, write_text


def load_split_rows(base_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        path = Path(base_dir) / f"{split}.jsonl"
        if path.exists():
            rows.extend(list(read_jsonl(path)))
    return rows


def build_report(strict_dir: str | Path, bootstrap_dir: str | Path) -> dict[str, Any]:
    strict_rows = load_split_rows(strict_dir)
    bootstrap_rows = load_split_rows(bootstrap_dir)
    strict_report = compute_report(strict_rows)
    bootstrap_report = compute_report(bootstrap_rows)

    return {
        "strict": strict_report,
        "bootstrap": bootstrap_report,
        "delta": {
            "total_samples": bootstrap_report["total_samples"] - strict_report["total_samples"],
            "json_valid_rate": bootstrap_report["json_valid_rate"] - strict_report["json_valid_rate"],
            "avg_responsibility_count": bootstrap_report["avg_responsibility_count"]
            - strict_report["avg_responsibility_count"],
            "avg_skill_count": bootstrap_report["avg_skill_count"] - strict_report["avg_skill_count"],
            "education_coverage": bootstrap_report["education_coverage"] - strict_report["education_coverage"],
            "experience_coverage": bootstrap_report["experience_coverage"] - strict_report["experience_coverage"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="data/sft")
    parser.add_argument("--bootstrap-dir", default="data/sft_jd_bootstrap")
    parser.add_argument("--out", default="outputs/eval_reports/jd_sft_track_comparison.json")
    args = parser.parse_args()

    report = build_report(args.strict_dir, args.bootstrap_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
