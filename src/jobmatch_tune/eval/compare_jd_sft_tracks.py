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


TRACK_METRIC_KEYS = (
    "total_samples",
    "json_valid_rate",
    "avg_responsibility_count",
    "avg_skill_count",
    "education_coverage",
    "experience_coverage",
)


def build_delta(base_report: dict[str, Any], compare_report: dict[str, Any]) -> dict[str, Any]:
    return {key: compare_report[key] - base_report[key] for key in TRACK_METRIC_KEYS}


def build_report(
    strict_dir: str | Path,
    bootstrap_dir: str | Path,
    strict_plus_dir: str | Path | None = None,
) -> dict[str, Any]:
    strict_rows = load_split_rows(strict_dir)
    bootstrap_rows = load_split_rows(bootstrap_dir)
    strict_report = compute_report(strict_rows)
    bootstrap_report = compute_report(bootstrap_rows)
    strict_plus_report = None
    if strict_plus_dir:
        strict_plus_rows = load_split_rows(strict_plus_dir)
        strict_plus_report = compute_report(strict_plus_rows)

    report = {
        "strict": strict_report,
        "bootstrap": bootstrap_report,
        "delta_strict_to_bootstrap": build_delta(strict_report, bootstrap_report),
    }
    if strict_plus_report is not None:
        report["strict_plus"] = strict_plus_report
        report["delta_strict_to_strict_plus"] = build_delta(strict_report, strict_plus_report)
        report["delta_strict_plus_to_bootstrap"] = build_delta(strict_plus_report, bootstrap_report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="data/sft")
    parser.add_argument("--strict-plus-dir", default="data/sft_jd_strict_plus")
    parser.add_argument("--bootstrap-dir", default="data/sft_jd_bootstrap")
    parser.add_argument("--out", default="outputs/eval_reports/jd_sft_track_comparison.json")
    args = parser.parse_args()

    report = build_report(args.strict_dir, args.bootstrap_dir, args.strict_plus_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
