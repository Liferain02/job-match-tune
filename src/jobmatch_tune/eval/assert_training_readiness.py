from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_report(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"training readiness report not found: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


READINESS_STAGES = ("sft", "jd_dpo", "product_dpo", "all")


def summarize_blockers(report: dict[str, Any], stage: str = "all") -> list[str]:
    if stage not in READINESS_STAGES:
        raise ValueError(f"unknown training stage: {stage}")
    summary = report.get("summary") or {}
    blockers: list[str] = []
    if stage in {"sft", "all"} and not summary.get("all_ready_for_sft"):
        not_ready_tasks = summary.get("not_ready_tasks") or []
        blockers.append(f"SFT not ready; tasks={not_ready_tasks}")
    if stage in {"jd_dpo", "all"} and not summary.get("ready_for_dpo"):
        blockers.append("JD preference DPO data is not ready")
    if stage in {"product_dpo", "all"} and not summary.get("ready_for_product_dpo"):
        blockers.append("product preference DPO data is not ready")
    resume = ((report.get("tasks") or {}).get("resume") or {})
    if stage in {"sft", "all"} and resume and not resume.get("privacy_ready", True):
        privacy_report = resume.get("privacy_report") or {}
        blockers.append(
            "resume privacy gate failed; "
            f"rows_with_pii={privacy_report.get('rows_with_pii', 'unknown')}"
        )
    return blockers


def assert_training_readiness(report: dict[str, Any], stage: str = "all") -> dict[str, Any]:
    summary = report.get("summary") or {}
    blockers = summarize_blockers(report, stage)
    stage_ready = {
        "sft": bool(summary.get("all_ready_for_sft")),
        "jd_dpo": bool(summary.get("ready_for_dpo")),
        "product_dpo": bool(summary.get("ready_for_product_dpo")),
        "all": bool(summary.get("all_ready_for_training")),
    }[stage]
    ready = stage_ready and not blockers
    return {
        "stage": stage,
        "ready": ready,
        "blockers": blockers,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="outputs/eval_reports/data_readiness_report.json")
    parser.add_argument("--stage", choices=READINESS_STAGES, default="all")
    args = parser.parse_args()

    result = assert_training_readiness(read_report(args.report), args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
