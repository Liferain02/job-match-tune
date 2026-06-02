from __future__ import annotations

import argparse
import json
from typing import Any

from jobmatch_tune.eval.report_product_readiness import build_product_readiness_report, read_report
from jobmatch_tune.utils.io import write_text


def _check_index(readiness_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in readiness_report["checks"]}


def build_product_regression_report(
    *,
    candidate_jd_report: dict[str, Any],
    candidate_resume_report: dict[str, Any],
    candidate_match_report: dict[str, Any],
    baseline_jd_report: dict[str, Any],
    baseline_resume_report: dict[str, Any],
    baseline_match_report: dict[str, Any],
    max_regression: float = 0.005,
) -> dict[str, Any]:
    candidate_readiness = build_product_readiness_report(
        jd_report=candidate_jd_report,
        resume_report=candidate_resume_report,
        match_report=candidate_match_report,
    )
    baseline_readiness = build_product_readiness_report(
        jd_report=baseline_jd_report,
        resume_report=baseline_resume_report,
        match_report=baseline_match_report,
    )
    baseline_checks = _check_index(baseline_readiness)

    checks = []
    epsilon = 1e-12
    for candidate_check in candidate_readiness["checks"]:
        name = candidate_check["name"]
        baseline_actual = float((baseline_checks.get(name) or {}).get("actual") or 0.0)
        candidate_actual = float(candidate_check["actual"])
        delta = candidate_actual - baseline_actual
        checks.append(
            {
                "name": name,
                "candidate": candidate_actual,
                "baseline": baseline_actual,
                "delta": delta,
                "max_regression": max_regression,
                "passed": delta + epsilon >= -max_regression,
            }
        )

    failed = [item for item in checks if not item["passed"]]
    return {
        "no_product_regression": not failed,
        "candidate_ready_for_user": candidate_readiness["ready_for_user"],
        "baseline_ready_for_user": baseline_readiness["ready_for_user"],
        "ready_to_promote": candidate_readiness["ready_for_user"] and not failed,
        "num_checks": len(checks),
        "num_failed_regressions": len(failed),
        "failed_regressions": failed,
        "checks": checks,
        "max_regression": max_regression,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-jd-report", required=True)
    parser.add_argument("--candidate-resume-report", required=True)
    parser.add_argument("--candidate-match-report", required=True)
    parser.add_argument("--baseline-jd-report", required=True)
    parser.add_argument("--baseline-resume-report", required=True)
    parser.add_argument("--baseline-match-report", required=True)
    parser.add_argument("--max-regression", type=float, default=0.005)
    parser.add_argument("--out", default="outputs/eval_reports/product_regression_report.json")
    args = parser.parse_args()

    report = build_product_regression_report(
        candidate_jd_report=read_report(args.candidate_jd_report),
        candidate_resume_report=read_report(args.candidate_resume_report),
        candidate_match_report=read_report(args.candidate_match_report),
        baseline_jd_report=read_report(args.baseline_jd_report),
        baseline_resume_report=read_report(args.baseline_resume_report),
        baseline_match_report=read_report(args.baseline_match_report),
        max_regression=args.max_regression,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
