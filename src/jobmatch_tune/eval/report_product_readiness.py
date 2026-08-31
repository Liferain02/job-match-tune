from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import write_text


DEFAULT_THRESHOLDS = {
    "jd_json_valid_rate": 0.98,
    "jd_list_f1": 0.98,
    "jd_direction_exact_match": 0.95,
    "jd_experience_exact_match": 0.98,
    "jd_education_exact_match": 0.98,
    "resume_json_valid_rate": 0.99,
    "resume_list_f1": 0.98,
    "resume_strength_f1": 0.98,
    "resume_direction_exact_match": 0.98,
    "match_parse_success_rate": 0.99,
    "match_analysis_json_valid_rate": 0.99,
    "match_hit_skill_f1": 0.85,
    "match_missing_skill_f1": 0.89,
    "match_level_exact_match": 0.78,
    "match_level_macro_f1": 0.75,
    "match_direction_exact_match": 0.90,
    "match_education_exact_match": 0.98,
    "match_experience_exact_match": 0.98,
}


def read_report(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    return json.loads(file_path.read_text(encoding="utf-8"))


def _overall(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("overall") or report


def _metric(report: dict[str, Any], field: str, key: str) -> float:
    metrics = _overall(report).get("field_metrics") or {}
    value = (metrics.get(field) or {}).get(key)
    return float(value or 0.0)


def _preferred_list_f1(report: dict[str, Any], field: str) -> float:
    metrics = (_overall(report).get("field_metrics") or {}).get(field) or {}
    return float(metrics.get("micro_f1", metrics.get("f1", 0.0)) or 0.0)


def _value(report: dict[str, Any], key: str) -> float:
    return float((_overall(report).get(key) or 0.0))


def _check(name: str, actual: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "actual": actual,
        "threshold": threshold,
        "passed": actual >= threshold,
    }


def build_product_readiness_report(
    *,
    jd_report: dict[str, Any],
    resume_report: dict[str, Any],
    match_report: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    checks = [
        _check("jd_json_valid_rate", _value(jd_report, "json_valid_rate"), limits["jd_json_valid_rate"]),
        _check("jd_core_responsibility_f1", _preferred_list_f1(jd_report, "核心职责"), limits["jd_list_f1"]),
        _check("jd_required_skill_f1", _preferred_list_f1(jd_report, "必备技能"), limits["jd_list_f1"]),
        _check("jd_bonus_f1", _preferred_list_f1(jd_report, "加分项"), limits["jd_list_f1"]),
        _check(
            "jd_direction_exact_match",
            _metric(jd_report, "岗位方向", "exact_match"),
            limits["jd_direction_exact_match"],
        ),
        _check(
            "jd_experience_exact_match",
            _metric(jd_report, "经验要求", "exact_match"),
            limits["jd_experience_exact_match"],
        ),
        _check(
            "jd_education_exact_match",
            _metric(jd_report, "学历要求", "exact_match"),
            limits["jd_education_exact_match"],
        ),
        _check(
            "resume_json_valid_rate",
            _value(resume_report, "json_valid_rate"),
            limits["resume_json_valid_rate"],
        ),
        _check("resume_education_f1", _preferred_list_f1(resume_report, "教育背景"), limits["resume_list_f1"]),
        _check("resume_skill_f1", _preferred_list_f1(resume_report, "核心技能"), limits["resume_list_f1"]),
        _check("resume_internship_f1", _preferred_list_f1(resume_report, "实习经历"), limits["resume_list_f1"]),
        _check("resume_project_f1", _preferred_list_f1(resume_report, "项目经历"), limits["resume_list_f1"]),
        _check("resume_strength_f1", _preferred_list_f1(resume_report, "优势标签"), limits["resume_strength_f1"]),
        _check(
            "resume_direction_exact_match",
            _metric(resume_report, "目标岗位", "exact_match"),
            limits["resume_direction_exact_match"],
        ),
        _check(
            "match_parse_success_rate",
            _value(match_report, "jd_resume_parse_success_rate"),
            limits["match_parse_success_rate"],
        ),
        _check(
            "match_analysis_json_valid_rate",
            _value(match_report, "analysis_json_valid_rate"),
            limits["match_analysis_json_valid_rate"],
        ),
        _check("match_hit_skill_f1", _preferred_list_f1(match_report, "命中技能"), limits["match_hit_skill_f1"]),
        _check("match_missing_skill_f1", _preferred_list_f1(match_report, "缺失技能"), limits["match_missing_skill_f1"]),
        _check(
            "match_level_exact_match",
            _metric(match_report, "匹配等级", "exact_match"),
            limits["match_level_exact_match"],
        ),
        _check(
            "match_level_macro_f1",
            float(((_overall(match_report).get("decision_metrics") or {}).get("macro_f1") or 0.0)),
            limits["match_level_macro_f1"],
        ),
        _check(
            "match_direction_exact_match",
            _metric(match_report, "岗位方向匹配", "exact_match"),
            limits["match_direction_exact_match"],
        ),
        _check(
            "match_education_exact_match",
            _metric(match_report, "学历匹配", "exact_match"),
            limits["match_education_exact_match"],
        ),
        _check(
            "match_experience_exact_match",
            _metric(match_report, "经验匹配", "exact_match"),
            limits["match_experience_exact_match"],
        ),
    ]
    not_ready = [item for item in checks if not item["passed"]]
    match_evaluation_validity = str(match_report.get("evaluation_validity") or "missing")
    match_decision_ready = bool(
        (match_report.get("dataset_profile") or {}).get("decision_evaluation_ready")
    )
    evidence_checks = [
        {
            "name": "jd_blind_holdout",
            "actual": str(jd_report.get("evaluation_validity") or "missing"),
            "expected": "blind_holdout",
            "passed": jd_report.get("evaluation_validity") == "blind_holdout",
        },
        {
            "name": "resume_blind_holdout",
            "actual": str(resume_report.get("evaluation_validity") or "missing"),
            "expected": "blind_holdout",
            "passed": resume_report.get("evaluation_validity") == "blind_holdout",
        },
        {
            "name": "match_blind_holdout",
            "actual": match_evaluation_validity,
            "expected": "blind_holdout",
            "passed": match_evaluation_validity == "blind_holdout",
        },
        {
            "name": "match_level_class_coverage",
            "actual": match_decision_ready,
            "expected": True,
            "passed": match_decision_ready,
        },
    ]
    failed_evidence_checks = [item for item in evidence_checks if not item["passed"]]
    return {
        "ready_for_user": not not_ready and not failed_evidence_checks,
        "engineering_regression_passed": not not_ready,
        "evaluation_evidence_ready": not failed_evidence_checks,
        "num_checks": len(checks),
        "num_failed_checks": len(not_ready),
        "checks": checks,
        "not_ready_checks": not_ready,
        "evidence_checks": evidence_checks,
        "failed_evidence_checks": failed_evidence_checks,
        "thresholds": limits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd-report", required=True)
    parser.add_argument("--resume-report", required=True)
    parser.add_argument("--match-report", required=True)
    parser.add_argument("--out", default="outputs/eval_reports/product_readiness_report.json")
    args = parser.parse_args()

    report = build_product_readiness_report(
        jd_report=read_report(args.jd_report),
        resume_report=read_report(args.resume_report),
        match_report=read_report(args.match_report),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
