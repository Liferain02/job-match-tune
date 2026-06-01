from jobmatch_tune.dataset.jd_quality_risk import extract_title, risk_reasons, risk_score
from jobmatch_tune.eval.report_jd_quality_risks import build_risk_report


def _row(title: str, assistant: str, tier: str = "quality_weak") -> dict:
    return {
        "id": "row_1_jd_parse",
        "task_type": "jd_parse",
        "meta": {"quality_tier": tier, "quality_reason": "test"},
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"JD：\n岗位名称：{title}\n岗位职责：负责系统开发"},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_extract_title_from_prompt() -> None:
    assert extract_title("岗位名称：后端开发工程师\n岗位职责：开发") == "后端开发工程师"


def test_risk_reasons_flags_quality_weak_missing_fields() -> None:
    row = _row(
        "少儿编程老师",
        '{"岗位方向":"后端开发","核心职责":[],"必备技能":[],"学历要求":"","经验要求":""}',
    )

    reasons = risk_reasons(row)

    assert "quality_weak_tier" in reasons
    assert "suspicious_title_keyword" in reasons
    assert "quality_weak_missing_core_field" in reasons
    assert risk_score(reasons) >= 4


def test_risk_reasons_flags_field_boundary_leakage() -> None:
    row = _row(
        "后端开发工程师",
        '{"岗位方向":"后端开发","核心职责":["负责服务开发。任职要求：本科及以上，熟悉Java。"],'
        '"必备技能":["Java"],"学历要求":"本科","经验要求":"3年"}',
        tier="strict",
    )

    reasons = risk_reasons(row)

    assert "responsibility_contains_requirement_marker" in reasons
    assert risk_score(reasons) >= 3


def test_build_risk_report_counts_high_risk_samples() -> None:
    rows = [
        _row(
            "少儿编程老师",
            '{"岗位方向":"后端开发","核心职责":[],"必备技能":[],"学历要求":"","经验要求":""}',
        )
    ]

    report, samples = build_risk_report(rows, sample_limit=10)

    assert report["total"] == 1
    assert report["high_risk_samples"] == 1
    assert samples[0]["risk_score"] >= 4
