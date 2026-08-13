from __future__ import annotations

from typing import Any

from jobmatch_tune.inference.postprocess_json import load_label_schema
from jobmatch_tune.preprocess.skill_canonicalization import extract_known_skills


ADVICE_VALIDITY_STATUS = {
    "status": "not_evaluated",
    "reason": "unsupported_by_current_data",
    "limitation": "没有真实用户反馈或投递结果，不能评估建议是否提高求职成功率。",
}


def _joined(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "")


def structural_contradictions(
    rule: dict[str, Any],
    analysis: dict[str, Any],
) -> list[str]:
    strengths = _joined(analysis.get("匹配优势"))
    gaps = _joined(analysis.get("主要短板"))
    conclusion = _joined(analysis.get("匹配结论"))
    contradictions: list[str] = []
    checks = [
        ("岗位方向匹配", "方向一致", "方向不一致"),
        ("学历匹配", "学历背景满足", "学历条件"),
        ("经验匹配", "经验背景满足", "经验条件"),
    ]
    for field, positive, negative in checks:
        if rule.get(field) is False and positive in strengths:
            contradictions.append(f"{field}:false_but_strength_positive")
        if rule.get(field) is True and negative in gaps and "差距" in gaps:
            contradictions.append(f"{field}:true_but_gap_negative")
    if rule.get("缺失技能") and "暂无明显硬性短板" in gaps:
        contradictions.append("missing_skills_but_no_hard_gap")
    level = str(rule.get("匹配等级") or "")
    if level == "高匹配" and any(word in conclusion for word in ("匹配度有限", "低匹配", "不匹配")):
        contradictions.append("high_level_but_negative_conclusion")
    if level == "低匹配" and any(word in conclusion for word in ("高度匹配", "整体较匹配")):
        contradictions.append("low_level_but_positive_conclusion")
    return contradictions


def evidence_grounding_issues(
    rule: dict[str, Any],
    analysis: dict[str, Any],
) -> list[str]:
    matched = set(str(item) for item in rule.get("命中技能") or [])
    missing = set(str(item) for item in rule.get("缺失技能") or [])
    known_evidence = matched | missing
    issues: list[str] = []

    for field in ("匹配优势", "主要短板", "简历优化建议"):
        text = _joined(analysis.get(field))
        for skill in extract_known_skills(text, load_label_schema()):
            if skill not in known_evidence:
                issues.append(f"{field}:unsupported_skill:{skill}")
            elif field == "匹配优势" and skill in missing:
                issues.append(f"{field}:claims_missing_skill_as_strength:{skill}")
            elif field == "主要短板" and skill in matched and any(
                cue in text for cue in ("缺失", "未掌握", "未体现", "欠缺")
            ):
                issues.append(f"{field}:claims_matched_skill_as_missing:{skill}")

    strengths = _joined(analysis.get("匹配优势"))
    gaps = _joined(analysis.get("主要短板"))
    if rule.get("学历匹配") is False and "学历满足" in strengths:
        issues.append("学历:unsupported_positive_claim")
    if rule.get("学历匹配") is True and "学历不满足" in gaps:
        issues.append("学历:unsupported_negative_claim")
    if rule.get("经验匹配") is False and "经验满足" in strengths:
        issues.append("经验:unsupported_positive_claim")
    if rule.get("经验匹配") is True and "经验不足" in gaps:
        issues.append("经验:unsupported_negative_claim")
    return list(dict.fromkeys(issues))


def evaluate_explanation(
    rule: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    contradictions = structural_contradictions(rule, analysis)
    grounding_issues = evidence_grounding_issues(rule, analysis)
    return {
        "structural_consistent": not contradictions,
        "structural_contradictions": contradictions,
        "evidence_grounded": not grounding_issues,
        "evidence_grounding_issues": grounding_issues,
        "advice_validity": dict(ADVICE_VALIDITY_STATUS),
    }
