from __future__ import annotations

from jobmatch_tune.eval.explanation_grounding import evaluate_explanation


def test_grounded_explanation_passes_both_deterministic_checks() -> None:
    result = evaluate_explanation(
        {
            "匹配等级": "较匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": False,
            "命中技能": ["Python"],
            "缺失技能": ["PyTorch"],
        },
        {
            "匹配结论": "整体较匹配。",
            "匹配优势": ["具备 Python 技能，方向一致"],
            "主要短板": ["经验条件有差距，缺失 PyTorch"],
            "简历优化建议": ["补充 PyTorch 项目证据"],
        },
    )
    assert result["structural_consistent"] is True
    assert result["evidence_grounded"] is True
    assert result["advice_validity"]["status"] == "not_evaluated"


def test_unsupported_skill_suggestion_is_rejected() -> None:
    result = evaluate_explanation(
        {"命中技能": ["Python"], "缺失技能": []},
        {"简历优化建议": ["建议补充 Kafka 项目"]},
    )
    assert result["evidence_grounded"] is False
    assert "简历优化建议:unsupported_skill:Kafka" in result["evidence_grounding_issues"]


def test_missing_skill_cannot_be_claimed_as_strength() -> None:
    result = evaluate_explanation(
        {"命中技能": [], "缺失技能": ["PyTorch"]},
        {"匹配优势": ["已经具备 PyTorch 能力"]},
    )
    assert result["evidence_grounded"] is False
    assert any("claims_missing_skill_as_strength" in item for item in result["evidence_grounding_issues"])


def test_structural_consistency_does_not_claim_advice_validity() -> None:
    result = evaluate_explanation(
        {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": [],
            "缺失技能": [],
        },
        {"匹配结论": "高度匹配", "匹配优势": [], "主要短板": []},
    )
    assert result["structural_consistent"] is True
    assert result["advice_validity"]["reason"] == "unsupported_by_current_data"
