from __future__ import annotations

from jobmatch_tune.inference.predict import build_prompt


def test_build_prompt_for_match() -> None:
    messages = build_prompt(
        "match",
        "岗位名称：AI应用开发工程师",
        resume_text="目标岗位：AI应用开发工程师",
        rule_result='{"匹配分数":82,"匹配等级":"较匹配"}',
    )
    assert messages[0]["role"] == "system"
    assert "规则评分结果" in messages[1]["content"]
    assert "匹配分数" in messages[1]["content"]
