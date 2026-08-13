from __future__ import annotations

import pytest

from jobmatch_tune.match.scoring import MatchScoringPolicy, compute_score_breakdown


def test_default_scoring_policy_is_explicit_and_preserves_total_weight() -> None:
    policy = MatchScoringPolicy()
    assert sum(policy.as_dict()["权重"].values()) == 100
    assert policy.calibration_status == "heuristic"
    assert policy.score_semantics == "heuristic_compatibility_score"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction_weight": 19},
        {"high_threshold": 60, "medium_threshold": 65},
        {"basic_threshold": -1},
        {"no_required_skills_score": 46},
    ],
)
def test_invalid_scoring_policy_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        MatchScoringPolicy(**kwargs)


def test_score_breakdown_matches_historical_default_behavior() -> None:
    breakdown = compute_score_breakdown(
        direction_match=True,
        required_skill_count=4,
        matched_skill_count=3,
        education_match=True,
        experience_match=False,
        matched_project_count=2,
    )
    assert breakdown == {"方向": 20, "技能": 34, "学历": 10, "经验": 0, "项目": 10}
    assert sum(breakdown.values()) == 74
