from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchScoringPolicy:
    direction_weight: int = 20
    skill_weight: int = 45
    education_weight: int = 10
    experience_weight: int = 15
    project_weight: int = 10
    no_required_skills_score: int = 20
    high_threshold: int = 85
    medium_threshold: int = 65
    basic_threshold: int = 45
    calibration_status: str = "heuristic"
    score_semantics: str = "heuristic_compatibility_score"

    def __post_init__(self) -> None:
        weights = (
            self.direction_weight,
            self.skill_weight,
            self.education_weight,
            self.experience_weight,
            self.project_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("scoring weights must be non-negative")
        if sum(weights) != 100:
            raise ValueError("scoring weights must sum to 100")
        if not (100 >= self.high_threshold > self.medium_threshold > self.basic_threshold >= 0):
            raise ValueError("thresholds must be strictly descending within 0..100")
        if not 0 <= self.no_required_skills_score <= self.skill_weight:
            raise ValueError("no_required_skills_score must be within the skill weight")
        if self.calibration_status != "heuristic":
            raise ValueError("the current policy is intentionally uncalibrated")

    def level_for(self, score: int) -> str:
        if score >= self.high_threshold:
            return "高匹配"
        if score >= self.medium_threshold:
            return "较匹配"
        if score >= self.basic_threshold:
            return "基本匹配"
        return "低匹配"

    def as_dict(self) -> dict[str, object]:
        return {
            "评分语义": self.score_semantics,
            "校准状态": self.calibration_status,
            "权重": {
                "方向": self.direction_weight,
                "技能": self.skill_weight,
                "学历": self.education_weight,
                "经验": self.experience_weight,
                "项目": self.project_weight,
            },
            "等级阈值": {
                "高匹配": self.high_threshold,
                "较匹配": self.medium_threshold,
                "基本匹配": self.basic_threshold,
            },
        }


DEFAULT_SCORING_POLICY = MatchScoringPolicy()


def compute_score_breakdown(
    *,
    direction_match: bool,
    required_skill_count: int,
    matched_skill_count: int,
    education_match: bool,
    experience_match: bool,
    matched_project_count: int,
    policy: MatchScoringPolicy = DEFAULT_SCORING_POLICY,
) -> dict[str, int]:
    if matched_skill_count < 0 or required_skill_count < 0:
        raise ValueError("skill counts must be non-negative")
    if matched_skill_count > required_skill_count:
        raise ValueError("matched_skill_count cannot exceed required_skill_count")
    skill_score = (
        round(policy.skill_weight * matched_skill_count / required_skill_count)
        if required_skill_count
        else policy.no_required_skills_score
    )
    return {
        "方向": policy.direction_weight if direction_match else 0,
        "技能": skill_score,
        "学历": policy.education_weight if education_match else 0,
        "经验": policy.experience_weight if experience_match else 0,
        "项目": min(policy.project_weight, matched_project_count * 5),
    }
