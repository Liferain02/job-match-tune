from __future__ import annotations

import re
from typing import Any

from jobmatch_tune.preprocess.jd_field_rules import merge_unique


EDUCATION_ORDER = {
    "中专": 1,
    "大专": 2,
    "本科": 3,
    "硕士": 4,
    "博士": 5,
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_skill_key(skill: str) -> str:
    return _normalize_text(skill).lower()


def _extract_years(text: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    matches = re.findall(r"([0-9]+)\s*年", normalized)
    if not matches:
        return 0
    return max(int(item) for item in matches)


def _extract_education_rank(text: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    for keyword, rank in sorted(EDUCATION_ORDER.items(), key=lambda item: item[1], reverse=True):
        if keyword in normalized:
            return rank
    return 0


def _direction_matches(jd_direction: str, resume_direction: str) -> bool:
    left = _normalize_text(jd_direction)
    right = _normalize_text(resume_direction)
    if not left or not right:
        return False
    if left == right:
        return True
    return left in right or right in left


def _skill_lists(jd_data: dict[str, Any], resume_data: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    jd_skills = merge_unique([_normalize_text(item) for item in jd_data.get("必备技能", []) if _normalize_text(item)])
    resume_skills = merge_unique([_normalize_text(item) for item in resume_data.get("核心技能", []) if _normalize_text(item)])
    resume_keys = {_normalize_skill_key(item): item for item in resume_skills}
    matched = [skill for skill in jd_skills if _normalize_skill_key(skill) in resume_keys]
    missing = [skill for skill in jd_skills if _normalize_skill_key(skill) not in resume_keys]
    return jd_skills, matched, missing


def _match_projects(jd_skills: list[str], resume_data: dict[str, Any], jd_direction: str) -> list[str]:
    project_lines = []
    for key in ("项目经历", "实习经历"):
        project_lines.extend([_normalize_text(item) for item in resume_data.get(key, []) if _normalize_text(item)])
    if not project_lines:
        return []
    jd_keywords = [item for item in jd_skills if _normalize_text(item)]
    if _normalize_text(jd_direction):
        jd_keywords.append(_normalize_text(jd_direction))
    matched = []
    for line in project_lines:
        lowered = line.lower()
        if any(keyword.lower() in lowered for keyword in jd_keywords):
            matched.append(line)
    return merge_unique(matched)


def _score_level(score: int) -> str:
    if score >= 85:
        return "高匹配"
    if score >= 65:
        return "较匹配"
    if score >= 45:
        return "基本匹配"
    return "低匹配"


def compute_match_rule_result(
    jd_data: dict[str, Any],
    resume_data: dict[str, Any],
    *,
    jd_text: str = "",
    resume_text: str = "",
) -> dict[str, Any]:
    jd_direction = _normalize_text(jd_data.get("岗位方向"))
    resume_direction = _normalize_text(resume_data.get("目标岗位"))
    direction_match = _direction_matches(jd_direction, resume_direction)

    jd_skills, matched_skills, missing_skills = _skill_lists(jd_data, resume_data)
    matched_projects = _match_projects(jd_skills, resume_data, jd_direction)

    jd_education_rank = _extract_education_rank(jd_data.get("学历要求"))
    resume_education_rank = max(
        [_extract_education_rank(item) for item in resume_data.get("教育背景", [])] + [_extract_education_rank(resume_text)]
    )
    education_match = jd_education_rank == 0 or resume_education_rank >= jd_education_rank

    jd_years = _extract_years(jd_data.get("经验要求"))
    resume_years = max(_extract_years(resume_text), _extract_years("\n".join(resume_data.get("实习经历", []) + resume_data.get("项目经历", []))))
    experience_match = jd_years == 0 or (resume_years > 0 and resume_years >= jd_years)

    score = 0
    score += 20 if direction_match else 0
    if jd_skills:
        score += round(45 * (len(matched_skills) / len(jd_skills)))
    else:
        score += 20
    score += 10 if education_match else 0
    score += 15 if experience_match else 0
    score += min(10, len(matched_projects) * 5)
    score = max(0, min(score, 100))

    return {
        "匹配分数": score,
        "匹配等级": _score_level(score),
        "岗位方向匹配": direction_match,
        "学历匹配": education_match,
        "经验匹配": experience_match,
        "命中技能": matched_skills,
        "缺失技能": missing_skills,
        "命中项目": matched_projects,
    }
