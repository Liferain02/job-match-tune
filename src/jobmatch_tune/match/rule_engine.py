from __future__ import annotations

import re
from typing import Any

from jobmatch_tune.inference.postprocess_json import load_label_schema
from jobmatch_tune.preprocess.jd_field_rules import extract_skills_from_text, merge_unique


EDUCATION_ORDER = {
    "中专": 1,
    "大专": 2,
    "本科": 3,
    "研究生": 4,
    "硕士": 4,
    "博士": 5,
}

CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

MATCH_EVIDENCE_SKILL_ALIASES = {
    "PostgreSQL": ["postgres"],
    "RTOS": [],
    "自动驾驶仿真": [],
    "Web安全": ["Web 安全"],
    "代码审计": [],
    "流量分析": [],
    "Kotlin": [],
    "Jetpack": [],
    "性能优化": ["首屏优化"],
    "RAG": ["检索增强问答"],
    "Python": ["Py thon"],
    "Kubernetes": ["Kubernet es"],
    "C++": ["C + +"],
    "MySQL": ["My SOL"],
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_skill_key(skill: str) -> str:
    return _normalize_text(skill).lower()


def _normalize_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        text = " ".join(item for nested in value.values() for item in _normalize_items(nested))
        return [_normalize_text(text)] if _normalize_text(text) else []
    if isinstance(value, (list, tuple, set)):
        return [item for nested in value for item in _normalize_items(nested)]
    text = _normalize_text(value)
    return [text] if text else []


def _extract_years(text: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    matches = re.finditer(r"([0-9]+|[零一二两三四五六七八九十]+)\s*年", normalized)
    durations = []
    for match in matches:
        following_context = normalized[match.end() : match.end() + 12]
        if re.search(r"(?:毕业|入学|应届|在校|出生|届)", following_context):
            continue
        years = _parse_number(match.group(1))
        # “25年校招” abbreviates the 2025 campus-recruitment cohort. Keep
        # genuine phrases such as “2年校招经验”, but reject two-digit cohorts.
        if years >= 20 and re.match(r"(?:校招|社招|春招|秋招)", following_context):
            continue
        durations.append(years)
    # Calendar years such as “2023年毕业” are dates, not 2,023 years of experience.
    reasonable_durations = [years for years in durations if 0 < years <= 50]
    return max(reasonable_durations, default=0)


def _extract_required_years(text: str) -> int:
    normalized = _normalize_text(text)
    range_match = re.search(
        r"([0-9]+|[零一二两三四五六七八九十]+)\s*[-~～至到]\s*"
        r"([0-9]+|[零一二两三四五六七八九十]+)\s*年",
        normalized,
    )
    if range_match:
        lower = _parse_number(range_match.group(1))
        upper = _parse_number(range_match.group(2))
        plausible = [years for years in (lower, upper) if 0 < years <= 50]
        return min(plausible, default=0)
    return _extract_years(normalized)


def _parse_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    if "十" not in value:
        return CHINESE_DIGITS.get(value, 0)
    tens, ones = value.split("十", 1)
    return (CHINESE_DIGITS.get(tens, 1) * 10) + CHINESE_DIGITS.get(ones, 0)


def _extract_education_rank(text: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    for keyword, rank in sorted(EDUCATION_ORDER.items(), key=lambda item: item[1], reverse=True):
        if keyword in normalized:
            return rank
    return 0


def _extract_required_education_rank(text: str) -> int:
    normalized = _normalize_text(text)
    if "优先" in normalized:
        return 0
    return _extract_education_rank(normalized)


def _direction_matches(jd_direction: str, resume_direction: str) -> bool:
    left = _normalize_text(jd_direction)
    right = _normalize_text(resume_direction)
    if not left or not right:
        return False
    if left == right:
        return True
    return left in right or right in left


def _skill_lists(
    jd_data: dict[str, Any],
    resume_data: dict[str, Any],
    *,
    jd_text: str,
    resume_text: str,
) -> tuple[list[str], list[str], list[str]]:
    match_evidence_schema = {"skill_alias": MATCH_EVIDENCE_SKILL_ALIASES}
    jd_skills = merge_unique(
        _normalize_items(jd_data.get("必备技能"))
        + extract_skills_from_text(jd_text, match_evidence_schema)
    )
    resume_evidence = "\n".join(
        _normalize_items(resume_data.get("核心技能"))
        + _normalize_items(resume_data.get("项目经历"))
        + _normalize_items(resume_data.get("实习经历"))
        + [resume_text]
    )
    resume_skills = merge_unique(
        _normalize_items(resume_data.get("核心技能"))
        + extract_skills_from_text(resume_evidence, load_label_schema())
        + extract_skills_from_text(resume_evidence, match_evidence_schema)
    )
    resume_keys = {_normalize_skill_key(item): item for item in resume_skills}
    matched = [skill for skill in jd_skills if _normalize_skill_key(skill) in resume_keys]
    missing = [skill for skill in jd_skills if _normalize_skill_key(skill) not in resume_keys]
    return jd_skills, matched, missing


def _match_projects(jd_skills: list[str], resume_data: dict[str, Any]) -> list[str]:
    project_lines = []
    for key in ("项目经历", "实习经历"):
        project_lines.extend(_normalize_items(resume_data.get(key)))
    if not project_lines:
        return []
    jd_keywords = [item for item in jd_skills if _normalize_text(item)]
    if not jd_keywords:
        return []
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

    jd_skills, matched_skills, missing_skills = _skill_lists(
        jd_data,
        resume_data,
        jd_text=jd_text,
        resume_text=resume_text,
    )
    matched_projects = _match_projects(jd_skills, resume_data)

    jd_education_rank = _extract_required_education_rank(jd_data.get("学历要求"))
    resume_education_rank = max(
        [_extract_education_rank(item) for item in _normalize_items(resume_data.get("教育背景"))]
        + [_extract_education_rank(resume_text)]
    )
    education_match = jd_education_rank == 0 or resume_education_rank >= jd_education_rank

    jd_years = _extract_required_years(jd_data.get("经验要求"))
    experience_text = "\n".join(
        _normalize_items(resume_data.get("实习经历")) + _normalize_items(resume_data.get("项目经历"))
    )
    resume_years = max(_extract_years(resume_text), _extract_years(experience_text))
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
