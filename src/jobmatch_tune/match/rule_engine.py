from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from jobmatch_tune.inference.postprocess_json import load_label_schema
from jobmatch_tune.match.direction_compatibility import evaluate_direction_compatibility
from jobmatch_tune.match.scoring import (
    DEFAULT_SCORING_POLICY,
    MatchScoringPolicy,
    compute_score_breakdown,
)
from jobmatch_tune.preprocess.jd_field_rules import extract_skills_from_text, merge_unique
from jobmatch_tune.preprocess.jd_skill_evidence import (
    collect_jd_skill_evidence,
    required_skills_from_evidence,
)
from jobmatch_tune.preprocess.skill_canonicalization import (
    canonicalize_skill_list,
    merge_skill_aliases,
)


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


@lru_cache(maxsize=1)
def _match_evidence_schema() -> dict[str, Any]:
    return merge_skill_aliases(
        load_label_schema(),
        {"skill_alias": MATCH_EVIDENCE_SKILL_ALIASES},
    )


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


def _skill_lists(
    jd_data: dict[str, Any],
    resume_data: dict[str, Any],
    *,
    jd_text: str,
    resume_text: str,
) -> tuple[list[str], list[str], list[str]]:
    match_evidence_schema = _match_evidence_schema()
    if jd_text.strip():
        jd_evidence = collect_jd_skill_evidence(jd_data, jd_text, match_evidence_schema)
        jd_skills = required_skills_from_evidence(jd_evidence)
    else:
        jd_skills = canonicalize_skill_list(
            _normalize_items(jd_data.get("必备技能")),
            match_evidence_schema,
            keep_unknown=True,
        )
    resume_evidence = "\n".join(
        _normalize_items(resume_data.get("核心技能"))
        + _normalize_items(resume_data.get("项目经历"))
        + _normalize_items(resume_data.get("实习经历"))
        + [resume_text]
    )
    resume_skills = merge_unique(
        _normalize_items(resume_data.get("核心技能"))
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


def compute_match_rule_result(
    jd_data: dict[str, Any],
    resume_data: dict[str, Any],
    *,
    jd_text: str = "",
    resume_text: str = "",
    scoring_policy: MatchScoringPolicy = DEFAULT_SCORING_POLICY,
) -> dict[str, Any]:
    jd_direction = _normalize_text(jd_data.get("岗位方向"))
    resume_direction = _normalize_text(resume_data.get("目标岗位"))

    jd_skills, matched_skills, missing_skills = _skill_lists(
        jd_data,
        resume_data,
        jd_text=jd_text,
        resume_text=resume_text,
    )
    matched_projects = _match_projects(jd_skills, resume_data)
    jd_direction_context = "\n".join(
        [jd_text]
        + _normalize_items(jd_data.get("核心职责"))
        + _normalize_items(jd_data.get("任职要求"))
        + jd_skills
    )
    resume_direction_context = "\n".join(
        [resume_text]
        + _normalize_items(resume_data.get("项目经历"))
        + _normalize_items(resume_data.get("实习经历"))
        + _normalize_items(resume_data.get("核心技能"))
    )
    direction_decision = evaluate_direction_compatibility(
        jd_direction,
        resume_direction,
        jd_context=jd_direction_context,
        resume_context=resume_direction_context,
        shared_skills=matched_skills,
    )
    direction_match = direction_decision.matches

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

    breakdown = compute_score_breakdown(
        direction_match=direction_match,
        required_skill_count=len(jd_skills),
        matched_skill_count=len(matched_skills),
        education_match=education_match,
        experience_match=experience_match,
        matched_project_count=len(matched_projects),
        policy=scoring_policy,
    )
    score = max(0, min(sum(breakdown.values()), 100))
    skill_schema = _match_evidence_schema()
    skill_evidence = (
        collect_jd_skill_evidence(jd_data, jd_text, skill_schema)
        if jd_text.strip()
        else []
    )

    return {
        "匹配分数": score,
        "匹配等级": scoring_policy.level_for(score),
        "岗位方向匹配": direction_match,
        "岗位方向关系": direction_decision.relation.value,
        "岗位方向证据": direction_decision.as_dict(),
        "学历匹配": education_match,
        "经验匹配": experience_match,
        "命中技能": matched_skills,
        "缺失技能": missing_skills,
        "命中项目": matched_projects,
        "技能证据": [item.as_dict() for item in skill_evidence],
        "匹配分项": breakdown,
        "评分策略": scoring_policy.as_dict(),
    }
