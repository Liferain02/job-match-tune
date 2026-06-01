from __future__ import annotations

import json
import re
from typing import Any


SUSPICIOUS_TITLE_KEYWORDS = [
    "教师",
    "老师",
    "讲师",
    "助教",
    "培训师",
    "编导",
    "编剧",
    "摄制",
    "新闻编辑",
    "校园招聘",
    "管培生",
    "实习",
]

WEAK_SOURCE_PREFIXES = ("hf_", "github_")

RISK_WEIGHTS = {
    "invalid_assistant_json": 5,
    "suspicious_title_keyword": 4,
    "responsibility_contains_requirement_marker": 3,
    "requirement_contains_responsibility_marker": 3,
    "quality_weak_missing_core_field": 3,
    "empty_responsibilities": 2,
    "oversized_single_responsibility": 2,
    "empty_skills": 1,
    "empty_education": 1,
    "empty_experience": 1,
    "quality_weak_tier": 0,
    "weak_public_source": 0,
}

HIGH_RISK_THRESHOLD = 4

REQUIREMENT_MARKERS = (
    "任职要求",
    "岗位要求",
    "职位要求",
    "任职资格",
    "能力要求",
    "技能要求",
    "专业要求",
)

RESPONSIBILITY_MARKERS = (
    "岗位职责",
    "工作职责",
    "职位描述",
    "工作内容",
    "职责描述",
)


def assistant_json(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return json.loads(row["messages"][-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if len(messages) < 2:
        return ""
    return str(messages[1].get("content") or "")


def extract_title(prompt: str) -> str:
    match = re.search(r"岗位名称[：:]\s*([^\n]+)", prompt)
    return match.group(1).strip() if match else ""


def infer_source(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    source = str(meta.get("source") or "")
    if source:
        return source
    row_id = str(row.get("id") or "")
    if row_id.startswith("hf_job_educational_train"):
        return "hf_job_educational_train_2026_05_17"
    if row_id.startswith("hf_job_educational_validation"):
        return "hf_job_educational_validation_2026_05_17"
    if row_id.startswith("github_workaggregation_test"):
        return "github_workaggregation_test"
    return row_id.replace("_jd_parse", "").split("_")[0]


def risk_reasons(row: dict[str, Any]) -> list[str]:
    assistant = assistant_json(row)
    if assistant is None:
        return ["invalid_assistant_json"]

    reasons = []
    meta = row.get("meta") or {}
    tier = str(meta.get("quality_tier") or "")
    source = infer_source(row)
    title = extract_title(prompt_text(row))

    if tier == "quality_weak":
        reasons.append("quality_weak_tier")
    if source.startswith(WEAK_SOURCE_PREFIXES):
        reasons.append("weak_public_source")
    if any(keyword in title for keyword in SUSPICIOUS_TITLE_KEYWORDS):
        reasons.append("suspicious_title_keyword")
    if not assistant.get("核心职责"):
        reasons.append("empty_responsibilities")
    if not assistant.get("必备技能"):
        reasons.append("empty_skills")
    if not assistant.get("学历要求"):
        reasons.append("empty_education")
    if not assistant.get("经验要求"):
        reasons.append("empty_experience")
    responsibilities = [str(item) for item in (assistant.get("核心职责") or [])]
    requirements_text = str(assistant.get("任职要求") or "")
    if any(marker in "\n".join(responsibilities) for marker in REQUIREMENT_MARKERS):
        reasons.append("responsibility_contains_requirement_marker")
    if requirements_text and any(marker in requirements_text for marker in RESPONSIBILITY_MARKERS):
        reasons.append("requirement_contains_responsibility_marker")
    if len(responsibilities) == 1:
        first = responsibilities[0]
        if len(first) > 500:
            reasons.append("oversized_single_responsibility")
    if tier == "quality_weak" and (not assistant.get("学历要求") or not assistant.get("必备技能")):
        reasons.append("quality_weak_missing_core_field")
    return reasons


def risk_score(reasons: list[str]) -> int:
    return sum(RISK_WEIGHTS.get(reason, 1) for reason in reasons)


def is_high_risk(row: dict[str, Any], *, threshold: int = HIGH_RISK_THRESHOLD) -> bool:
    return risk_score(risk_reasons(row)) >= threshold
