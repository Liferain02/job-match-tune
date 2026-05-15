from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from functools import lru_cache

import yaml

from jobmatch_tune.preprocess.jd_field_rules import (
    canonicalize_job_direction,
    extract_education_requirement,
    extract_experience_requirement,
    extract_skills_from_text,
    merge_unique,
    split_bullets,
)


JOB_DIRECTION_ALIASES = {
    "ai开发": "AI应用开发",
    "ai应用": "AI应用开发",
    "ai应用开发": "AI应用开发",
    "大模型开发": "大模型应用开发",
    "大模型应用": "大模型应用开发",
    "大模型应用开发": "大模型应用开发",
}


def remove_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json_text(text: str) -> str:
    text = remove_thinking(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text.strip()
    return text[start : end + 1]


def repair_json_text(text: str) -> str:
    text = extract_json_text(text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def deduplicate_list(items: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for item in items:
        normalized = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    return result


@lru_cache(maxsize=1)
def load_label_schema() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "label_schema.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _ensure_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [_normalize_string(item) for item in value]
    elif isinstance(value, str):
        items = [_normalize_string(part) for part in value.splitlines()]
    else:
        return []
    return [item for item in items if item]


def _canonicalize_skills(skills: list[str], schema: dict[str, Any]) -> list[str]:
    canonical_map = {}
    for canonical, aliases in schema.get("skill_alias", {}).items():
        canonical_map[canonical.lower()] = canonical
        for alias in aliases:
            canonical_map[str(alias).strip().lower()] = canonical
    result = []
    for skill in skills:
        normalized = _normalize_string(skill).lower()
        canonical = canonical_map.get(normalized)
        if canonical:
            result.append(canonical)
    return merge_unique(result)


def _contains_skill_alias(text: str, canonical: str, schema: dict[str, Any]) -> bool:
    haystack = _normalize_string(text)
    if not haystack:
        return False
    aliases = [canonical, *schema.get("skill_alias", {}).get(canonical, [])]
    for alias in aliases:
        alias_text = _normalize_string(alias)
        if not alias_text:
            continue
        if re.search(r"[A-Za-z0-9+#._-]", alias_text):
            pattern = rf"(?<![A-Za-z0-9]){re.escape(alias_text)}(?![A-Za-z0-9])"
            if re.search(pattern, haystack, flags=re.I):
                return True
        elif alias_text in haystack:
            return True
    return False


def _filter_skills_by_evidence(skills: list[str], evidence_text: str, schema: dict[str, Any]) -> list[str]:
    return [skill for skill in skills if _contains_skill_alias(evidence_text, skill, schema)]


def _extract_responsibility_lines(context_text: str) -> list[str]:
    if "岗位职责" not in context_text:
        return []
    section = context_text.split("岗位职责", 1)[1]
    section = re.sub(r"^[：:\s]*", "", section)
    section = re.split(r"\n(?:任职要求|岗位要求|职位要求|任职资格|经验要求|工作经验|学历要求|学历)[：:]", section, maxsplit=1)[0]
    return [line for line in split_bullets(section) if re.match(r"^\d+\.", line) or re.match(r"^[一二三四五六七八九十]+[、.]", line)]


def _merge_missing_responsibilities(parsed_lines: list[str], context_text: str) -> list[str]:
    context_lines = _extract_responsibility_lines(context_text)
    if not context_lines:
        return merge_unique(parsed_lines)
    parsed_set = {line.strip() for line in parsed_lines}
    missing = [line for line in context_lines if line.strip() not in parsed_set]
    if not parsed_lines or not missing:
        return merge_unique(parsed_lines)
    matched_prefix = 0
    for parsed, source in zip(parsed_lines, context_lines):
        if parsed.strip() == source.strip():
            matched_prefix += 1
        else:
            break
    if matched_prefix >= min(len(parsed_lines), 2):
        return merge_unique(parsed_lines + missing)
    return merge_unique(parsed_lines)


def _split_misplaced_fields(data: dict[str, Any], context_text: str = "") -> dict[str, Any]:
    responsibilities = _ensure_string_list(data.get("核心职责"))
    requirements = _ensure_string_list(data.get("任职要求") or data.get("岗位要求"))
    skills = _ensure_string_list(data.get("必备技能") or data.get("核心技能"))
    bonus = _ensure_string_list(data.get("加分项"))

    moved_requirements = []
    cleaned_responsibilities = []
    for line in responsibilities:
        if re.search(r"^(任职要求|岗位要求|职位要求|任职资格)", line):
            moved_requirements.append(re.sub(r"^(任职要求|岗位要求|职位要求|任职资格)[：: ]*", "", line).strip())
            continue
        if re.search(r"^(加分项|优先|优先考虑)", line):
            bonus.append(re.sub(r"^(加分项|优先考虑|优先)[：: ]*", "", line).strip())
            continue
        if re.search(r"^(经验要求|工作经验)", line):
            data["经验要求"] = data.get("经验要求") or extract_experience_requirement(line)
            continue
        if re.search(r"^(学历要求|学历)", line):
            data["学历要求"] = data.get("学历要求") or extract_education_requirement(line)
            continue
        cleaned_responsibilities.append(line)

    all_requirement_lines = requirements + moved_requirements
    for line in all_requirement_lines:
        if not data.get("经验要求"):
            data["经验要求"] = extract_experience_requirement(line) or data.get("经验要求", "")
        if not data.get("学历要求"):
            data["学历要求"] = extract_education_requirement(line) or data.get("学历要求", "")

    cleaned_responsibilities = _merge_missing_responsibilities(cleaned_responsibilities, context_text)
    combined_text = "\n".join(cleaned_responsibilities + all_requirement_lines + bonus + skills + [context_text])
    evidence_text = "\n".join(cleaned_responsibilities + all_requirement_lines + bonus + [context_text])
    schema = load_label_schema()
    inferred_skills = extract_skills_from_text(combined_text, schema)

    data["核心职责"] = merge_unique(cleaned_responsibilities)
    data["任职要求"] = merge_unique(all_requirement_lines)
    canonical_skills = _canonicalize_skills(skills + inferred_skills, schema)
    data["必备技能"] = merge_unique(_filter_skills_by_evidence(canonical_skills, evidence_text, schema))
    data["加分项"] = merge_unique(bonus)
    data["经验要求"] = _normalize_string(data.get("经验要求") or extract_experience_requirement(combined_text))
    data["学历要求"] = _normalize_string(data.get("学历要求") or extract_education_requirement(combined_text))
    if "核心技能" in data and not data.get("核心技能"):
        data.pop("核心技能")
    if "岗位要求" in data and not data["岗位要求"]:
        data.pop("岗位要求")
    return data


def normalize_parsed_data(data: Any, context_text: str = "") -> Any:
    if isinstance(data, list):
        return deduplicate_list([normalize_parsed_data(item, context_text=context_text) for item in data])
    if not isinstance(data, dict):
        return data

    normalized = {key: normalize_parsed_data(value, context_text=context_text) for key, value in data.items()}
    normalized = _split_misplaced_fields(normalized, context_text=context_text)
    direction = normalized.get("岗位方向")
    if isinstance(direction, str):
        direction_key = re.sub(r"\s+", "", direction).lower()
        direction = JOB_DIRECTION_ALIASES.get(direction_key, direction)
        context = "\n".join(
            _ensure_string_list(normalized.get("核心职责"))
            + _ensure_string_list(normalized.get("任职要求"))
            + _ensure_string_list(normalized.get("必备技能"))
        )
        full_context = "\n".join(part for part in [context_text, context] if part)
        normalized["岗位方向"] = canonicalize_job_direction(direction, full_context, load_label_schema())
    return normalized


def parse_json_output(text: str, context_text: str = "") -> dict[str, Any]:
    repaired = repair_json_text(text)
    try:
        return {
            "ok": True,
            "data": normalize_parsed_data(json.loads(repaired), context_text=context_text),
            "raw_output": text,
        }
    except json.JSONDecodeError as exc:
        return {"ok": False, "data": None, "raw_output": text, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps(parse_json_output(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
