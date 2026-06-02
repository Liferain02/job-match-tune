from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_jsonl

DIRECTION_FALLBACKS = {
    "前端开发": "后端开发",
    "后端开发": "前端开发",
    "算法工程": "AI应用开发",
    "AI应用开发": "算法工程",
    "测试开发": "后端开发",
    "AI Infra": "运维开发",
    "运维开发": "网络与基础设施",
    "网络与基础设施": "运维开发",
    "硬件研发": "嵌入式开发",
    "嵌入式开发": "硬件研发",
}


def _stable_index(row_id: str, modulo: int) -> int:
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _add_unexpected_field(label: dict[str, Any]) -> dict[str, Any]:
    rejected = copy.deepcopy(label)
    rejected["岗位摘要"] = "候选输出中不应额外生成未约定字段"
    return rejected


def _change_direction(label: dict[str, Any]) -> dict[str, Any] | None:
    direction = str(label.get("岗位方向") or "")
    if not direction:
        return None
    rejected = copy.deepcopy(label)
    fallback = "后端开发" if direction != "后端开发" else "算法工程"
    rejected["岗位方向"] = DIRECTION_FALLBACKS.get(direction, fallback)
    return rejected


def _drop_responsibility(label: dict[str, Any]) -> dict[str, Any] | None:
    responsibilities = list(label.get("核心职责") or [])
    if not responsibilities:
        return None
    rejected = copy.deepcopy(label)
    rejected["核心职责"] = responsibilities[:-1]
    return rejected


def _leak_responsibility_into_skills(label: dict[str, Any]) -> dict[str, Any] | None:
    responsibilities = list(label.get("核心职责") or [])
    if not responsibilities:
        return None
    rejected = copy.deepcopy(label)
    skills = list(rejected.get("必备技能") or [])
    skills.append(responsibilities[0])
    rejected["必备技能"] = skills
    return rejected


def _mix_education_into_experience(label: dict[str, Any]) -> dict[str, Any] | None:
    education = str(label.get("学历要求") or "")
    if not education:
        return None
    rejected = copy.deepcopy(label)
    experience = str(rejected.get("经验要求") or "")
    rejected["经验要求"] = f"{experience}；{education}".strip("；")
    return rejected


def _drop_resume_strength(label: dict[str, Any]) -> dict[str, Any] | None:
    strengths = list(label.get("优势标签") or [])
    if not strengths:
        return None
    rejected = copy.deepcopy(label)
    rejected["优势标签"] = strengths[:-1]
    return rejected


def _drop_resume_project(label: dict[str, Any]) -> dict[str, Any] | None:
    projects = list(label.get("项目经历") or [])
    if not projects:
        return None
    rejected = copy.deepcopy(label)
    rejected["项目经历"] = projects[:-1]
    return rejected


def _mix_resume_education_into_skills(label: dict[str, Any]) -> dict[str, Any] | None:
    education = list(label.get("教育背景") or [])
    if not education:
        return None
    rejected = copy.deepcopy(label)
    skills = list(rejected.get("核心技能") or [])
    skills.append(education[0])
    rejected["核心技能"] = skills
    return rejected


def _drop_match_gap(label: dict[str, Any]) -> dict[str, Any] | None:
    gaps = list(label.get("主要短板") or [])
    if not gaps:
        return None
    rejected = copy.deepcopy(label)
    rejected["主要短板"] = gaps[:-1]
    return rejected


def _swap_match_strength_gap(label: dict[str, Any]) -> dict[str, Any] | None:
    strengths = list(label.get("匹配优势") or [])
    gaps = list(label.get("主要短板") or [])
    if not strengths or not gaps:
        return None
    rejected = copy.deepcopy(label)
    rejected["匹配优势"] = gaps
    rejected["主要短板"] = strengths
    return rejected


def _drop_match_suggestion(label: dict[str, Any]) -> dict[str, Any] | None:
    suggestions = list(label.get("简历优化建议") or [])
    if not suggestions:
        return None
    rejected = copy.deepcopy(label)
    rejected["简历优化建议"] = suggestions[:-1]
    return rejected


JD_CORRUPTIONS: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any] | None]]] = [
    ("unexpected_field", _add_unexpected_field),
    ("direction_mismatch", _change_direction),
    ("responsibility_drop", _drop_responsibility),
    ("responsibility_skill_leak", _leak_responsibility_into_skills),
    ("education_experience_mix", _mix_education_into_experience),
]

RESUME_CORRUPTIONS: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any] | None]]] = [
    ("unexpected_field", _add_unexpected_field),
    ("resume_strength_drop", _drop_resume_strength),
    ("resume_project_drop", _drop_resume_project),
    ("resume_education_skill_leak", _mix_resume_education_into_skills),
]

MATCH_CORRUPTIONS: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any] | None]]] = [
    ("unexpected_field", _add_unexpected_field),
    ("match_gap_drop", _drop_match_gap),
    ("match_strength_gap_swap", _swap_match_strength_gap),
    ("match_suggestion_drop", _drop_match_suggestion),
]


CorruptionFn = Callable[[dict[str, Any]], dict[str, Any] | None]


def _corruptions_for_task(
    task_type: str,
    chosen_obj: dict[str, Any],
) -> list[tuple[str, CorruptionFn]]:
    if task_type == "resume_parse" or "目标岗位" in chosen_obj:
        return RESUME_CORRUPTIONS
    if task_type == "match" or "匹配结论" in chosen_obj:
        return MATCH_CORRUPTIONS
    return JD_CORRUPTIONS


def build_bootstrap_preference(row: dict[str, Any]) -> dict[str, Any]:
    row_id = str(row["id"])
    messages = row["messages"]
    chosen_obj = json.loads(messages[-1]["content"])
    task_type = row.get("task_type", "jd_parse")
    corruptions = _corruptions_for_task(task_type, chosen_obj)
    start = _stable_index(row_id, len(corruptions))
    for offset in range(len(corruptions)):
        strategy, corruption = corruptions[(start + offset) % len(corruptions)]
        rejected_obj = corruption(chosen_obj)
        if rejected_obj is not None and rejected_obj != chosen_obj:
            chosen = json.dumps(chosen_obj, ensure_ascii=False, sort_keys=True)
            rejected = json.dumps(rejected_obj, ensure_ascii=False, sort_keys=True)
            return {
                "id": f"{row_id}_{strategy}",
                "source_id": row_id,
                "task_type": task_type,
                "prompt": copy.deepcopy(messages[:-1]),
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "meta": {
                    "provenance": "synthetic_structured_hard_negative",
                    "rejection_strategy": strategy,
                },
            }
    raise ValueError(f"Unable to build a rejected answer for {row_id}")


def build_split(input_path: str, output_path: str) -> int:
    rows = [build_bootstrap_preference(row) for row in read_jsonl(input_path)]
    write_jsonl(output_path, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", default="data/sft_jd_quality/train.jsonl")
    parser.add_argument("--valid-input", default="data/sft_jd_quality/valid.jsonl")
    parser.add_argument("--train-out", default="data/preference/train.jsonl")
    parser.add_argument("--valid-out", default="data/preference/valid.jsonl")
    args = parser.parse_args()

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    train_count = build_split(args.train_input, args.train_out)
    valid_count = build_split(args.valid_input, args.valid_out)
    print(f"train: {train_count}")
    print(f"valid: {valid_count}")


if __name__ == "__main__":
    main()
