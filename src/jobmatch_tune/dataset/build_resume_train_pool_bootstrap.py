from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_skills_from_text,
    infer_job_direction,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def load_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def build_project_text(direction: str, title: str, company: str, skills: list[str]) -> list[str]:
    skill_text = "、".join(skills[:4]) if skills else "相关技术栈"
    company_text = company or "目标业务系统"
    return [
        f"围绕{title or direction}相关场景完成核心模块开发与优化，重点使用{skill_text}支持业务落地。",
        f"参与{company_text}相关项目迭代，负责功能实现、问题定位与稳定性优化。",
    ]


def build_internship_text(direction: str, title: str) -> list[str]:
    return [f"在{direction}方向团队参与{title or direction}相关实习，负责基础开发、测试或联调工作。"]


def build_strengths(direction: str, skills: list[str]) -> list[str]:
    strengths = [f"{direction}方向实践经验"]
    if skills:
        strengths.append("核心技能覆盖：" + "、".join(skills[:3]))
    strengths.append("具备较强工程落地能力")
    return strengths


def build_row(row: dict[str, Any], schema: dict[str, Any], index: int) -> dict[str, Any] | None:
    title = normalize_text(row.get("job_title"))
    company = normalize_text(row.get("company"))
    raw_text = normalize_text(row.get("raw_text"))
    direction = infer_job_direction(title, raw_text, schema)
    if not direction:
        return None

    education = extract_education_requirement(raw_text) or "本科"
    skills = extract_skills_from_text(raw_text, schema)
    if len(skills) < 2:
        return None

    label = {
        "目标岗位": direction,
        "教育背景": [f"{education}，计算机相关专业，示例高校"],
        "核心技能": skills[:8],
        "实习经历": build_internship_text(direction, title),
        "项目经历": build_project_text(direction, title, company, skills),
        "优势标签": build_strengths(direction, skills),
    }
    text = "\n".join(
        [
            f"目标岗位：{label['目标岗位']}",
            f"教育背景：{label['教育背景'][0]}",
            "核心技能：" + "、".join(label["核心技能"]),
            "实习经历：" + "；".join(label["实习经历"]),
            "项目经历：" + "；".join(label["项目经历"]),
            "优势：" + "、".join(label["优势标签"]),
        ]
    )
    return {
        "id": f"resume_bootstrap_{index}_{row.get('id', '')}",
        "task": "resume_parse",
        "source_type": "bootstrap_from_jd",
        "text": text,
        "label": label,
        "meta": {
            "language": "zh",
            "generator": "resume_bootstrap_from_jd_v1",
            "source_jd": row.get("id", ""),
        },
    }


def build_rows(rows: list[dict[str, Any]], schema: dict[str, Any], max_rows: int) -> list[dict[str, Any]]:
    built = []
    for index, row in enumerate(rows, start=1):
        item = build_row(row, schema, index)
        if item is None:
            continue
        built.append(item)
        if len(built) >= max_rows:
            break
    return built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/jd_train_pool_combined.jsonl")
    parser.add_argument("--schema", default="configs/label_schema.yaml")
    parser.add_argument("--out", default="data/eval/resume_train_pool_bootstrap.jsonl")
    parser.add_argument("--max-rows", type=int, default=2600)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    schema = load_schema(args.schema)
    built = build_rows(rows, schema, args.max_rows)
    write_jsonl(args.out, built)
    print(f"bootstrap={len(built)}")


if __name__ == "__main__":
    main()
