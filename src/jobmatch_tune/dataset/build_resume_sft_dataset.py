from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from jobmatch_tune.dataset.templates import SYSTEM_PROMPT, resume_parse_prompt
from jobmatch_tune.dataset.grouped_split import split_linked_samples
from jobmatch_tune.resume.privacy import redact_resume_pii
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def _render_lines(items: list[str], prefix: str = "- ") -> str:
    if not items:
        return ""
    return "\n".join(f"{prefix}{item}" for item in items)


def _label_to_json(label: dict[str, Any]) -> str:
    return json.dumps(label, ensure_ascii=False)


def _variant_original(row: dict[str, Any]) -> str:
    return row["text"].strip()


def _variant_profile_card(label: dict[str, Any]) -> str:
    sections = [
        "候选人简历",
        f"目标岗位：{label.get('目标岗位', '')}",
        f"教育背景：{'; '.join(label.get('教育背景', []))}",
        f"核心技能：{'、'.join(label.get('核心技能', []))}",
    ]
    internships = label.get("实习经历", [])
    projects = label.get("项目经历", [])
    strengths = label.get("优势标签", [])
    if internships:
        sections.append("实习经历：\n" + _render_lines(internships, prefix="1. "))
    if projects:
        sections.append("项目经历：\n" + _render_lines(projects, prefix="1. "))
    if strengths:
        sections.append(f"优势标签：{'、'.join(strengths)}")
    return "\n".join(part for part in sections if part.strip())


def _variant_bullets(label: dict[str, Any]) -> str:
    parts = [
        f"求职方向：{label.get('目标岗位', '')}",
        "教育：",
        _render_lines(label.get("教育背景", [])),
        "技能栈：",
        _render_lines(label.get("核心技能", [])),
        "实习：",
        _render_lines(label.get("实习经历", [])),
        "项目：",
        _render_lines(label.get("项目经历", [])),
        "优势：",
        _render_lines(label.get("优势标签", [])),
    ]
    return "\n".join(part for part in parts if part and part.strip())


def _variant_compact(label: dict[str, Any]) -> str:
    parts = [
        f"目标岗位 {label.get('目标岗位', '')}",
        f"教育 {'；'.join(label.get('教育背景', []))}",
        f"技能 {'、'.join(label.get('核心技能', []))}",
        f"实习 {'；'.join(label.get('实习经历', []))}",
        f"项目 {'；'.join(label.get('项目经历', []))}",
        f"优势 {'、'.join(label.get('优势标签', []))}",
    ]
    return "\n".join(part for part in parts if not part.endswith(" "))


def _variant_mixed(label: dict[str, Any]) -> str:
    lines = [
        f"候选人目标：{label.get('目标岗位', '')}",
        f"教育经历：{'；'.join(label.get('教育背景', []))}",
        "核心技能：" + " / ".join(label.get("核心技能", [])),
    ]
    if label.get("实习经历"):
        lines.append("实习经历：")
        lines.extend(f"- {item}" for item in label["实习经历"])
    if label.get("项目经历"):
        lines.append("项目经历：")
        lines.extend(f"- {item}" for item in label["项目经历"])
    if label.get("优势标签"):
        lines.append("个人优势：" + "、".join(label["优势标签"]))
    return "\n".join(lines)


def _variant_timeline(label: dict[str, Any]) -> str:
    parts = [
        f"求职目标：{label.get('目标岗位', '')}",
        f"教育背景：{'；'.join(label.get('教育背景', []))}",
        "技能概览：" + " / ".join(label.get("核心技能", [])),
    ]
    if label.get("实习经历"):
        parts.append("实习时间线：\n" + _render_lines(label.get("实习经历", []), prefix="• "))
    if label.get("项目经历"):
        parts.append("项目时间线：\n" + _render_lines(label.get("项目经历", []), prefix="• "))
    if label.get("优势标签"):
        parts.append("候选人优势：" + "、".join(label.get("优势标签", [])))
    return "\n".join(part for part in parts if part.strip())


def _variant_skill_first(label: dict[str, Any]) -> str:
    parts = [
        f"目标岗位：{label.get('目标岗位', '')}",
        "技能优先视图：",
        _render_lines(label.get("核心技能", [])),
        "项目/实习：",
        _render_lines([*label.get("项目经历", []), *label.get("实习经历", [])]),
        "教育背景：",
        _render_lines(label.get("教育背景", [])),
        "个人亮点：",
        _render_lines(label.get("优势标签", [])),
    ]
    return "\n".join(part for part in parts if part and part.strip())


def _variant_project_first(label: dict[str, Any]) -> str:
    parts = [
        f"应聘方向：{label.get('目标岗位', '')}",
        "重点项目：",
        _render_lines(label.get("项目经历", []), prefix="1. "),
        "相关实习：",
        _render_lines(label.get("实习经历", []), prefix="1. "),
        f"核心技能：{'、'.join(label.get('核心技能', []))}",
        f"教育背景：{'；'.join(label.get('教育背景', []))}",
        f"优势标签：{'、'.join(label.get('优势标签', []))}",
    ]
    return "\n".join(part for part in parts if part and part.strip())


def _variant_plain_sections(label: dict[str, Any]) -> str:
    blocks = [
        f"目标岗位：{label.get('目标岗位', '')}",
        "教育背景：\n" + "\n".join(label.get("教育背景", [])),
        "核心技能：\n" + "\n".join(label.get("核心技能", [])),
        "实习经历：\n" + "\n".join(label.get("实习经历", [])),
        "项目经历：\n" + "\n".join(label.get("项目经历", [])),
        "优势标签：\n" + "\n".join(label.get("优势标签", [])),
    ]
    return "\n\n".join(block for block in blocks if block.strip())


def _variant_ocr_like(label: dict[str, Any]) -> str:
    text = _variant_mixed(label)
    text = text.replace("：", ":")
    text = text.replace("，", ",")
    text = text.replace("；", ";")
    text = text.replace("。", "")
    text = text.replace("、", " ")
    text = text.replace("MySQL", "My SOL")
    text = text.replace("Pytest", "Py test")
    text = text.replace("Kubernetes", "Kubernet es")
    return text


def _variant_education_first(label: dict[str, Any]) -> str:
    parts = [
        f"教育背景：{'；'.join(label.get('教育背景', []))}",
        f"目标岗位：{label.get('目标岗位', '')}",
        f"核心技能：{'、'.join(label.get('核心技能', []))}",
        f"项目经历：{'；'.join(label.get('项目经历', []))}",
        f"实习经历：{'；'.join(label.get('实习经历', []))}",
        f"优势标签：{'、'.join(label.get('优势标签', []))}",
    ]
    return "\n".join(part for part in parts if part.strip())


def _variant_internship_first(label: dict[str, Any]) -> str:
    parts = [
        f"目标岗位：{label.get('目标岗位', '')}",
        "实习优先：",
        _render_lines(label.get("实习经历", []), prefix="- "),
        "项目补充：",
        _render_lines(label.get("项目经历", []), prefix="- "),
        f"技能：{' / '.join(label.get('核心技能', []))}",
        f"教育：{'；'.join(label.get('教育背景', []))}",
        f"优势：{'、'.join(label.get('优势标签', []))}",
    ]
    return "\n".join(part for part in parts if part and part.strip())


def _variant_markdown_sections(label: dict[str, Any]) -> str:
    blocks = [
        f"## 目标岗位\n{label.get('目标岗位', '')}",
        "## 教育背景\n" + "\n".join(f"- {item}" for item in label.get("教育背景", [])),
        "## 核心技能\n" + "\n".join(f"- {item}" for item in label.get("核心技能", [])),
        "## 实习经历\n" + "\n".join(f"- {item}" for item in label.get("实习经历", [])),
        "## 项目经历\n" + "\n".join(f"- {item}" for item in label.get("项目经历", [])),
        "## 优势标签\n" + "\n".join(f"- {item}" for item in label.get("优势标签", [])),
    ]
    return "\n\n".join(block for block in blocks if block.strip())


def _variant_semicolon(label: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"岗位={label.get('目标岗位', '')}",
            f"教育={'；'.join(label.get('教育背景', []))}",
            f"技能={'；'.join(label.get('核心技能', []))}",
            f"实习={'；'.join(label.get('实习经历', []))}",
            f"项目={'；'.join(label.get('项目经历', []))}",
            f"优势={'；'.join(label.get('优势标签', []))}",
        ]
    )


def _variant_mixed_cn_en(label: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in [
            f"Target Role: {label.get('目标岗位', '')}",
            f"Education: {'; '.join(label.get('教育背景', []))}",
            f"Skills: {', '.join(label.get('核心技能', []))}",
            "Projects:\n" + "\n".join(f"* {item}" for item in label.get("项目经历", [])),
            "Internships:\n" + "\n".join(f"* {item}" for item in label.get("实习经历", [])),
            f"Strengths: {', '.join(label.get('优势标签', []))}",
        ]
        if part.strip()
    )


def _variant_achievement_first(label: dict[str, Any]) -> str:
    parts = [
        f"求职岗位：{label.get('目标岗位', '')}",
        "个人优势：",
        _render_lines(label.get("优势标签", [])),
        "项目经历：",
        _render_lines(label.get("项目经历", [])),
        "实习经历：",
        _render_lines(label.get("实习经历", [])),
        f"核心技能：{'、'.join(label.get('核心技能', []))}",
        f"教育背景：{'；'.join(label.get('教育背景', []))}",
    ]
    return "\n".join(part for part in parts if part and part.strip())


def _variant_table_like(label: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"目标岗位 | {label.get('目标岗位', '')}",
            f"教育背景 | {'；'.join(label.get('教育背景', []))}",
            f"核心技能 | {' / '.join(label.get('核心技能', []))}",
            f"实习经历 | {'；'.join(label.get('实习经历', []))}",
            f"项目经历 | {'；'.join(label.get('项目经历', []))}",
            f"优势标签 | {'；'.join(label.get('优势标签', []))}",
        ]
    )


def _variant_short_paragraph(label: dict[str, Any]) -> str:
    return (
        f"候选人目标岗位为{label.get('目标岗位', '')}。"
        f"教育背景包括{'；'.join(label.get('教育背景', []))}。"
        f"核心技能覆盖{'、'.join(label.get('核心技能', []))}。"
        f"实习经历包括{'；'.join(label.get('实习经历', []))}。"
        f"项目经历包括{'；'.join(label.get('项目经历', []))}。"
        f"优势标签包括{'、'.join(label.get('优势标签', []))}。"
    )


def _variant_skill_matrix(label: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Role={label.get('目标岗位', '')}",
            "Skill Matrix:",
            " | ".join(label.get("核心技能", [])),
            "Projects:",
            _render_lines(label.get("项目经历", []), prefix="> "),
            "Internships:",
            _render_lines(label.get("实习经历", []), prefix="> "),
            "Education:",
            _render_lines(label.get("教育背景", []), prefix="> "),
            "Strengths:",
            _render_lines(label.get("优势标签", []), prefix="> "),
        ]
    )


def _variant_dense_resume(label: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"岗位:{label.get('目标岗位', '')}",
            f"教育:{'|'.join(label.get('教育背景', []))}",
            f"技能:{'|'.join(label.get('核心技能', []))}",
            f"实习:{'|'.join(label.get('实习经历', []))}",
            f"项目:{'|'.join(label.get('项目经历', []))}",
            f"优势:{'|'.join(label.get('优势标签', []))}",
        ]
    )


VARIANT_BUILDERS = [
    ("original", _variant_original),
    ("profile_card", lambda row: _variant_profile_card(row["label"])),
    ("bullets", lambda row: _variant_bullets(row["label"])),
    ("compact", lambda row: _variant_compact(row["label"])),
    ("mixed", lambda row: _variant_mixed(row["label"])),
    ("timeline", lambda row: _variant_timeline(row["label"])),
    ("skill_first", lambda row: _variant_skill_first(row["label"])),
    ("project_first", lambda row: _variant_project_first(row["label"])),
    ("plain_sections", lambda row: _variant_plain_sections(row["label"])),
    ("ocr_like", lambda row: _variant_ocr_like(row["label"])),
    ("education_first", lambda row: _variant_education_first(row["label"])),
    ("internship_first", lambda row: _variant_internship_first(row["label"])),
    ("markdown_sections", lambda row: _variant_markdown_sections(row["label"])),
    ("semicolon", lambda row: _variant_semicolon(row["label"])),
    ("mixed_cn_en", lambda row: _variant_mixed_cn_en(row["label"])),
    ("achievement_first", lambda row: _variant_achievement_first(row["label"])),
    ("table_like", lambda row: _variant_table_like(row["label"])),
    ("short_paragraph", lambda row: _variant_short_paragraph(row["label"])),
    ("skill_matrix", lambda row: _variant_skill_matrix(row["label"])),
    ("dense_resume", lambda row: _variant_dense_resume(row["label"])),
]


def build_resume_sample(row: dict[str, Any], variant_name: str, rendered_text: str) -> dict[str, Any]:
    source_group = str(row.get("source_group") or row["id"])
    source_group = source_group.removesuffix("_ocr")
    return {
        "id": f"{row['id']}_{variant_name}",
        "task_type": "resume_parse",
        "source_group": source_group,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": resume_parse_prompt(rendered_text)},
            {"role": "assistant", "content": _label_to_json(row["label"])},
        ],
    }


def _sample_content_hash(sample: dict[str, Any]) -> str:
    user_text = str(sample["messages"][1].get("content") or "")
    assistant_text = str(sample["messages"][-1].get("content") or "")
    return hashlib.sha1(f"{user_text}\n---\n{assistant_text}".encode("utf-8")).hexdigest()


def deduplicate_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for sample in samples:
        key = _sample_content_hash(sample)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def split_grouped_samples(
    samples: list[dict[str, Any]], train_ratio: float, valid_ratio: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    return split_linked_samples(samples, train_ratio, valid_ratio, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/resume_manual_train_pool.jsonl")
    parser.add_argument("--out-dir", default="data/sft_resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    samples: list[dict[str, Any]] = []
    for row in rows:
        for variant_name, builder in VARIANT_BUILDERS:
            rendered_text = redact_resume_pii(builder(row)).strip()
            if not rendered_text:
                continue
            samples.append(build_resume_sample(row, variant_name, rendered_text))
    samples = deduplicate_samples(samples)

    splits = split_grouped_samples(samples, args.train_ratio, args.valid_ratio, args.seed)
    for split, split_rows in splits.items():
        write_jsonl(f"{args.out_dir}/{split}.jsonl", split_rows)
        print(f"{split}: {len(split_rows)}")


if __name__ == "__main__":
    main()
