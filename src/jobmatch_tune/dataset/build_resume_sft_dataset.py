from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from typing import Any

from jobmatch_tune.dataset.templates import SYSTEM_PROMPT, resume_parse_prompt
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


VARIANT_BUILDERS = [
    ("original", _variant_original),
    ("profile_card", lambda row: _variant_profile_card(row["label"])),
    ("bullets", lambda row: _variant_bullets(row["label"])),
    ("compact", lambda row: _variant_compact(row["label"])),
    ("mixed", lambda row: _variant_mixed(row["label"])),
]


def build_resume_sample(row: dict[str, Any], variant_name: str, rendered_text: str) -> dict[str, Any]:
    return {
        "id": f"{row['id']}_{variant_name}",
        "task_type": "resume_parse",
        "source_group": row["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": resume_parse_prompt(rendered_text)},
            {"role": "assistant", "content": _label_to_json(row["label"])},
        ],
    }


def split_grouped_samples(
    samples: list[dict[str, Any]], train_ratio: float, valid_ratio: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample["source_group"]].append(sample)

    group_keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    n = len(group_keys)
    if n < 3:
        all_rows = [row for key in group_keys for row in groups[key]]
        return {"train": all_rows, "valid": all_rows[:1], "test": all_rows[:1]}

    valid_count = max(1, int(n * valid_ratio))
    test_count = max(1, n - int(n * train_ratio) - valid_count)
    train_count = max(1, n - valid_count - test_count)
    train_keys = group_keys[:train_count]
    valid_keys = group_keys[train_count : train_count + valid_count]
    test_keys = group_keys[train_count + valid_count :]
    return {
        "train": [row for key in train_keys for row in groups[key]],
        "valid": [row for key in valid_keys for row in groups[key]],
        "test": [row for key in test_keys for row in groups[key]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/resume_manual_eval_seed.jsonl")
    parser.add_argument("--out-dir", default="data/sft_resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    samples: list[dict[str, Any]] = []
    for row in rows:
        for variant_name, builder in VARIANT_BUILDERS:
            rendered_text = builder(row).strip()
            if not rendered_text:
                continue
            samples.append(build_resume_sample(row, variant_name, rendered_text))

    splits = split_grouped_samples(samples, args.train_ratio, args.valid_ratio, args.seed)
    for split, split_rows in splits.items():
        write_jsonl(f"{args.out_dir}/{split}.jsonl", split_rows)
        print(f"{split}: {len(split_rows)}")


if __name__ == "__main__":
    main()
