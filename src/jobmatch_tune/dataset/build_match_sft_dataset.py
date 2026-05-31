from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from typing import Any

from jobmatch_tune.dataset.templates import SYSTEM_PROMPT, match_prompt
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def build_analysis_from_label(label: dict[str, Any]) -> dict[str, Any]:
    level = str(label.get("匹配等级") or "")
    matched = list(label.get("命中技能") or [])
    missing = list(label.get("缺失技能") or [])
    direction_match = bool(label.get("岗位方向匹配"))
    education_match = bool(label.get("学历匹配"))
    experience_match = bool(label.get("经验匹配"))

    if level == "高匹配":
        conclusion = "候选人与岗位整体高度匹配，关键技能和基础要求基本覆盖。"
    elif level == "较匹配":
        conclusion = "候选人与岗位整体较匹配，具备主要能力，但仍存在少量补齐空间。"
    else:
        conclusion = "候选人与岗位匹配度有限，核心能力或基础条件存在明显缺口。"

    strengths = []
    if direction_match:
        strengths.append("求职方向与岗位方向一致")
    if education_match:
        strengths.append("学历背景满足岗位要求")
    if experience_match:
        strengths.append("经验背景满足岗位要求")
    if matched:
        strengths.append("已覆盖关键技能：" + "、".join(matched[:6]))
    if not strengths:
        strengths.append("已有简历信息可作为初步评估基础")

    gaps = []
    if not direction_match:
        gaps.append("目标岗位方向与 JD 方向不一致")
    if not education_match:
        gaps.append("学历条件与 JD 要求存在差距")
    if not experience_match:
        gaps.append("经验条件与 JD 要求存在差距")
    if missing:
        gaps.append("缺失关键技能：" + "、".join(missing[:6]))
    if not gaps:
        gaps.append("暂无明显硬性短板，后续可继续补充更量化的项目成果")

    suggestions = []
    if missing:
        suggestions.append("在简历中补充或强化以下技能证据：" + "、".join(missing[:6]))
    if not experience_match:
        suggestions.append("增加与岗位相关的项目、实习或工作经历描述")
    if not direction_match:
        suggestions.append("明确简历目标岗位，并突出与目标方向一致的项目与技能")
    if not suggestions:
        suggestions.append("保持当前简历结构，继续强化核心项目成果与量化结果")

    recommended_roles = []
    if direction_match:
        recommended_roles.append("当前 JD 对应方向")
    if level == "高匹配":
        recommended_roles.append("同方向相近岗位")
    elif matched:
        recommended_roles.append("技能相近岗位")
    if not recommended_roles:
        recommended_roles.append("技能相近或低门槛过渡岗位")

    return {
        "匹配结论": conclusion,
        "匹配优势": strengths,
        "主要短板": gaps,
        "简历优化建议": suggestions,
        "推荐投递岗位方向": recommended_roles,
    }


def build_match_sample(row: dict[str, Any]) -> dict[str, Any]:
    rule_result = {
        "匹配等级": row["label"].get("匹配等级", ""),
        "岗位方向匹配": row["label"].get("岗位方向匹配", False),
        "学历匹配": row["label"].get("学历匹配", False),
        "经验匹配": row["label"].get("经验匹配", False),
        "命中技能": row["label"].get("命中技能", []),
        "缺失技能": row["label"].get("缺失技能", []),
    }
    assistant = build_analysis_from_label(row["label"])
    return {
        "id": row["id"],
        "task_type": "match",
        "source_group": row["id"].rsplit("_", 1)[0],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": match_prompt(
                    row["jd_text"],
                    row["resume_text"],
                    json.dumps(rule_result, ensure_ascii=False),
                ),
            },
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
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
    parser.add_argument("--input", default="data/eval/match_train_pool_combined.jsonl")
    parser.add_argument("--out-dir", default="data/sft_match")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    samples = [build_match_sample(row) for row in rows]
    splits = split_grouped_samples(samples, args.train_ratio, args.valid_ratio, args.seed)
    for split, split_rows in splits.items():
        write_jsonl(f"{args.out_dir}/{split}.jsonl", split_rows)
        print(f"{split}: {len(split_rows)}")


if __name__ == "__main__":
    main()
