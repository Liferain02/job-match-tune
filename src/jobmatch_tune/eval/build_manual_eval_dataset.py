from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import compose_jd_input_text
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


SELECTED_DIRECTIONS = {
    "tencent_1926867300854636544": "前端开发",
    "tencent_1958073004445556736": "前端开发",
    "tencent_2043950303379877888": "前端开发",
    "tencent_1891443603608215552": "前端开发",
    "tencent_2013906220049653760": "前端开发",
    "tencent_2028682235867201536": "前端开发",
    "tencent_1986760187561795584": "前端开发",
    "tencent_2028045000553689088": "前端开发",
    "tencent_2036346197547057152": "后端开发",
    "tencent_2020703486018220032": "后端开发",
    "tencent_1955829491829985280": "后端开发",
    "tencent_2037467994921271296": "后端开发",
    "tencent_2026648914832293888": "后端开发",
    "tencent_1986055471202721792": "后端开发",
    "tencent_1946505228073435136": "后端开发",
    "tencent_1936827253514149888": "后端开发",
    "tencent_1966418821166292992": "后端开发",
    "tencent_2037392056137183232": "后端开发",
    "tencent_2005480420615016448": "后端开发",
    "tencent_2029388403971358720": "后端开发",
    "tencent_2031246319435284480": "测试开发",
    "tencent_2027292979852640256": "测试开发",
    "tencent_1950026512501678080": "测试开发",
    "tencent_2020809610717458432": "测试开发",
    "tencent_2020809543059144704": "测试开发",
    "tencent_1904008033529323520": "测试开发",
    "tencent_2044962246391660544": "测试开发",
    "tencent_2005808331578560512": "测试开发",
    "tencent_1976128387357495296": "算法工程",
    "tencent_1670301957484453888": "算法工程",
    "tencent_2029338861385187328": "算法工程",
    "tencent_2032757901336145920": "算法工程",
    "tencent_2011070685463601152": "算法工程",
    "tencent_1942609674138394624": "算法工程",
    "tencent_1985973388233039872": "算法工程",
    "tencent_1902190189934100480": "算法工程",
    "tencent_2013865961572159488": "算法工程",
    "tencent_1998689727519281152": "算法工程",
    "tencent_1967484294205169664": "算法工程",
    "tencent_2001186278137286656": "算法工程",
    "tencent_2036630269611638784": "AI应用开发",
    "tencent_1967441978455117824": "AI应用开发",
    "tencent_1956586714755649536": "AI应用开发",
    "tencent_2031289286338445312": "AI应用开发",
    "tencent_1985973381228552192": "AI应用开发",
    "tencent_2039174621139464192": "AI应用开发",
    "tencent_2035224441180553216": "AI应用开发",
    "tencent_2034114867123875840": "AI应用开发",
    "tencent_2039533180888973312": "AI应用开发",
    "tencent_2037177915745136640": "AI应用开发",
}


def split_lines(text: str) -> list[str]:
    lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip(" -•\t")]
    return [
        line
        for line in lines
        if not line.startswith("经验要求")
        and not line.startswith("学历要求")
        and not line.startswith("任职要求")
        and not line.startswith("岗位要求")
    ]


def build_label(row: dict[str, Any]) -> dict[str, Any]:
    sections = row.get("sections", {})
    labels = row.get("labels", {})
    text = row.get("clean_text", "")
    return {
        "岗位方向": SELECTED_DIRECTIONS[row["id"]],
        "核心职责": split_lines(sections.get("responsibilities", ""))[:8],
        "必备技能": labels.get("必备技能", []),
        "加分项": split_lines(sections.get("bonus", ""))[:8],
        "经验要求": labels.get("经验要求") or extract_experience_requirement(text),
        "学历要求": labels.get("学历要求") or extract_education_requirement(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/jd_clean.jsonl")
    parser.add_argument("--out", default="data/eval/jd_manual_eval_50.jsonl")
    args = parser.parse_args()

    rows = {row["id"]: row for row in read_jsonl(args.input)}
    output = []
    for row_id in SELECTED_DIRECTIONS:
        row = rows[row_id]
        output.append(
            {
                "id": f"{row_id}_jd_parse",
                "source_id": row_id,
                "job_title": row.get("job_title", ""),
                "task": "jd_parse",
                "text": compose_jd_input_text(row),
                "label": build_label(row),
            }
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, output)
    print(f"wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
