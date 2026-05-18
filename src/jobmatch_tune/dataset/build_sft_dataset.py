from __future__ import annotations

import argparse
import json
import random
from typing import Any

from jobmatch_tune.dataset.templates import SYSTEM_PROMPT, jd_parse_prompt
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
    extract_skills_from_text,
)
from jobmatch_tune.preprocess.normalize_jd import split_sections
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


WEAK_TECH_SOURCES = {
    "hf_job_educational_train_2026_05_17",
    "hf_job_educational_validation_2026_05_17",
    "hf_job_educational_test_2026_05_17",
    "github_jhcoco_bosszp",
    "github_workaggregation_test",
}

HIGH_TRUST_SOURCES = {
    "zhaopin.jd.com",
    "careers.tencent.com",
    "talent.baidu.com",
    "moka_voyah",
    "moka_se",
    "moka_eastmoney",
    "moka_supcon",
    "moka_baai",
    "moka_hq",
    "moka_threatbook",
    "moka_sina",
    "moka_bosssoft",
    "moka_ztgame",
    "moka_transwarp",
    "moka_shopee",
    "moka_rastar",
    "moka_qianli1",
    "moka_cyou_inc",
    "moka_reo",
    "moka_jspdg",
    "moka_step",
    "moka_high_flyer",
    "moka_whfhtx",
    "moka_thfund",
    "moka_huahong",
    "moka_xmyanquhr",
}

TECH_TITLE_KEYWORDS = [
    "工程师",
    "开发",
    "算法",
    "测试",
    "研发",
    "数据",
    "前端",
    "后端",
    "运维",
    "架构",
    "java",
    "python",
    "c++",
    "go",
    "嵌入式",
    "客户端",
    "android",
    "ios",
]

STRONG_TITLE_INCLUDE_KEYWORDS = [
    "工程师",
    "软件",
    "software",
    "研发",
    "算法",
    "开发",
    "测试",
    "数据",
    "前端",
    "后端",
    "客户端",
    "服务端",
    "java",
    "python",
    "c++",
    "golang",
    "go",
    "运维",
    "sre",
    "infra",
    "平台",
    "数据库",
    "ai",
    "agent",
    "大模型",
    "机器学习",
    "产品经理",
    "架构师",
    "安全",
    "ios",
    "android",
    "嵌入式",
    "固件",
    "dba",
]

STRONG_TITLE_EXCLUDE_KEYWORDS = [
    "销售",
    "实施",
    "理财顾问",
    "投资顾问",
    "课程顾问",
    "售前",
    "售后",
    "物流",
    "采购",
    "运营",
    "市场",
    "设计师",
    "美术",
    "策划",
    "财务",
    "法务",
    "人力",
    "hr",
    "行政",
    "助教",
    "机械",
    "结构工程师",
    "解决方案架构师",
    "售前解决方案",
    "电气",
    "工艺",
    "材料",
    "飞机",
    "土建",
    "消防",
    "公关",
    "客服",
    "采销",
    "渠道经理",
    "商品经理",
    "招商",
    "店长",
    "陈列",
    "merchandising",
    "account manager",
    "客户经理",
    "品牌经理",
    "投放",
]

STRICT_DIRECTION_TITLE_HINTS = {
    "产品经理": ["产品", "产品负责人", "策略产品", "商业策略产品"],
    "算法工程": [
        "专家",
        "研究员",
        "科学家",
        "推理",
        "训练",
        "分布式通信库",
        "多模态",
        "推荐策略",
        "深度学习",
        "机器学习",
        "nlp",
    ],
    "后端开发": ["olap", "控制面", "中间件", "引擎", "数据库", "dba", "分布式存储"],
    "网络与基础设施": ["网络规划", "网络开发", "网络工程师", "网络交付", "基础架构", "基础设施"],
    "AI Infra": ["ai infra", "机器学习平台", "训练平台", "推理平台", "训推平台", "算力平台", "rl infra"],
    "高性能计算": ["高性能计算", "hpc", "gpu集群", "gpu 集群", "计算集群"],
    "汽车软件/智驾研发": ["智驾系统", "驾驶辅助", "智能行车系统", "泊车功能开发", "底盘集成控制系统开发", "底盘电控功能开发工程师", "感知质量开发工程师", "智驾软件集成"],
    "运维开发": ["gpu资源", "指挥中心", "可靠性", "devops", "sre", "infra"],
    "客户端开发": ["app", "ios", "android", "unity", "ue", "u3d"],
    "嵌入式开发": ["固件", "驱动", "bsp", "rtos", "单片机", "嵌入式"],
    "硬件研发": ["硬件开发", "硬件工程师", "功率硬件", "电力电子", "结构设计工程师", "系统集成工程师", "音响开发", "npi工程师", "电子电器"],
    "安全工程": ["渗透", "攻防", "漏洞", "安全", "威胁"],
}

STRICT_TITLE_EXCLUSION_EXCEPTIONS = {
    "安全工程": ["安全运营", "大模型安全运营"],
    "后端开发": ["研发岗"],
    "运维开发": ["sre", "devops"],
    "网络与基础设施": ["网络规划", "网络开发", "网络工程师", "网络交付"],
    "AI Infra": ["ai infra", "机器学习平台", "训练平台", "推理平台", "训推平台", "算力平台", "rl infra"],
    "高性能计算": ["高性能计算", "hpc"],
    "汽车软件/智驾研发": ["智驾系统", "驾驶辅助", "智能行车系统", "泊车功能开发", "底盘集成控制系统开发", "底盘电控功能开发工程师", "感知质量开发工程师", "智驾软件集成"],
}

MINIMAL_SKILL_SCHEMA = {
    "skill_alias": {
        "Python": ["python"],
        "Java": ["java"],
        "C++": ["c++"],
        "SQL": ["sql"],
        "Linux": ["linux"],
    }
}

WEAK_TITLE_INCLUDE_KEYWORDS = [
    "java",
    "python",
    "后端",
    "后台",
    "前端",
    "测试",
    "qa",
    "算法",
    "数据",
    "开发",
    "软件",
    "客户端",
    "android",
    "ios",
    "嵌入式",
    "运维",
    "sre",
    "infra",
    "平台",
]

WEAK_TITLE_EXCLUDE_KEYWORDS = [
    "销售",
    "实施",
    "技术支持",
    "售前",
    "售后",
    "硬件",
    "机械",
    "电气",
    "工艺",
    "材料",
    "质量",
    "生产",
    "结构",
    "管道",
    "飞机",
    "电力",
    "化工",
    "暖通",
    "土木",
    "采购",
    "运营",
]


def build_jd_parse_sample(row: dict[str, Any]) -> dict[str, Any]:
    labels = row.get("labels", {})
    sections = row.get("sections", {})
    source_text = row.get("clean_text", "")
    assistant = {
        "岗位方向": labels.get("岗位方向", ""),
        "核心职责": _split_lines(sections.get("responsibilities", ""))[:6],
        "必备技能": labels.get("必备技能", []),
        "加分项": _split_lines(sections.get("bonus", ""))[:6],
        "经验要求": labels.get("经验要求") or extract_experience_requirement(source_text),
        "学历要求": labels.get("学历要求") or extract_education_requirement(source_text),
    }
    return {
        "id": f"{row['id']}_jd_parse",
        "task_type": "jd_parse",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": jd_parse_prompt(compose_jd_input_text(row))},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
    }


def _split_lines(text: str) -> list[str]:
    lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip(" -•\t")]
    return [
        line
        for line in lines
        if not line.startswith("经验要求")
        and not line.startswith("学历要求")
        and not line.startswith("任职要求")
        and not line.startswith("岗位要求")
    ]


def compose_jd_input_text(row: dict[str, Any]) -> str:
    clean_text = row.get("clean_text", "").strip()
    title = str(row.get("job_title") or "").strip()
    company = str(row.get("company") or "").strip()
    location = str(row.get("location") or "").strip()
    header = []
    if title and "岗位名称" not in clean_text[:80]:
        header.append(f"岗位名称：{title}")
    if company and "公司" not in clean_text[:120]:
        header.append(f"公司：{company}")
    if location and "工作地点" not in clean_text[:120]:
        header.append(f"工作地点：{location}")
    return "\n".join(header + ([clean_text] if clean_text else []))


def split_samples(
    samples: list[dict[str, Any]], train_ratio: float, valid_ratio: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n < 3:
        return {"train": shuffled, "valid": shuffled[:1], "test": shuffled[:1]}
    valid_count = max(1, int(n * valid_ratio))
    test_count = max(1, n - int(n * train_ratio) - valid_count)
    train_count = max(1, n - valid_count - test_count)
    train_end = train_count
    valid_end = train_end + valid_count
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def title_has_excluded_signal(title: str) -> bool:
    return any(keyword in title for keyword in STRONG_TITLE_EXCLUDE_KEYWORDS)


def title_has_strong_tech_signal(title: str, direction: str) -> bool:
    if any(keyword in title for keyword in STRONG_TITLE_INCLUDE_KEYWORDS):
        return True
    hints = STRICT_DIRECTION_TITLE_HINTS.get(direction, [])
    return any(keyword in title for keyword in hints)


def title_has_exclusion_exception(title: str, direction: str) -> bool:
    exceptions = STRICT_TITLE_EXCLUSION_EXCEPTIONS.get(direction, [])
    return any(keyword in title for keyword in exceptions)


def is_high_trust_strong_row(row: dict[str, Any]) -> bool:
    if row.get("language") != "zh":
        return False
    if not row.get("sft_ready", True):
        return False
    if row.get("source") not in HIGH_TRUST_SOURCES:
        return False
    title = str(row.get("job_title") or "").strip()
    lowered_title = title.lower()
    clean_text = str(row.get("clean_text") or "").strip()
    labels = row.get("labels") or {}
    direction = str(labels.get("岗位方向") or "").strip()
    if not title or not clean_text or not direction:
        return False
    if title_has_excluded_signal(lowered_title) and not title_has_exclusion_exception(lowered_title, direction):
        return False
    if not title_has_strong_tech_signal(lowered_title, direction):
        return False
    sections = row.get("sections") or {}
    has_responsibilities = bool(str(sections.get("responsibilities") or "").strip())
    has_requirements = bool(str(sections.get("requirements") or "").strip())
    has_skills = bool(labels.get("必备技能"))
    has_education = bool(labels.get("学历要求") or extract_education_requirement(clean_text))
    has_experience = bool(labels.get("经验要求") or extract_experience_requirement(clean_text))
    return (
        ((has_responsibilities and has_requirements) or (len(clean_text) >= 180 and (has_responsibilities or has_requirements)))
        and (has_education or has_experience or has_skills)
    )


def is_high_confidence_weak_tech_row(row: dict[str, Any]) -> bool:
    if row.get("language") != "zh":
        return False
    if row.get("source") not in WEAK_TECH_SOURCES:
        return False

    title = str(row.get("job_title") or "").strip().lower()
    if not title or not any(keyword in title for keyword in TECH_TITLE_KEYWORDS):
        return False
    if not any(keyword in title for keyword in WEAK_TITLE_INCLUDE_KEYWORDS):
        return False
    if any(keyword in title for keyword in WEAK_TITLE_EXCLUDE_KEYWORDS):
        return False

    clean_text = str(row.get("clean_text") or "").strip()
    if len(clean_text) < 180:
        return False

    sections = row.get("sections") or split_sections(clean_text)
    responsibilities = str(sections.get("responsibilities") or "").strip()
    requirements = str(sections.get("requirements") or "").strip()
    has_structure_marker = any(
        marker in clean_text
        for marker in (
            "岗位职责",
            "工作职责",
            "职位描述",
            "工作内容",
            "职责描述",
            "任职要求",
            "岗位要求",
            "职位要求",
            "任职资格",
            "能力要求",
            "技能要求",
        )
    )
    if not responsibilities or not requirements or not has_structure_marker:
        return False

    labels = row.get("labels") or {}
    education = str(labels.get("学历要求") or extract_education_requirement(clean_text)).strip()
    experience = str(labels.get("经验要求") or extract_experience_requirement(clean_text)).strip()
    skills = labels.get("必备技能") or extract_skills_from_text(clean_text, MINIMAL_SKILL_SCHEMA)

    if not education:
        return False
    if not (experience or skills):
        return False
    if len(clean_text) < 260 and not skills:
        return False
    return True


def collect_sft_rows(
    rows: list[dict[str, Any]],
    *,
    include_weak_tech: bool,
    target_total: int | None,
    seed: int,
    quality_profile: str,
) -> list[dict[str, Any]]:
    if quality_profile == "strict":
        strong_rows = [row for row in rows if is_high_trust_strong_row(row)]
    else:
        strong_rows = [row for row in rows if row.get("sft_ready", True)]
    if not include_weak_tech:
        return strong_rows

    chosen_ids = {str(row.get("id")) for row in strong_rows}
    weak_rows = [
        row
        for row in rows
        if str(row.get("id")) not in chosen_ids and is_high_confidence_weak_tech_row(row)
    ]
    if target_total is None or len(strong_rows) >= target_total:
        return strong_rows + weak_rows

    needed = max(0, target_total - len(strong_rows))
    rng = random.Random(seed)
    rng.shuffle(weak_rows)
    return strong_rows + weak_rows[:needed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", default="data/interim/jd_clean.jsonl")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--include-weak-tech", action="store_true")
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--quality-profile", choices=["strict", "expanded"], default="strict")
    args = parser.parse_args()

    rows = list(read_jsonl(args.jd))
    selected_rows = collect_sft_rows(
        rows,
        include_weak_tech=args.include_weak_tech,
        target_total=args.target_total,
        seed=args.seed,
        quality_profile=args.quality_profile,
    )
    samples = [build_jd_parse_sample(row) for row in selected_rows]
    splits = split_samples(samples, args.train_ratio, args.valid_ratio, args.seed)
    for split, rows in splits.items():
        write_jsonl(f"{args.out_dir}/{split}.jsonl", rows)
        print(f"wrote {len(rows)} {split} samples")


if __name__ == "__main__":
    main()
