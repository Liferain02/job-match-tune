import random

from jobmatch_tune.dataset.build_match_train_pool_synthetic import (
    build_jd_structured,
    build_rows,
    partition_match_entities,
    sanitize_jd_text_for_matching,
    select_jds_with_educational_source_cap,
)


def test_build_rows_generates_positive_and_negative_pairs():
    jd_rows = [
        {
            "id": "jd_1",
            "job_title": "后端开发工程师",
            "raw_text": "岗位名称：后端开发工程师\n任职要求：熟悉 Java、Spring Boot、MySQL、Redis，本科及以上学历，3年以上工作经验。",
        }
    ]
    resume_rows = [
        {
            "id": "resume_pos",
            "text": "姓名：张三\n电话：138-1234-5678\n婚姻状况：未婚\n目标岗位：后端开发\n教育背景：本科，计算机科学与技术\n核心技能：Java、Spring Boot、MySQL、Redis",
            "label": {
                "目标岗位": "后端开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Java", "Spring Boot", "MySQL", "Redis"],
                "项目经历": ["负责订单中心重构。"],
                "实习经历": ["参与接口开发。"],
            },
        },
        {
            "id": "resume_neg",
            "text": "目标岗位：前端开发\n教育背景：本科，软件工程\n核心技能：TypeScript、React、Vite",
            "label": {
                "目标岗位": "前端开发",
                "教育背景": ["本科，软件工程"],
                "核心技能": ["TypeScript", "React", "Vite"],
                "项目经历": ["负责活动页开发。"],
                "实习经历": ["参与前端页面开发。"],
            },
        },
    ]
    schema = {
        "skill_alias": {
            "Java": ["java"],
            "Spring Boot": ["spring boot"],
            "MySQL": ["mysql"],
            "Redis": ["redis"],
        }
    }
    rows = build_rows(
        jd_rows,
        resume_rows,
        schema,
        seed=1,
        positive_per_jd=1,
        negatives_per_jd=1,
        max_jd_rows=10,
    )
    assert len(rows) == 2
    levels = {row["label"]["匹配等级"] for row in rows}
    assert "高匹配" in levels or "较匹配" in levels
    assert "低匹配" in levels or "基本匹配" in levels
    positive = next(row for row in rows if "resume_pos" in row["id"])
    assert "张三" not in positive["resume_text"]
    assert "138-1234-5678" not in positive["resume_text"]
    assert "婚姻状况" not in positive["resume_text"]
    assert positive["meta"]["entity_split"] == "train"
    assert positive["meta"]["jd_entity_hash"]
    assert positive["meta"]["resume_entity_hash"]


def test_match_jd_sanitizer_removes_upstream_label_wrapper() -> None:
    text = "岗位名称：后端工程师\n任务类型：从岗位中提取学历\n岗位描述：负责服务开发。\n学历提示：本科"
    schema = {"skill_alias": {}}

    sanitized = sanitize_jd_text_for_matching(text)
    structured = build_jd_structured(
        {"job_title": "后端工程师", "raw_text": text},
        schema,
    )

    assert "任务类型" not in sanitized
    assert "学历提示" not in sanitized
    assert "岗位描述：负责服务开发" in sanitized
    assert structured["学历要求"] == ""


def test_match_jd_selection_caps_weak_educational_source() -> None:
    rows = [
        ({"id": f"hf_job_educational_{index}"}, {"岗位方向": "后端开发"})
        for index in range(8)
    ] + [
        ({"id": f"official_{index}"}, {"岗位方向": "后端开发"})
        for index in range(4)
    ]

    selected = select_jds_with_educational_source_cap(
        rows,
        max_rows=10,
        max_educational_source_rate=0.4,
        rng=random.Random(42),
    )

    educational = sum(row[0]["id"].startswith("hf_job_educational_") for row in selected)
    assert len(selected) == 8
    assert educational == 4


def test_match_jd_selection_is_stable_when_unselected_candidate_is_removed() -> None:
    rows = [
        (
            {"id": f"official_{index}", "raw_text": f"岗位描述 {index}"},
            {"岗位方向": "后端开发"},
        )
        for index in range(30)
    ]

    selected = select_jds_with_educational_source_cap(
        rows,
        max_rows=10,
        max_educational_source_rate=0.4,
        rng=random.Random(42),
    )
    selected_ids = {item[0]["id"] for item in selected}
    unselected_id = next(item[0]["id"] for item in rows if item[0]["id"] not in selected_ids)
    selected_after_removal = select_jds_with_educational_source_cap(
        [item for item in rows if item[0]["id"] != unselected_id],
        max_rows=10,
        max_educational_source_rate=0.4,
        rng=random.Random(42),
    )

    assert [item[0]["id"] for item in selected_after_removal] == [
        item[0]["id"] for item in selected
    ]


def test_match_entities_are_partitioned_before_pairing() -> None:
    jds = [
        (
            {
                "id": f"{'hf_job_educational_' if index < 3 else 'official_'}{index}",
                "raw_text": f"岗位描述 {index}",
            },
            {"岗位方向": "后端开发"},
        )
        for index in range(6)
    ]
    resumes = [
        {
            "id": f"resume_{index}",
            "text": f"目标岗位：后端开发\n核心技能：Java\n项目：{index}",
            "label": {"目标岗位": "后端开发"},
        }
        for index in range(6)
    ]

    jd_splits, resume_splits = partition_match_entities(jds, resumes, seed=42)

    assert {split: len(rows) for split, rows in jd_splits.items()} == {
        "train": 2,
        "valid": 2,
        "test": 2,
    }
    assert {split: len(rows) for split, rows in resume_splits.items()} == {
        "train": 4,
        "valid": 1,
        "test": 1,
    }
