from jobmatch_tune.dataset.build_match_train_pool_synthetic import build_rows


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
            "text": "目标岗位：后端开发\n教育背景：本科，计算机科学与技术\n核心技能：Java、Spring Boot、MySQL、Redis",
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
