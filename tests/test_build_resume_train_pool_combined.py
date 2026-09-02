from jobmatch_tune.dataset.build_resume_train_pool_combined import (
    build_combined_rows,
    is_usable_public_resume_row,
)


def test_is_usable_public_resume_row_requires_resume_parse_and_signals():
    row = {
        "task": "resume_parse",
        "text": (
            "姓名：张三\n目标岗位：后端开发工程师\n教育背景：本科，计算机科学与技术\n"
            "核心技能：Java、MySQL、Redis、Spring Boot\n项目经历：负责订单中心重构和缓存优化，"
            "参与支付服务接口开发与压测，完成链路追踪接入和告警治理。"
        ),
        "label": {
            "目标岗位": "后端开发",
            "教育背景": ["本科，计算机科学与技术"],
            "核心技能": ["Java", "MySQL"],
            "项目经历": ["订单中心重构"],
            "实习经历": [],
        },
        "meta": {
            "language": "zh",
            "license_status": "confirmed",
            "intended_usage": "training",
        },
    }
    assert is_usable_public_resume_row(row) is True


def test_build_combined_rows_merges_and_deduplicates():
    manual_rows = [
        {
            "id": "manual_1",
            "task": "resume_parse",
            "source_type": "text",
            "text": "姓名：张三\n目标岗位：后端开发工程师\n教育背景：本科",
            "label": {"目标岗位": "后端开发"},
        }
    ]
    public_rows = [
        {
            "id": "public_1",
            "task": "resume_parse",
            "source_type": "public_text",
            "text": (
                "姓名：李四\n目标岗位：后端开发工程师\n教育背景：本科，计算机科学与技术\n"
                "核心技能：Java、MySQL、Redis、Spring Boot\n项目经历：负责订单中心重构和缓存优化，"
                "参与支付服务接口开发与压测，完成链路追踪接入和告警治理。"
            ),
            "label": {
                "目标岗位": "后端开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Java", "MySQL"],
                "项目经历": ["订单中心重构"],
                "实习经历": [],
            },
            "meta": {
                "language": "zh",
                "license_status": "confirmed",
                "intended_usage": "training",
            },
        },
        {
            "id": "public_2",
            "task": "resume_parse",
            "source_type": "public_text",
            "text": (
                "姓名：李四\n目标岗位：后端开发工程师\n教育背景：本科，计算机科学与技术\n"
                "核心技能：Java、MySQL、Redis、Spring Boot\n项目经历：负责订单中心重构和缓存优化，"
                "参与支付服务接口开发与压测，完成链路追踪接入和告警治理。"
            ),
            "label": {
                "目标岗位": "后端开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Java", "MySQL"],
                "项目经历": ["订单中心重构"],
                "实习经历": [],
            },
            "meta": {
                "language": "zh",
                "license_status": "confirmed",
                "intended_usage": "training",
            },
        },
    ]
    combined = build_combined_rows(manual_rows, public_rows)
    assert len(combined) == 2
    assert all("姓名：" not in row["text"] for row in combined)


def test_public_resume_candidate_only_source_never_enters_training_pool():
    row = {
        "id": "candidate_only",
        "task": "resume_parse",
        "source_type": "public_text",
        "text": "目标岗位：后端开发\n教育背景：本科\n核心技能：Java、MySQL\n项目经历：负责订单中心开发和缓存优化，完成接口治理和压测。",
        "label": {
            "目标岗位": "后端开发",
            "教育背景": ["本科"],
            "核心技能": ["Java", "MySQL"],
            "项目经历": ["订单中心"],
        },
        "meta": {
            "language": "zh",
            "license_status": "unconfirmed",
            "intended_usage": "candidate_pool_only",
        },
    }

    assert is_usable_public_resume_row(row) is False
    assert build_combined_rows([], [row]) == []


def test_default_resume_pool_rejects_english_even_when_training_is_allowed():
    row = {
        "task": "resume_parse",
        "text": (
            "Target role: Backend Engineer. Education: Bachelor of Computer Science. "
            "Skills: Python, PostgreSQL. Projects: built a production API and monitoring pipeline."
        ),
        "label": {
            "目标岗位": "Backend Engineer",
            "教育背景": ["Bachelor of Computer Science"],
            "核心技能": ["Python", "PostgreSQL"],
            "项目经历": ["built a production API and monitoring pipeline"],
        },
        "meta": {
            "language": "en",
            "license_status": "source_declared_cc",
            "intended_usage": "sft_training",
        },
    }

    assert is_usable_public_resume_row(row) is False


def test_combined_resume_pool_excludes_product_manager_for_technical_scope():
    row = {
        "id": "product_resume",
        "task": "resume_parse",
        "text": "目标岗位：产品经理\n教育背景：本科\n项目经历：负责需求设计和版本迭代。",
        "label": {"目标岗位": "产品经理"},
    }

    assert build_combined_rows([row], []) == []


def test_combined_resume_pool_excludes_teacher_and_sales_targets():
    rows = [
        {
            "id": f"non_technical_{target}",
            "task": "resume_parse",
            "text": f"目标岗位：{target}\n教育背景：本科\n项目经历：相关项目。",
            "label": {"目标岗位": target},
        }
        for target in ("教师", "销售")
    ]

    assert build_combined_rows(rows, []) == []
