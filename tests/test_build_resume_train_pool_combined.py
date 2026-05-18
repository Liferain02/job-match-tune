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
        "meta": {"language": "zh"},
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
            "meta": {"language": "zh"},
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
            "meta": {"language": "zh"},
        },
    ]
    combined = build_combined_rows(manual_rows, public_rows)
    assert len(combined) == 2
