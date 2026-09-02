from jobmatch_tune.dataset.build_match_train_pool_combined import (
    build_combined_rows,
    cap_educational_source_rows,
    is_usable_public_match_row,
)


def test_is_usable_public_match_row_requires_pair_text_and_label():
    row = {
        "task": "match",
        "jd_text": (
            "岗位名称：后端开发工程师\n岗位职责：负责交易链路服务开发与治理。\n"
            "任职要求：熟悉 Java、Spring Boot、MySQL、Redis、Kafka，"
            "负责高并发接口开发、服务治理、缓存优化和稳定性建设。"
        ),
        "resume_text": (
            "目标岗位：后端开发\n教育背景：本科，计算机科学与技术\n"
            "核心技能：Java、Spring Boot、MySQL、Redis、Kafka\n"
            "项目经历：负责订单中心重构与缓存优化，参与支付服务接口开发、压测和链路追踪接入。"
        ),
        "label": {"raw_label": "fit", "raw_score": 0.91},
        "meta": {
            "language": "zh",
            "jd_direction": "后端开发",
            "resume_direction": "后端开发",
            "license_status": "confirmed",
            "intended_usage": "training",
            "provenance_status": "human_annotated",
        },
    }
    assert is_usable_public_match_row(row) is True


def test_is_usable_public_match_row_rejects_audit_only_usage():
    row = {
        "task": "match",
        "jd_text": "岗位描述" * 30,
        "resume_text": "简历内容" * 30,
        "label": {"raw_label": "fit"},
        "meta": {
            "language": "zh",
            "license_status": "unconfirmed",
            "intended_usage": "audit_only",
            "provenance_status": "human_annotated",
        },
    }
    assert is_usable_public_match_row(row) is False


def test_default_match_pool_rejects_english_even_when_training_is_allowed():
    row = {
        "task": "match",
        "jd_text": "Backend role requiring Python, PostgreSQL, API design, testing, and monitoring. " * 3,
        "resume_text": "Backend engineer with Python, PostgreSQL, API, testing, and monitoring experience. " * 3,
        "label": {"raw_label": "match", "raw_score": 8.0},
        "meta": {
            "language": "en",
            "license_status": "source_declared_cc",
            "intended_usage": "sft_training",
            "provenance_status": "documented_machine_generated",
        },
    }

    assert is_usable_public_match_row(row) is False


def test_combined_match_pool_excludes_product_manager_for_technical_scope():
    row = {
        "id": "product_pair",
        "task": "match",
        "jd_text": "产品经理岗位描述",
        "resume_text": "产品经理候选人简历",
        "label": {"匹配等级": "较匹配"},
        "meta": {"jd_direction": "产品经理", "resume_direction": "产品经理"},
    }

    assert build_combined_rows([row], []) == []


def test_combined_match_pool_excludes_teacher_direction():
    row = {
        "id": "teacher_pair",
        "task": "match",
        "jd_text": "教师岗位描述",
        "resume_text": "教师候选人简历",
        "label": {"匹配等级": "较匹配"},
        "meta": {"jd_direction": "教师", "resume_direction": "教师"},
    }

    assert build_combined_rows([row], []) == []


def test_build_combined_rows_merges_and_deduplicates():
    manual_rows = [
        {
            "id": "manual_1",
            "task": "match",
            "source_type": "text",
            "jd_text": "岗位名称：前端开发工程师\n任职要求：熟悉 TypeScript、React、Vite、ECharts。",
            "resume_text": "目标岗位：前端开发\n核心技能：TypeScript、React、Vite、ECharts",
            "label": {"匹配等级": "高匹配"},
            "meta": {"jd_direction": "前端开发", "resume_direction": "前端开发"},
        }
    ]
    public_rows = [
        {
            "id": "public_1",
            "task": "match",
            "source_type": "public_pair",
            "jd_text": (
                "岗位名称：后端开发工程师\n岗位职责：负责交易链路服务开发与治理。\n"
                "任职要求：熟悉 Java、Spring Boot、MySQL、Redis、Kafka，"
                "负责高并发接口开发、服务治理、缓存优化和稳定性建设。"
            ),
            "resume_text": (
                "目标岗位：后端开发\n教育背景：本科，计算机科学与技术\n"
                "核心技能：Java、Spring Boot、MySQL、Redis、Kafka\n"
                "项目经历：负责订单中心重构与缓存优化，参与支付服务接口开发、压测和链路追踪接入。"
            ),
            "label": {"raw_label": "fit", "raw_score": 0.91},
            "meta": {
                "language": "zh",
                "jd_direction": "后端开发",
                "resume_direction": "后端开发",
                "license_status": "confirmed",
                "intended_usage": "training",
                "provenance_status": "human_annotated",
            },
        },
        {
            "id": "public_2",
            "task": "match",
            "source_type": "public_pair",
            "jd_text": (
                "岗位名称：后端开发工程师\n岗位职责：负责交易链路服务开发与治理。\n"
                "任职要求：熟悉 Java、Spring Boot、MySQL、Redis、Kafka，"
                "负责高并发接口开发、服务治理、缓存优化和稳定性建设。"
            ),
            "resume_text": (
                "目标岗位：后端开发\n教育背景：本科，计算机科学与技术\n"
                "核心技能：Java、Spring Boot、MySQL、Redis、Kafka\n"
                "项目经历：负责订单中心重构与缓存优化，参与支付服务接口开发、压测和链路追踪接入。"
            ),
            "label": {"raw_label": "fit", "raw_score": 0.91},
            "meta": {
                "language": "zh",
                "jd_direction": "后端开发",
                "resume_direction": "后端开发",
                "license_status": "confirmed",
                "intended_usage": "training",
                "provenance_status": "human_annotated",
            },
        },
    ]
    combined = build_combined_rows(manual_rows, public_rows)
    assert len(combined) == 2


def test_final_match_pool_reapplies_educational_cap_after_deduplication():
    rows = [
        {"id": f"synthetic_match_hf_job_educational_{index}"}
        for index in range(5)
    ] + [{"id": f"synthetic_match_official_{index}"} for index in range(6)]

    capped = cap_educational_source_rows(rows, max_rate=0.4)

    educational = sum("hf_job_educational_" in row["id"] for row in capped)
    assert len(capped) == 10
    assert educational == 4
