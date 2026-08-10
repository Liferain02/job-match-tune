from jobmatch_tune.dataset.build_match_train_pool_combined import (
    build_combined_rows,
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
            "language": "en",
            "license_status": "confirmed",
            "intended_usage": "training",
            "provenance_status": "human_annotated",
        },
    }
    assert is_usable_public_match_row(row) is True


def test_is_usable_public_match_row_rejects_unconfirmed_license():
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


def test_build_combined_rows_merges_and_deduplicates():
    manual_rows = [
        {
            "id": "manual_1",
            "task": "match",
            "source_type": "text",
            "jd_text": "岗位名称：前端开发工程师\n任职要求：熟悉 TypeScript、React、Vite、ECharts。",
            "resume_text": "目标岗位：前端开发\n核心技能：TypeScript、React、Vite、ECharts",
            "label": {"匹配等级": "高匹配"},
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
                "language": "en",
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
                "language": "en",
                "license_status": "confirmed",
                "intended_usage": "training",
                "provenance_status": "human_annotated",
            },
        },
    ]
    synthetic_rows = [
        {
            "id": "synthetic_1",
            "task": "match",
            "source_type": "synthetic_text",
            "jd_text": "岗位名称：AI Infra工程师\n任职要求：熟悉 Python、Go、Kubernetes、Linux。",
            "resume_text": "目标岗位：AI Infra\n核心技能：Python、Go、Kubernetes、Linux",
            "label": {"匹配等级": "高匹配", "raw_score": 95},
        }
    ]
    combined = build_combined_rows(manual_rows, public_rows, synthetic_rows)
    assert len(combined) == 3
