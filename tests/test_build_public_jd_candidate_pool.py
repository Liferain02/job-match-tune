from jobmatch_tune.dataset.build_public_jd_candidate_pool import (
    build_candidate_rows,
    is_usable_public_jd_row,
)


def test_is_usable_public_jd_row_accepts_zh_tech_row():
    row = {
        "id": "r1",
        "source": "github_workaggregation_test",
        "job_title": "后端开发工程师",
        "company": "示例公司",
        "location": "北京",
        "salary": "20-30K",
        "raw_text": (
            "岗位名称：后端开发工程师\n薪资范围：20-30K\n岗位职责：负责交易链路服务开发与治理，推进缓存优化、"
            "接口稳定性建设和告警治理，参与高并发接口开发、发布链路治理、监控告警建设和线上问题排查。"
            "负责数据库访问优化、缓存一致性治理、接口性能调优、链路监控告警建设、日志排障与容量评估，"
            "推动服务高可用与发布稳定性落地。\n经验要求：3年以上\n学历要求：本科及以上"
        ),
        "meta": {
            "language": "zh",
            "training_eligible": True,
            "intended_usage": "weak_supervision_only",
        },
    }
    assert is_usable_public_jd_row(row) is True


def test_build_candidate_rows_filters_and_deduplicates():
    row = {
        "id": "r1",
        "source": "github_workaggregation_test",
        "job_title": "后端开发工程师",
        "company": "示例公司",
        "location": "北京",
        "salary": "20-30K",
        "raw_text": (
            "岗位名称：后端开发工程师\n薪资范围：20-30K\n岗位职责：负责交易链路服务开发与治理，推进缓存优化、"
            "接口稳定性建设和告警治理，参与高并发接口开发、发布链路治理、监控告警建设和线上问题排查。"
            "负责数据库访问优化、缓存一致性治理、接口性能调优、链路监控告警建设、日志排障与容量评估，"
            "推动服务高可用与发布稳定性落地。\n经验要求：3年以上\n学历要求：本科及以上"
        ),
        "meta": {
            "language": "zh",
            "training_eligible": True,
            "intended_usage": "weak_supervision_only",
        },
    }
    rows = [row, dict(row, id="r2")]
    candidates = build_candidate_rows(rows)
    assert len(candidates) == 1


def test_is_usable_public_jd_row_blocks_audit_only_source():
    row = {
        "id": "r1",
        "source": "unlicensed_scrape",
        "job_title": "后端开发工程师",
        "salary": "20-30K",
        "raw_text": (
            "岗位名称：后端开发工程师\n薪资范围：20-30K\n岗位职责：负责服务开发、稳定性治理、"
            "数据库优化、监控告警、容量评估和线上问题排查。\n经验要求：3年以上\n学历要求：本科及以上"
        ),
        "meta": {
            "language": "zh",
            "training_eligible": False,
            "intended_usage": "audit_only",
        },
    }
    assert is_usable_public_jd_row(row) is False
