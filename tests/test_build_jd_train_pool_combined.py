from jobmatch_tune.dataset.build_jd_train_pool_combined import (
    build_combined_rows,
    build_manual_rows,
)


def test_build_manual_rows_keeps_only_strict_high_trust_rows():
    rows = [
        {
            "id": "strict_1",
            "source": "zhaopin.jd.com",
            "language": "zh",
            "job_title": "后端开发工程师",
            "company": "示例公司",
            "location": "北京",
            "salary": "20-30K",
            "clean_text": (
                "岗位职责：负责交易链路服务开发与治理，推进缓存优化、接口稳定性建设和告警治理。\n"
                "任职要求：本科及以上，熟悉 Java、MySQL、Redis。\n"
                "技能要求：熟悉 Linux、SQL 和日志排障。"
            ),
            "sections": {
                "responsibilities": "负责交易链路服务开发与治理。",
                "requirements": "本科及以上，熟悉 Java、MySQL、Redis。",
            },
            "labels": {"岗位方向": "后端开发", "必备技能": ["Java", "MySQL", "Redis"], "学历要求": "本科"},
            "sft_ready": True,
        },
        {
            "id": "noise_1",
            "source": "github_workaggregation_test",
            "language": "zh",
            "job_title": "销售经理",
            "clean_text": "岗位职责：负责销售",
            "sections": {},
            "labels": {},
            "sft_ready": False,
        },
    ]
    converted = build_manual_rows(rows)
    assert len(converted) == 1
    assert converted[0]["id"] == "strict_1"


def test_build_combined_rows_merges_and_deduplicates():
    manual_rows = [
        {
            "id": "strict_1",
            "source": "zhaopin.jd.com",
            "language": "zh",
            "job_title": "后端开发工程师",
            "company": "示例公司",
            "location": "北京",
            "salary": "20-30K",
            "clean_text": (
                "岗位职责：负责交易链路服务开发与治理，推进缓存优化、接口稳定性建设和告警治理。\n"
                "任职要求：本科及以上，熟悉 Java、MySQL、Redis。\n"
                "技能要求：熟悉 Linux、SQL 和日志排障。"
            ),
            "sections": {
                "responsibilities": "负责交易链路服务开发与治理。",
                "requirements": "本科及以上，熟悉 Java、MySQL、Redis。",
            },
            "labels": {"岗位方向": "后端开发", "必备技能": ["Java", "MySQL", "Redis"], "学历要求": "本科"},
            "sft_ready": True,
        }
    ]
    public_rows = [
        {
            "id": "public_1",
            "source": "github_workaggregation_test",
            "job_title": "后端开发工程师",
            "company": "示例公司",
            "location": "北京",
            "salary": "20-30K",
            "raw_text": (
                "岗位职责：负责交易链路服务开发与治理，推进缓存优化、接口稳定性建设和告警治理。\n"
                "任职要求：本科及以上，熟悉 Java、MySQL、Redis。\n"
                "技能要求：熟悉 Linux、SQL 和日志排障。"
            ),
            "meta": {"language": "zh"},
        }
    ]
    combined = build_combined_rows(manual_rows, public_rows)
    assert len(combined) == 1
