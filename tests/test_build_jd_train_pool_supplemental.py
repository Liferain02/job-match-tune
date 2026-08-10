from jobmatch_tune.dataset.build_jd_train_pool_supplemental import build_supplemental_rows


def test_build_supplemental_rows_keeps_only_high_confidence_weak_rows():
    rows = [
        {
            "id": "weak_1",
            "source": "github_workaggregation_test",
            "job_title": "后端开发工程师",
            "company": "A",
            "location": "北京",
            "salary": "20-30K",
            "language": "zh",
            "clean_text": (
                "岗位名称：后端开发工程师\n岗位职责：负责服务开发、缓存优化和接口治理。\n"
                "任职要求：熟悉 Java、Spring Boot、MySQL、Redis，本科及以上学历，3年以上工作经验。\n"
                "工作内容：参与交易系统接口开发、缓存治理、日志排障和性能优化，负责服务稳定性建设与问题定位。\n"
                "补充说明：需要参与高并发链路排障、数据库调优、发布治理、监控告警优化和线上事故复盘。"
            ),
            "sections": {
                "responsibilities": "负责服务开发、缓存优化和接口治理。",
                "requirements": "熟悉 Java、Spring Boot、MySQL、Redis，本科及以上学历，3年以上工作经验。",
            },
            "labels": {"岗位方向": "后端开发", "学历要求": "本科", "必备技能": ["Java", "MySQL", "Redis"]},
            "meta": {"training_eligible": True, "intended_usage": "weak_supervision_only"},
        },
        {
            "id": "weak_2",
            "source": "github_workaggregation_test",
            "job_title": "销售经理",
            "company": "B",
            "location": "上海",
            "salary": "10-20K",
            "language": "zh",
            "clean_text": "岗位名称：销售经理\n岗位职责：负责客户拓展。\n任职要求：沟通能力强。",
            "labels": {"岗位方向": ""},
        },
    ]
    built = build_supplemental_rows(rows)
    assert len(built) == 1
    assert built[0]["id"] == "weak_1"
    assert built[0]["meta"]["training_eligible"] is True


def test_build_supplemental_rows_blocks_audit_only_external_source():
    row = {
        "id": "blocked",
        "source": "github_workaggregation_test",
        "job_title": "后端开发工程师",
        "language": "zh",
        "clean_text": "岗位职责：负责 Java 服务开发、稳定性治理。任职要求：本科，熟悉 Java、MySQL。" * 5,
        "sections": {"responsibilities": "负责 Java 服务开发。", "requirements": "本科，熟悉 Java、MySQL。"},
        "labels": {"岗位方向": "后端开发", "学历要求": "本科", "必备技能": ["Java", "MySQL"]},
        "meta": {"training_eligible": False, "intended_usage": "audit_only"},
    }
    assert build_supplemental_rows([row]) == []
