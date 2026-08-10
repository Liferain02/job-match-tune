from jobmatch_tune.dataset.build_jd_train_pool_weak_structured import build_rows


def test_build_rows_keeps_weak_structured_technical_jd():
    rows = [
        {
            "id": "weak_struct_1",
            "source": "hf_job_educational_train_2026_05_17",
            "language": "zh",
            "job_title": "后端开发工程师",
            "company": "A",
            "location": "北京",
            "salary": "",
            "clean_text": (
                "岗位名称：后端开发工程师\n岗位描述：\n岗位职责：负责服务开发、接口治理、缓存优化和监控告警建设；"
                "任职要求：本科及以上学历，熟悉 Java、MySQL、Redis、Linux，有高并发系统经验优先。"
                "补充说明：需要参与链路追踪、日志排障、缓存治理、告警优化和发布稳定性建设。"
                "同时负责接口性能压测、灰度发布支持和线上问题复盘。"
            ),
            "sections": {
                "responsibilities": "负责服务开发、接口治理、缓存优化和监控告警建设。",
                "requirements": "本科及以上学历，熟悉 Java、MySQL、Redis、Linux，有高并发系统经验优先。",
            },
            "labels": {"岗位方向": "后端开发", "学历要求": "本科", "必备技能": ["Java", "MySQL", "Redis"]},
            "meta": {"training_eligible": True, "intended_usage": "weak_supervision_only"},
        },
        {
            "id": "weak_struct_2",
            "source": "hf_job_educational_train_2026_05_17",
            "language": "zh",
            "job_title": "招商主管",
            "company": "B",
            "location": "上海",
            "salary": "",
            "clean_text": "岗位名称：招商主管\n岗位职责：负责客户拓展和商务谈判。\n任职要求：本科及以上学历。",
            "sections": {"requirements": "本科及以上学历。"},
            "labels": {"岗位方向": "产品经理", "学历要求": "本科"},
            "meta": {"training_eligible": True, "intended_usage": "weak_supervision_only"},
        },
    ]
    built = build_rows(rows)
    assert len(built) == 1
    assert built[0]["id"] == "weak_struct_1"
    assert built[0]["meta"]["training_eligible"] is True


def test_build_rows_blocks_external_source_without_training_opt_in():
    row = {
        "id": "blocked",
        "source": "github_workaggregation_test",
        "language": "zh",
        "job_title": "后端开发工程师",
        "clean_text": "岗位职责：负责 Java 服务开发、稳定性治理和监控告警。任职要求：本科，熟悉 Java、MySQL。" * 5,
        "sections": {"responsibilities": "负责 Java 服务开发。", "requirements": "本科，熟悉 Java、MySQL。"},
        "labels": {"岗位方向": "后端开发", "学历要求": "本科", "必备技能": ["Java", "MySQL"]},
        "meta": {"training_eligible": False, "intended_usage": "audit_only"},
    }
    assert build_rows([row]) == []


def test_build_rows_accepts_single_pass_iterable():
    row = {
        "id": "streamed",
        "source": "hf_job_educational_train_2026_05_17",
        "language": "zh",
        "job_title": "后端开发工程师",
        "clean_text": (
            "岗位职责：负责 Java 服务开发、稳定性治理、发布流程和监控告警建设。"
            "任职要求：本科及以上学历，熟悉 Java、MySQL、Redis 和 Linux，"
            "具备高并发系统开发经验，能够完成问题排查、性能优化和项目交付。"
            "同时参与接口治理、链路追踪、容量规划、故障复盘、灰度发布和技术文档维护。"
        ),
        "sections": {"responsibilities": "负责 Java 服务开发。", "requirements": "本科及以上。"},
        "labels": {"岗位方向": "后端开发", "学历要求": "本科", "必备技能": ["Java"]},
        "meta": {"training_eligible": True, "intended_usage": "weak_supervision_only"},
    }

    built = build_rows(row for row in [row])

    assert [item["id"] for item in built] == ["streamed"]
