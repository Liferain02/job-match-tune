from jobmatch_tune.dataset.build_jd_strict_plus_sft_dataset import is_strict_plus_row


def test_is_strict_plus_row_accepts_stronger_weak_source():
    row = {
        "job_title": "后端开发工程师",
        "source": "hf_job_educational_train_2026_05_17",
        "labels": {"岗位方向": "后端开发", "学历要求": "本科", "必备技能": ["Java", "MySQL"]},
        "sections": {
            "responsibilities": "负责服务开发、缓存优化和接口治理，参与链路追踪、告警建设、接口稳定性治理和数据库性能优化。",
            "requirements": "本科及以上学历，熟悉 Java、MySQL、Redis，理解分布式系统基础，具备线上问题排查经验。",
        },
    }
    assert is_strict_plus_row(row) is True


def test_is_strict_plus_row_rejects_low_signal_title():
    row = {
        "job_title": "产品实习生",
        "source": "hf_job_educational_train_2026_05_17",
        "labels": {"岗位方向": "产品经理", "学历要求": "本科", "必备技能": ["Axure"]},
        "sections": {"responsibilities": "协助产品经理做需求。", "requirements": "本科及以上。"},
    }
    assert is_strict_plus_row(row) is False
