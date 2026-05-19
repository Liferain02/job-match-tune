from jobmatch_tune.dataset.build_jd_bootstrap_sft_dataset import build_bootstrap_rows, is_usable_bootstrap_row


def test_build_bootstrap_rows_normalizes_pool_rows():
    rows = [
        {
            "id": "pool_1",
            "job_title": "后端开发工程师",
            "source": "zhaopin.jd.com",
            "company": "示例公司",
            "location": "北京",
            "raw_text": (
                "岗位名称：后端开发工程师\n岗位职责：负责服务开发、缓存优化和接口治理。\n"
                "任职要求：本科及以上学历，熟悉 Java、MySQL、Redis，有3年以上相关开发经验。"
            ),
            "meta": {"language": "zh"},
        }
    ]
    schema = {"skill_alias": {"Java": ["java"], "MySQL": ["mysql"], "Redis": ["redis"]}}
    built = build_bootstrap_rows(rows, schema)
    assert len(built) == 1
    assert built[0]["labels"]["岗位方向"] == "后端开发"
    assert "Java" in built[0]["labels"]["必备技能"]


def test_is_usable_bootstrap_row_filters_low_signal_weak_rows():
    weak_row = {
        "job_title": "产品实习生",
        "source": "hf_job_educational_train_2026_05_17",
        "clean_text": "岗位名称：产品实习生\n岗位职责：协助产品经理处理需求。\n学历提示：本科",
        "labels": {"岗位方向": "产品经理", "学历要求": "本科", "必备技能": []},
        "sections": {"responsibilities": "协助产品经理处理需求。"},
    }
    assert is_usable_bootstrap_row(weak_row) is False
