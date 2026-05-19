from jobmatch_tune.dataset.build_jd_bootstrap_sft_dataset import build_bootstrap_rows


def test_build_bootstrap_rows_normalizes_pool_rows():
    rows = [
        {
            "id": "pool_1",
            "job_title": "后端开发工程师",
            "source": "github_workaggregation_test",
            "company": "示例公司",
            "location": "北京",
            "raw_text": (
                "岗位名称：后端开发工程师\n岗位职责：负责服务开发、缓存优化和接口治理。\n"
                "任职要求：本科及以上学历，熟悉 Java、MySQL、Redis。"
            ),
            "meta": {"language": "zh"},
        }
    ]
    schema = {"skill_alias": {"Java": ["java"], "MySQL": ["mysql"], "Redis": ["redis"]}}
    built = build_bootstrap_rows(rows, schema)
    assert len(built) == 1
    assert built[0]["labels"]["岗位方向"] == "后端开发"
    assert "Java" in built[0]["labels"]["必备技能"]
