from jobmatch_tune.dataset.build_resume_train_pool_bootstrap import build_rows


def test_build_rows_generates_resume_rows_from_jd():
    rows = [
        {
            "id": "jd_1",
            "job_title": "后端开发工程师",
            "company": "示例公司",
            "raw_text": "岗位名称：后端开发工程师\n任职要求：本科及以上学历，熟悉 Java、MySQL、Redis、Linux。",
        }
    ]
    schema = {"skill_alias": {"Java": ["java"], "MySQL": ["mysql"], "Redis": ["redis"], "Linux": ["linux"]}}
    built = build_rows(rows, schema, 10)
    assert len(built) == 1
    assert built[0]["label"]["目标岗位"] == "后端开发"
    assert "Java" in built[0]["label"]["核心技能"]
