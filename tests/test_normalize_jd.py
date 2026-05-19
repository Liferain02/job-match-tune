from jobmatch_tune.preprocess.normalize_jd import normalize_jd_row


def test_normalize_jd_row_falls_back_to_meta_experience():
    row = {
        "id": "demo",
        "job_title": "后端开发工程师",
        "raw_text": "岗位职责：负责服务开发与接口治理。\n任职要求：本科及以上学历，熟悉 Java、MySQL。",
        "meta": {"work_year": "3-5年"},
    }
    schema = {"skill_alias": {"Java": [], "MySQL": []}}
    normalized = normalize_jd_row(row, schema)
    assert normalized["labels"]["经验要求"] == "3-5年"


def test_normalize_jd_row_prefers_text_experience_over_meta():
    row = {
        "id": "demo2",
        "job_title": "后端开发工程师",
        "raw_text": "岗位职责：负责服务开发。\n任职要求：3年以上工作经验，本科及以上学历，熟悉 Java。",
        "meta": {"work_year": "1-3年"},
    }
    schema = {"skill_alias": {"Java": []}}
    normalized = normalize_jd_row(row, schema)
    assert normalized["labels"]["经验要求"] == "3年以上工作经验"
