from jobmatch_tune.crawler.jd_careers import convert_jd_post, is_probably_tech_job


def test_is_probably_tech_job_accepts_yanfa() -> None:
    post = {"jobTypeCode": "YANFA", "positionNameOpen": "策略运营"}
    assert is_probably_tech_job(post) is True


def test_convert_jd_post_marks_non_tech_false() -> None:
    post = {
        "requirementId": 1001,
        "positionId": 2002,
        "positionNameOpen": "销售经理",
        "positionDeptName": "京东零售",
        "jobType": "运营类",
        "jobTypeCode": "YUNGYUN",
        "workCity": "北京市",
        "formatPublishTime": "2026-05-17",
        "reqNumber": "ZPTEST001",
        "workContent": "负责销售拓展与客户维护。",
        "qualification": "专科及以上学历。",
    }
    converted = convert_jd_post(post, "2026-05-17 00:00:00")
    assert converted["id"] == "jd_careers_1001"
    assert converted["company"] == "京东"
    assert converted["meta"]["sft_ready"] is False
    assert "岗位职责：" in converted["raw_text"]
