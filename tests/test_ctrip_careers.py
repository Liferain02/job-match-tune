from __future__ import annotations

from jobmatch_tune.crawler.ctrip_careers import (
    build_raw_text,
    convert_ctrip_job,
    is_probably_tech_job,
)


def test_is_probably_tech_job_accepts_family_code() -> None:
    post = {"jobTitle": "高级/资深Android开发工程师", "jobFamilyGroupCode": "JFG_31"}
    assert is_probably_tech_job(post) is True


def test_is_probably_tech_job_rejects_non_tech_strategy() -> None:
    post = {
        "jobTitle": "用户增长渠道拓展专家",
        "jobFamilyGroupCode": "JFG_61",
        "jobFamilyGroupName": "Business development",
        "requirements": "5年以上渠道拓展经验",
    }
    assert is_probably_tech_job(post) is False


def test_build_raw_text_contains_sections() -> None:
    post = {
        "jobTitle": "高级/资深Android开发工程师(MJ032645)",
        "cityName": "Shanghai",
        "publishDate": "2026-05-22",
        "jobFamilyGroupName": "Software development",
        "jobFamilyGroupCode": "JFG_31",
        "buName": "Accommodation",
        "requirements": "1. 2年以上Android UI开发经验；2. 熟悉Java/Kotlin。",
    }
    text = build_raw_text(post)
    assert "公司名称：携程" in text
    assert "职位描述：" in text
    assert "岗位名称：" in text


def test_convert_ctrip_job_marks_tech_ready() -> None:
    post = {
        "id": "27908962",
        "jobId": "b370ee7a-92f3-4ed8-8768-1bf27cebf9e8",
        "fromId": "MJ032645",
        "jobTitle": "高级/资深Android开发工程师(MJ032645)",
        "publishDate": "2026-05-22",
        "cityName": "Shanghai",
        "jobFamilyGroupCode": "JFG_31",
        "jobFamilyGroupName": "Software development",
        "buName": "Accommodation",
        "requirements": "2年以上Android UI开发经验，熟悉Java/Kotlin。",
    }
    row = convert_ctrip_job(post, crawl_time="2026-05-24 20:00:00")
    assert row["id"] == "ctrip_27908962"
    assert row["company"] == "携程"
    assert row["source"] == "careers.ctrip.com"
    assert row["meta"]["sft_ready"] is True
