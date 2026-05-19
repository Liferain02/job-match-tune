from __future__ import annotations

from jobmatch_tune.crawler.meituan_careers import (
    build_raw_text,
    convert_meituan_job,
    is_probably_tech_job,
)


def test_is_probably_tech_job_accepts_family() -> None:
    post = {"name": "商品BML产品运营", "jobFamily": "技术类", "jobDuty": "负责流程建设"}
    assert is_probably_tech_job(post) is True


def test_is_probably_tech_job_rejects_generic_product_family_without_signal() -> None:
    post = {"name": "商品BML产品运营", "jobFamily": "产品类", "jobDuty": "负责运营策略制定"}
    assert is_probably_tech_job(post) is False


def test_is_probably_tech_job_accepts_title_keyword() -> None:
    post = {"name": "AI Agent研发工程师", "jobFamily": "职能类", "jobDuty": "负责 agent pipeline"}
    assert is_probably_tech_job(post) is True


def test_build_raw_text_contains_sections() -> None:
    detail = {
        "name": "AI Agent研发工程师",
        "jobFamily": "技术类",
        "jobFamilyGroup": "软件",
        "cityList": [{"name": "北京市"}],
        "workYear": "3年",
        "department": [{"name": "软硬件服务-软件研发部"}],
        "departmentIntro": "部门介绍",
        "jobDuty": "职责内容",
        "jobRequirement": "要求内容",
        "precedence": "优先条件",
        "highLight": "职位亮点",
    }
    text = build_raw_text(detail)
    assert "公司名称：美团" in text
    assert "岗位职责：" in text
    assert "任职要求：" in text


def test_convert_meituan_job() -> None:
    detail = {
        "jobUnionId": "4449544248",
        "name": "AI Agent研发工程师",
        "jobFamily": "技术类",
        "jobFamilyGroup": "软件",
        "cityList": [{"name": "北京市"}],
        "workYear": "3年",
        "department": [{"name": "软硬件服务-软件研发部"}],
        "departmentIntro": "部门介绍",
        "jobDuty": "职责内容",
        "jobRequirement": "要求内容",
        "precedence": "优先条件",
        "highLight": "职位亮点",
    }
    row = convert_meituan_job(detail, crawl_time="2026-05-19 12:00:00")
    assert row["id"] == "meituan_4449544248"
    assert row["meta"]["sft_ready"] is True
    assert row["company"] == "美团"
