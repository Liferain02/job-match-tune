from __future__ import annotations

from jobmatch_tune.crawler.didi_careers import (
    build_raw_text,
    convert_didi_job,
    is_probably_tech_job,
)


def test_is_probably_tech_job_accepts_title_keyword() -> None:
    post = {"jobName": "AI Agent工程师（研发效能方向）", "jobType": 1}
    assert is_probably_tech_job(post) is True


def test_is_probably_tech_job_rejects_non_tech_strategy() -> None:
    post = {"jobName": "Strategy and Planning Manager", "jobType": "战略"}
    assert is_probably_tech_job(post) is False


def test_build_raw_text_contains_sections() -> None:
    detail = {
        "jobName": "AI Agent工程师（研发效能方向）",
        "deptName": "技术平台",
        "workArea": "北京市",
        "publishTime": "2026-05-23 09:36:13",
        "refreshTime": "2026-05-23 09:36:13",
        "recruitType": "1",
        "jdNo": "JR20260203002",
        "recruitNum": 1,
        "jobType": "技术",
        "jobDesc": "负责 AI Agent 研发与平台建设",
        "qualification": "本科及以上，3年以上后端开发经验，熟悉 Python",
    }
    text = build_raw_text(detail)
    assert "公司名称：滴滴" in text
    assert "岗位职责：" in text
    assert "任职要求：" in text


def test_convert_didi_job_marks_tech_ready() -> None:
    detail = {
        "jobName": "AI Agent工程师（研发效能方向）",
        "deptName": "技术平台",
        "workArea": "北京市",
        "publishTime": "2026-05-23 09:36:13",
        "refreshTime": "2026-05-23 09:36:13",
        "recruitType": "1",
        "jdNo": "JR20260203002",
        "recruitNum": 1,
        "jobType": "技术",
        "jobDesc": "负责 AI Agent 研发与平台建设",
        "qualification": "本科及以上，3年以上后端开发经验，熟悉 Python",
    }
    row = convert_didi_job(detail, jd_id="61510", crawl_time="2026-05-24 12:00:00")
    assert row["id"] == "didi_61510"
    assert row["company"] == "滴滴"
    assert row["meta"]["sft_ready"] is True
    assert row["source"] == "talent.didiglobal.com"
