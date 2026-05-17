from jobmatch_tune.crawler.moka_careers import (
    build_experience_string,
    build_salary_string,
    convert_moka_job,
    detect_language,
    is_probably_tech_job,
    strip_html_text,
)


def test_strip_html_text_converts_breaks() -> None:
    text = strip_html_text("<p>岗位职责</p><p>负责Python开发<br/>维护服务</p>")
    assert "岗位职责" in text
    assert "负责Python开发" in text
    assert "维护服务" in text


def test_detect_language_prefers_zh() -> None:
    assert detect_language("岗位职责：负责大模型应用开发") == "zh"
    assert detect_language("Build AI platform services") == "en"


def test_build_salary_and_experience_string() -> None:
    assert build_salary_string({"minSalary": 30, "maxSalary": 50}) == "30-50K"
    assert build_experience_string({"minExperience": 3, "maxExperience": 5}) == "3-5年"


def test_is_probably_tech_job_matches_title() -> None:
    job = {"title": "后端开发工程师", "department": {"name": "平台研发"}}
    assert is_probably_tech_job(job, "负责服务端开发") is True


def test_convert_moka_job_marks_sft_ready_for_zh_tech() -> None:
    job = {
        "id": "job-001",
        "title": "算法工程师",
        "description": "<p>岗位职责：负责推荐算法、模型训练与效果优化。</p>",
        "education": "本科",
        "minExperience": 2,
        "maxExperience": 5,
        "minSalary": 40,
        "maxSalary": 70,
        "commitment": "全职",
        "department": {"name": "技术部"},
        "locations": [{"country": "中国", "city": "上海"}],
        "zhineng": {"name": "技术类"},
        "status": "open",
        "publishedAt": "2026-05-17T00:00:00.000Z",
    }
    converted = convert_moka_job(
        job,
        org_id="eastmoney",
        company="东方财富",
        mode="social",
        site_id=57970,
        source_name="moka_eastmoney",
        source_url="https://app.mokahr.com/social-recruitment/eastmoney/57970#/",
        crawl_time="2026-05-17 00:00:00",
    )
    assert converted["id"] == "moka_eastmoney_social_job-001"
    assert converted["company"] == "东方财富"
    assert converted["meta"]["language"] == "zh"
    assert converted["meta"]["sft_ready"] is True
    assert "岗位名称：算法工程师" in converted["raw_text"]
