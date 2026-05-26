from __future__ import annotations

from jobmatch_tune.crawler.xiaomi_careers import (
    build_search_path,
    build_page_path,
    convert_xiaomi_job,
    parse_detail_fields,
    parse_list_rows,
)


def test_parse_list_rows() -> None:
    html = """
    <tbody>
      <tr>
        <td class="first"><a href="https://hr.xiaomi.com/job/view/800">Android Multimedia系统工程师（小米电视）<span class="hot-tip"></span></a></td>
        <td>研发工程师</td>
        <td>Peking</td>
        <td>2017-08-31</td>
      </tr>
    </tbody>
    """
    rows = parse_list_rows(html)
    assert len(rows) == 1
    assert rows[0]["job_id"] == "800"
    assert rows[0]["title"] == "Android Multimedia系统工程师（小米电视）"
    assert rows[0]["location"] == "北京"


def test_parse_detail_fields() -> None:
    html = """
    <table class="job-information">
      <tr>
        <td class="details-title">职位名称：</td>
        <td class="job-details">天线工程师</td>
        <td class="details-title">工作地点：</td>
        <td class="job-details">Peking</td>
      </tr>
      <tr>
        <td class="details-title require">工作职责：</td>
        <td class="details-list" colspan="3">1. 负责开发；<br />2. 负责优化；<br /></td>
      </tr>
      <tr>
        <td class="details-title require">工作要求：</td>
        <td class="details-list" colspan="3">1. 本科及以上；<br />2. 3年以上经验；<br /></td>
      </tr>
    </table>
    """
    fields = parse_detail_fields(html)
    assert fields["职位名称"] == "天线工程师"
    assert fields["工作地点"] == "Peking"
    assert "负责开发" in fields["工作职责"]
    assert "本科及以上" in fields["工作要求"]


def test_convert_xiaomi_job_marks_tech_ready() -> None:
    row = {
        "job_id": "800",
        "href": "https://hr.xiaomi.com/job/view/800",
        "title": "Android Multimedia系统工程师（小米电视）",
        "category": "研发工程师",
        "location": "北京",
        "publish_date": "2017-08-31",
    }
    detail_fields = {
        "职位名称": "Android Multimedia系统工程师（小米电视）",
        "工作地点": "Peking",
        "职位类别": "研发工程师",
        "招聘渠道": "社会招聘",
        "工作职责": "1. 负责 Android Multimedia 系统开发。",
        "工作要求": "1. 本科及以上学历；2. 3年以上 Android 开发经验。",
    }
    converted = convert_xiaomi_job(row, detail_fields, crawl_time="2026-05-19 10:00:00", list_path="8-0-2")
    assert converted["id"] == "xiaomi_800"
    assert converted["company"] == "小米"
    assert converted["location"] == "北京"
    assert converted["meta"]["sft_ready"] is True
    assert converted["meta"]["list_path"] == "8-0-2"
    assert "岗位职责：" in converted["raw_text"]


def test_build_page_path() -> None:
    assert build_page_path("8-0-2", 1) == "8-0-2"
    assert build_page_path("8-0-2", 3) == "8-0-2-0-3"


def test_build_search_path() -> None:
    assert build_search_path("开发", 1) == "%E5%BC%80%E5%8F%91"
    assert build_search_path("开发", 2) == "%E5%BC%80%E5%8F%91-0-2"
