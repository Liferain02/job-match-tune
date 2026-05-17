from __future__ import annotations

from jobmatch_tune.crawler.baidu_talent import (
    build_search_url,
    convert_baidu_post,
    extract_initial_data_payload,
)


def test_extract_initial_data_payload() -> None:
    html = """
    <script>
    window.__INITIAL_DATA__ ={"listData":{"recruitType":"SOCIAL","listDetailData":[{"postId":"abc","name":"算法工程师","workContent":"负责算法开发","serviceCondition":"本科"}],"projectType":undefined}};
    window.prefix="/jobs";
    </script>
    """
    payload = extract_initial_data_payload(html)
    assert payload["listData"]["recruitType"] == "SOCIAL"
    assert payload["listData"]["projectType"] is None
    assert payload["listData"]["listDetailData"][0]["postId"] == "abc"


def test_convert_baidu_post() -> None:
    row = convert_baidu_post(
        {
            "postId": "abc-123",
            "jobId": "job-123",
            "name": "大模型算法工程师（J12345）",
            "postType": "技术",
            "publishDate": "2026-05-16",
            "updateDate": "2026-05-16",
            "recruitNum": "2",
            "serviceCondition": "-本科及以上\n-熟悉Python",
            "workContent": "-负责模型训练\n-负责效果优化",
            "workPlace": "北京市",
            "bgShortName": "ACG",
            "hotFlag": True,
        },
        keyword="大模型",
        crawl_time="2026-05-16 10:00:00",
        recruit_type="SOCIAL",
    )
    assert row["id"] == "baidu_abc-123"
    assert row["company"] == "百度"
    assert row["url"].endswith("/SOCIAL/abc-123")
    assert "岗位职责：" in row["raw_text"]
    assert row["meta"]["keyword"] == "大模型"
    assert row["meta"]["sft_ready"] is True


def test_build_search_url() -> None:
    assert build_search_url("") == "https://talent.baidu.com/jobs/social-list"
    assert "search=%E5%A4%A7%E6%A8%A1%E5%9E%8B" in build_search_url("大模型")
