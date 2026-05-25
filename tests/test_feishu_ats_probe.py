from __future__ import annotations

from jobmatch_tune.crawler.feishu_ats_probe import extract_website_info


def test_extract_website_info_returns_payload() -> None:
    html = """
    <html><head></head><body>
    <script id="js-websiteInfo" type="text/json">
    {"tenant_info":{"tenant_name":"测试公司"},"website_info":{"path":"index","language":"zh-CN"}}
    </script>
    </body></html>
    """
    payload = extract_website_info(html)
    assert payload["tenant_info"]["tenant_name"] == "测试公司"
    assert payload["website_info"]["path"] == "index"


def test_extract_website_info_handles_missing_block() -> None:
    assert extract_website_info("<html></html>") == {}
