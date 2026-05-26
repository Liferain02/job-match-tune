from __future__ import annotations

from jobmatch_tune.crawler.feishu_ats_probe import (
    BYTEDANCE_HUNTER_MARKER,
    extract_api_paths,
    extract_script_urls,
    extract_website_info,
    select_candidate_bundle_urls,
)


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


def test_extract_script_urls_returns_unique_sources() -> None:
    html = """
    <html><head>
    <script src="https://example.com/a.js"></script>
    <script defer src="https://example.com/b.js"></script>
    <script src="https://example.com/a.js"></script>
    </head></html>
    """
    urls = extract_script_urls(html)
    assert urls == ["https://example.com/a.js", "https://example.com/b.js"]


def test_extract_api_paths_returns_unique_values() -> None:
    text = '"/api/v1/search/job/posts" x "/api/v1/job/posts/1" y "/api/v1/search/job/posts"'
    paths = extract_api_paths(text)
    assert paths == ["/api/v1/search/job/posts", "/api/v1/job/posts/1"]


def test_select_candidate_bundle_urls_prefers_index_main_app() -> None:
    urls = [
        "https://example.com/static/runtime.js",
        "https://example.com/static/main-1.js",
        "https://example.com/static/index-2.js",
        "https://example.com/static/app-3.js",
    ]
    selected = select_candidate_bundle_urls(urls)
    assert selected[:3] == [
        "https://example.com/static/main-1.js",
        "https://example.com/static/index-2.js",
        "https://example.com/static/app-3.js",
    ]


def test_bytedance_hunter_marker_constant() -> None:
    assert BYTEDANCE_HUNTER_MARKER == "字节跳动猎头平台"
