from __future__ import annotations

from jobmatch_tune.crawler.pdd_campus_probe import (
    extract_api_paths,
    extract_next_data,
    extract_script_urls,
    select_candidate_bundle_urls,
    summarize_next_data,
)


def test_extract_next_data_returns_payload() -> None:
    html = """
    <html><body>
    <script id="__NEXT_DATA__" type="application/json">
    {"page":"/","buildId":"abc123","props":{"pageProps":{"banners":[],"siteName":"拼多多校园招聘"}}}
    </script>
    </body></html>
    """
    payload = extract_next_data(html)
    assert payload["page"] == "/"
    assert payload["buildId"] == "abc123"


def test_extract_script_urls_normalizes_relative_paths() -> None:
    html = """
    <html><head>
    <script src="/_next/static/chunks/a.js"></script>
    <script src="//cdn.example.com/b.js"></script>
    </head></html>
    """
    urls = extract_script_urls(html)
    assert urls == [
        "https://careers.pinduoduo.com/_next/static/chunks/a.js",
        "https://cdn.example.com/b.js",
    ]


def test_extract_api_paths_returns_unique_values() -> None:
    text = '"/api/campus/moment/list" x "/api/campus/moment/detail" y "/api/campus/moment/list"'
    paths = extract_api_paths(text)
    assert paths == ["/api/campus/moment/list", "/api/campus/moment/detail"]


def test_summarize_next_data_extracts_keys() -> None:
    payload = {
        "page": "/",
        "buildId": "build-1",
        "props": {"pageProps": {"banners": [], "moments": []}},
    }
    summary = summarize_next_data(payload)
    assert summary["page"] == "/"
    assert summary["build_id"] == "build-1"
    assert summary["page_prop_keys"] == ["banners", "moments"]


def test_select_candidate_bundle_urls_prefers_page_and_main() -> None:
    urls = [
        "https://example.com/_next/static/chunks/framework.js",
        "https://example.com/_next/static/chunks/main-123.js",
        "https://example.com/_next/static/chunks/pages/index-456.js",
        "https://example.com/_next/static/chunks/pages/_app-789.js",
        "https://example.com/_next/static/build/_ssgManifest.js",
    ]
    selected = select_candidate_bundle_urls(urls)
    assert selected[:3] == [
        "https://example.com/_next/static/chunks/main-123.js",
        "https://example.com/_next/static/chunks/pages/index-456.js",
        "https://example.com/_next/static/chunks/pages/_app-789.js",
    ]
