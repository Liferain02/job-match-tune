from __future__ import annotations

from jobmatch_tune.crawler.xiaohongshu_careers_probe import (
    extract_endpoint_snippets,
    extract_api_paths,
    extract_script_urls,
    select_candidate_bundle_urls,
)


def test_extract_script_urls_normalizes_sources() -> None:
    html = """
    <html><head>
    <script src="//fe-static.xhscdn.com/a.js"></script>
    <script src="/b.js"></script>
    </head></html>
    """
    urls = extract_script_urls(html)
    assert urls == [
        "https://fe-static.xhscdn.com/a.js",
        "https://jobs.xiaohongshu.com/b.js",
    ]


def test_extract_api_paths_returns_unique_values() -> None:
    text = '"/api/store/jpd/main" "/api/sns/web" "/api/store/jpd/main"'
    assert extract_api_paths(text) == ["/api/store/jpd/main", "/api/sns/web"]


def test_select_candidate_bundle_urls_prefers_main_assets() -> None:
    urls = [
        "https://fe-static.xhscdn.com/a.js",
        "https://fe-static.xhscdn.com/runtime-main.123.js",
        "https://fe-static.xhscdn.com/main.456.js",
    ]
    selected = select_candidate_bundle_urls(urls)
    assert selected[:2] == [
        "https://fe-static.xhscdn.com/runtime-main.123.js",
        "https://fe-static.xhscdn.com/main.456.js",
    ]


def test_extract_endpoint_snippets_returns_context() -> None:
    text = 'aaa "/api/store/jpd/main" bbb "/api/store/jpd/main" ccc'
    snippets = extract_endpoint_snippets(text, "/api/store/jpd/main", radius=4)
    assert snippets
    assert "/api/store/jpd/main" in snippets[0]
