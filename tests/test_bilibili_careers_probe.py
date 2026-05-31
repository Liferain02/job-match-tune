from __future__ import annotations

from jobmatch_tune.crawler.bilibili_careers_probe import (
    extract_api_paths,
    extract_endpoint_snippets,
    extract_script_urls,
    select_candidate_bundle_urls,
    select_probe_paths,
)


def test_extract_script_urls_normalizes_sources() -> None:
    html = """
    <html><head>
    <script src="//s1.hdslb.com/a.js"></script>
    <script src="/b.js"></script>
    </head></html>
    """
    urls = extract_script_urls(html)
    assert urls == [
        "https://s1.hdslb.com/a.js",
        "https://jobs.bilibili.com/b.js",
    ]


def test_extract_api_paths_returns_unique_values() -> None:
    text = '"/api/user/info" "/api/login/exit/v2" "/api/user/info"'
    assert extract_api_paths(text) == ["/api/user/info", "/api/login/exit/v2"]


def test_select_candidate_bundle_urls_prefers_main_assets() -> None:
    urls = [
        "https://s1.hdslb.com/a.js",
        "https://s1.hdslb.com/bfs/static/zhaopin-toc/assets/js/chunk-vendors.1.js",
        "https://s1.hdslb.com/bfs/static/zhaopin-toc/assets/js/app.2.js",
    ]
    selected = select_candidate_bundle_urls(urls)
    assert selected[:2] == [
        "https://s1.hdslb.com/bfs/static/zhaopin-toc/assets/js/chunk-vendors.1.js",
        "https://s1.hdslb.com/bfs/static/zhaopin-toc/assets/js/app.2.js",
    ]


def test_select_probe_paths_keeps_relevant_api_paths() -> None:
    paths = [
        "/api/login/exit/v2",
        "/api/user/info",
        "/api/sns/web",
        "/api/resume/analysis",
    ]
    assert select_probe_paths(paths) == [
        "/api/resume/analysis",
        "/api/login/exit/v2",
        "/api/user/info",
    ]


def test_extract_endpoint_snippets_returns_context() -> None:
    text = 'aaa "/api/resume/analysis" bbb'
    snippets = extract_endpoint_snippets(text, "/api/resume/analysis", radius=4)
    assert snippets
    assert "/api/resume/analysis" in snippets[0]
