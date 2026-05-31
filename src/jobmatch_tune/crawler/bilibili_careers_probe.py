from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests


BILIBILI_JOBS_BASE_URL = "https://jobs.bilibili.com"
SCRIPT_SRC_RE = re.compile(r'<script[^>]+src="([^"]+)"', flags=re.IGNORECASE)
API_PATH_RE = re.compile(r"/api/[A-Za-z0-9_./?=&-]+")


def _with_timeout(request_fn: Any, timeout: float) -> Any:
    def wrapped(method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", timeout)
        return request_fn(method, url, **kwargs)

    return wrapped


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/json,text/plain,*/*",
        }
    )
    session.request = _with_timeout(session.request, timeout)
    return session


def fetch_text(session: requests.Session, url: str) -> str:
    response = session.get(url)
    response.raise_for_status()
    return response.text


def extract_script_urls(html: str) -> list[str]:
    urls: list[str] = []
    for match in SCRIPT_SRC_RE.finditer(html):
        src = match.group(1).strip()
        if not src:
            continue
        if src.startswith("//"):
            src = f"https:{src}"
        elif src.startswith("/"):
            src = f"{BILIBILI_JOBS_BASE_URL}{src}"
        if src not in urls:
            urls.append(src)
    return urls


def extract_api_paths(text: str) -> list[str]:
    paths: list[str] = []
    for path in API_PATH_RE.findall(text):
        if path not in paths:
            paths.append(path)
    return paths


def extract_endpoint_snippets(text: str, endpoint: str, radius: int = 180) -> list[str]:
    snippets: list[str] = []
    start = 0
    while True:
        idx = text.find(endpoint, start)
        if idx < 0:
            break
        left = max(0, idx - radius)
        right = min(len(text), idx + len(endpoint) + radius)
        snippet = text[left:right]
        if snippet not in snippets:
            snippets.append(snippet)
        start = idx + len(endpoint)
    return snippets


def select_candidate_bundle_urls(script_urls: list[str]) -> list[str]:
    preferred = [url for url in script_urls if "/assets/js/app." in url or "/assets/js/chunk-vendors." in url]
    fallback = [url for url in script_urls if url not in preferred]
    return preferred + fallback


def select_probe_paths(api_paths: list[str]) -> list[str]:
    priority_groups = [
        ("position", "deliver", "analysis"),
        ("resume", "record"),
        ("user", "login", "token"),
    ]
    selected: list[str] = []
    lower_paths = [(path, path.lower()) for path in api_paths]
    for group in priority_groups:
        for path, lower in lower_paths:
            if path in selected:
                continue
            if any(token in lower for token in group):
                selected.append(path)
            if len(selected) >= 8:
                return selected
    return selected[:8]


def probe_candidate_endpoint(
    session: requests.Session,
    path: str,
    *,
    method: str = "GET",
    post_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{BILIBILI_JOBS_BASE_URL}{path}"
    if method == "POST":
        response = session.post(
            url,
            json=post_payload or {},
            headers={"Content-Type": "application/json", "Origin": BILIBILI_JOBS_BASE_URL, "Referer": f"{BILIBILI_JOBS_BASE_URL}/social"},
        )
    else:
        response = session.get(url)
    result: dict[str, Any] = {
        "method": method,
        "url": url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
    }
    try:
        result["json"] = response.json()
    except ValueError:
        result["json"] = None
        result["text_preview"] = response.text[:300]
    return result


def probe_endpoint_variants(session: requests.Session, api_paths: list[str]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for path in select_probe_paths(api_paths):
        key = path.strip("/").replace("/", "_")
        variants[f"{key}_get"] = probe_candidate_endpoint(session, path, method="GET")
        lowered = path.lower()
        if any(token in lowered for token in ("position", "resume", "record", "analysis", "deliver")):
            variants[f"{key}_post"] = probe_candidate_endpoint(session, path, method="POST", post_payload={})
    return variants


def probe_bilibili_careers(timeout: float = 20.0) -> dict[str, Any]:
    session = build_session(timeout)
    try:
        html = fetch_text(session, f"{BILIBILI_JOBS_BASE_URL}/social")
    except requests.RequestException as exc:
        return {
            "base_url": BILIBILI_JOBS_BASE_URL,
            "error": str(exc),
            "script_url_count": 0,
            "script_urls": [],
            "candidate_bundle_urls": [],
            "bundle_api_path_count": 0,
            "bundle_api_paths": [],
            "bundle_text_preview_by_url": {},
            "endpoint_snippets": {},
            "endpoint_probes": {},
        }
    script_urls = extract_script_urls(html)
    candidate_bundle_urls = select_candidate_bundle_urls(script_urls)
    bundle_api_paths: list[str] = []
    bundle_text_preview_by_url: dict[str, str] = {}
    endpoint_snippets: dict[str, list[str]] = {}
    for url in candidate_bundle_urls[:4]:
        try:
            bundle_text = fetch_text(session, url)
        except requests.RequestException as exc:
            bundle_text_preview_by_url[url] = str(exc)
            continue
        bundle_text_preview_by_url[url] = bundle_text[:300]
        for path in extract_api_paths(bundle_text):
            if path not in bundle_api_paths:
                bundle_api_paths.append(path)
        for endpoint in select_probe_paths(bundle_api_paths):
            snippets = extract_endpoint_snippets(bundle_text, endpoint)
            if snippets:
                endpoint_snippets[endpoint] = snippets[:3]
    return {
        "base_url": BILIBILI_JOBS_BASE_URL,
        "script_url_count": len(script_urls),
        "script_urls": script_urls[:20],
        "candidate_bundle_urls": candidate_bundle_urls[:10],
        "bundle_api_path_count": len(bundle_api_paths),
        "bundle_api_paths": bundle_api_paths[:50],
        "bundle_text_preview_by_url": bundle_text_preview_by_url,
        "endpoint_snippets": endpoint_snippets,
        "endpoint_probes": probe_endpoint_variants(session, bundle_api_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    report = probe_bilibili_careers(timeout=args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote probe report: {out_path}")


if __name__ == "__main__":
    main()
