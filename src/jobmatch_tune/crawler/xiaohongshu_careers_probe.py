from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests


XHS_JOBS_BASE_URL = "https://jobs.xiaohongshu.com"
SCRIPT_SRC_RE = re.compile(r'<script[^>]+src="([^"]+)"', flags=re.IGNORECASE)
API_PATH_RE = re.compile(r"/api/[A-Za-z0-9_./?=&-]+")


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
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


def _with_timeout(request_fn: Any, timeout: float) -> Any:
    def wrapped(method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", timeout)
        return request_fn(method, url, **kwargs)

    return wrapped


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url)
    response.raise_for_status()
    return response.text


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
            src = f"{XHS_JOBS_BASE_URL}{src}"
        if src not in urls:
            urls.append(src)
    return urls


def extract_api_paths(text: str) -> list[str]:
    paths: list[str] = []
    for path in API_PATH_RE.findall(text):
        if path not in paths:
            paths.append(path)
    return paths


def select_candidate_bundle_urls(script_urls: list[str]) -> list[str]:
    preferred = [url for url in script_urls if "/main." in url or "/runtime-main." in url]
    fallback = [url for url in script_urls if url not in preferred]
    return preferred + fallback


def probe_candidate_endpoint(
    session: requests.Session,
    path: str,
    *,
    post_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{XHS_JOBS_BASE_URL}{path}"
    if post_payload is None:
        response = session.get(url)
    else:
        response = session.post(
            url,
            json=post_payload,
            headers={"Content-Type": "application/json", "Origin": XHS_JOBS_BASE_URL, "Referer": f"{XHS_JOBS_BASE_URL}/"},
        )
    result: dict[str, Any] = {
        "url": url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "payload": post_payload,
    }
    try:
        result["json"] = response.json()
    except ValueError:
        result["json"] = None
        result["text_preview"] = response.text[:300]
    return result


def probe_xiaohongshu_careers(timeout: float = 20.0) -> dict[str, Any]:
    session = build_session(timeout)
    html = fetch_html(session, f"{XHS_JOBS_BASE_URL}/")
    script_urls = extract_script_urls(html)
    candidate_bundle_urls = select_candidate_bundle_urls(script_urls)
    bundle_api_paths: list[str] = []
    bundle_text_preview_by_url: dict[str, str] = {}
    for url in candidate_bundle_urls[:3]:
        try:
            bundle_text = fetch_text(session, url)
        except requests.RequestException as exc:
            bundle_text_preview_by_url[url] = str(exc)
            continue
        bundle_text_preview_by_url[url] = bundle_text[:300]
        for path in extract_api_paths(bundle_text):
            if path not in bundle_api_paths:
                bundle_api_paths.append(path)
    return {
        "base_url": XHS_JOBS_BASE_URL,
        "script_url_count": len(script_urls),
        "script_urls": script_urls[:20],
        "candidate_bundle_urls": candidate_bundle_urls[:10],
        "bundle_api_path_count": len(bundle_api_paths),
        "bundle_api_paths": bundle_api_paths[:50],
        "bundle_text_preview_by_url": bundle_text_preview_by_url,
        "store_jpd_main_get": probe_candidate_endpoint(session, "/api/store/jpd/main"),
        "store_jpd_main_post": probe_candidate_endpoint(session, "/api/store/jpd/main", post_payload={}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    report = probe_xiaohongshu_careers(timeout=args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote probe report: {out_path}")


if __name__ == "__main__":
    main()
