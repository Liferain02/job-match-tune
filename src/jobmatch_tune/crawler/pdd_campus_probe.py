from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests


PDD_CAMPUS_BASE_URL = "https://careers.pinduoduo.com"
NEXT_DATA_RE = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
    flags=re.IGNORECASE | re.DOTALL,
)
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


def extract_next_data(html: str) -> dict[str, Any]:
    match = NEXT_DATA_RE.search(html)
    if not match:
        return {}
    raw = match.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_script_urls(html: str) -> list[str]:
    urls: list[str] = []
    for match in SCRIPT_SRC_RE.finditer(html):
        src = match.group(1).strip()
        if not src:
            continue
        if src.startswith("//"):
            src = f"https:{src}"
        elif src.startswith("/"):
            src = f"{PDD_CAMPUS_BASE_URL}{src}"
        if src not in urls:
            urls.append(src)
    return urls


def extract_api_paths(text: str) -> list[str]:
    seen: list[str] = []
    for path in API_PATH_RE.findall(text):
        if path not in seen:
            seen.append(path)
    return seen


def select_candidate_bundle_urls(script_urls: list[str]) -> list[str]:
    preferred = [
        url
        for url in script_urls
        if "/pages/index-" in url or "/chunks/main-" in url or "/pages/_app-" in url
    ]
    fallback = [url for url in script_urls if "_next/static/" in url and url not in preferred]
    return preferred + fallback


def summarize_next_data(payload: dict[str, Any]) -> dict[str, Any]:
    page = payload.get("page")
    build_id = payload.get("buildId")
    props = payload.get("props") or {}
    page_props = props.get("pageProps") or {}
    keys = sorted(page_props.keys()) if isinstance(page_props, dict) else []
    return {
        "page": page,
        "build_id": build_id,
        "page_prop_keys": keys,
    }


def probe_api_root(session: requests.Session) -> dict[str, Any]:
    url = f"{PDD_CAMPUS_BASE_URL}/api/"
    response = session.get(url)
    result: dict[str, Any] = {
        "url": url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
    }
    try:
        payload = response.json()
    except ValueError:
        result["json"] = None
        result["text_preview"] = response.text[:300]
        return result
    result["json"] = payload
    return result


def probe_pdd_campus(timeout: float = 20.0) -> dict[str, Any]:
    session = build_session(timeout)
    html = fetch_html(session, f"{PDD_CAMPUS_BASE_URL}/")
    next_data = extract_next_data(html)
    script_urls = extract_script_urls(html)
    bundle_urls = [url for url in script_urls if "_next/static/" in url]
    candidate_bundle_urls = select_candidate_bundle_urls(script_urls)
    bundle_api_paths: list[str] = []
    bundle_text_preview_by_url: dict[str, str] = {}
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
    return {
        "base_url": PDD_CAMPUS_BASE_URL,
        "next_data_summary": summarize_next_data(next_data),
        "script_url_count": len(script_urls),
        "script_urls": script_urls[:20],
        "bundle_url_count": len(bundle_urls),
        "bundle_urls": bundle_urls[:10],
        "candidate_bundle_urls": candidate_bundle_urls[:10],
        "bundle_api_path_count": len(bundle_api_paths),
        "bundle_api_paths": bundle_api_paths[:50],
        "bundle_text_preview_by_url": bundle_text_preview_by_url,
        "api_root_probe": probe_api_root(session),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    report = probe_pdd_campus(timeout=args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote probe report: {out_path}")


if __name__ == "__main__":
    main()
