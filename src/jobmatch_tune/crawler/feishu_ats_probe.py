from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests


WEBSITE_INFO_RE = re.compile(
    r'<script[^>]+id="js-websiteInfo"[^>]*>(.*?)</script>',
    flags=re.IGNORECASE | re.DOTALL,
)
SCRIPT_SRC_RE = re.compile(r'<script[^>]+src="([^"]+)"', flags=re.IGNORECASE)
API_PATH_RE = re.compile(r"/api/[A-Za-z0-9_./?=&-]+")
BYTEDANCE_HUNTER_MARKER = "字节跳动猎头平台"


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
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


def extract_website_info(html: str) -> dict[str, Any]:
    match = WEBSITE_INFO_RE.search(html)
    if not match:
        return {}
    raw = match.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def extract_script_urls(html: str) -> list[str]:
    urls: list[str] = []
    for match in SCRIPT_SRC_RE.finditer(html):
        src = match.group(1).strip()
        if src and src not in urls:
            urls.append(src)
    return urls


def extract_api_paths(text: str) -> list[str]:
    paths: list[str] = []
    for match in API_PATH_RE.findall(text):
        if match not in paths:
            paths.append(match)
    return paths


def select_candidate_bundle_urls(script_urls: list[str]) -> list[str]:
    preferred = [
        url
        for url in script_urls
        if "/index" in url or "/main" in url or "/app" in url or "chunk" in url
    ]
    fallback = [url for url in script_urls if url not in preferred]
    return preferred + fallback


def probe_filters(session: requests.Session, base_url: str, portal_type: int) -> dict[str, Any]:
    url = f"{base_url}/api/v1/config/job/filters/{portal_type}"
    response = session.get(url)
    result: dict[str, Any] = {
        "url": url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
    }
    try:
        payload = response.json()
    except ValueError:
        result["ok"] = False
        result["json"] = None
        result["text_preview"] = response.text[:300]
        return result
    result["json"] = payload
    result["ok"] = isinstance(payload, dict) and payload.get("code") == 0
    if result["ok"]:
        data = payload.get("data") or {}
        job_type_list = data.get("job_type_list") or []
        location_list = data.get("location_list") or []
        result["job_type_count"] = len(job_type_list)
        result["location_count"] = len(location_list)
    return result


def probe_detail(session: requests.Session, base_url: str, job_id: str = "1") -> dict[str, Any]:
    url = f"{base_url}/api/v1/job/posts/{job_id}"
    response = session.get(url)
    result: dict[str, Any] = {
        "url": url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
    }
    try:
        payload = response.json()
    except ValueError:
        result["ok"] = False
        result["json"] = None
        result["text_preview"] = response.text[:300]
        return result
    result["json"] = payload
    result["ok"] = isinstance(payload, dict) and payload.get("code") == 0
    return result


def probe_list(session: requests.Session, base_url: str, portal_type: int) -> dict[str, Any]:
    url = f"{base_url}/api/v1/search/job/posts"
    payload = {
        "keyword": "",
        "limit": 20,
        "offset": 0,
        "job_category_id_list": [],
        "tag_id_list": [],
        "location_code_list": [],
        "subject_id_list": [],
        "recruitment_id_list": [],
        "portal_type": portal_type,
        "job_function_id_list": [],
        "storefront_id_list": [],
    }
    response = session.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json;charset=UTF-8",
            "Origin": base_url,
            "Referer": f"{base_url}/index",
        },
    )
    result: dict[str, Any] = {
        "url": url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "payload": payload,
    }
    try:
        parsed = response.json()
    except ValueError:
        result["json"] = None
        result["text_preview"] = response.text[:300]
    else:
        result["json"] = parsed
    return result


def probe_list_get(
    session: requests.Session,
    base_url: str,
    *,
    params: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    url = f"{base_url}/api/v1/search/job/posts"
    response = session.get(
        url,
        params=params,
        headers={
            "Origin": base_url,
            "Referer": f"{base_url}/index",
        },
    )
    result: dict[str, Any] = {
        "label": label,
        "url": response.url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "params": params,
    }
    try:
        parsed = response.json()
    except ValueError:
        text = response.text
        result["json"] = None
        result["text_preview"] = text[:300]
        result["redirected_to_bytedance_hunter"] = BYTEDANCE_HUNTER_MARKER in text
    else:
        result["json"] = parsed
        result["redirected_to_bytedance_hunter"] = False
    return result


def probe_site(base_url: str, *, timeout: float = 20.0) -> dict[str, Any]:
    session = build_session(timeout)
    html = fetch_html(session, f"{base_url}/index")
    website_info = extract_website_info(html)
    script_urls = extract_script_urls(html)
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
    report: dict[str, Any] = {
        "base_url": base_url,
        "website_info": website_info,
        "script_url_count": len(script_urls),
        "script_urls": script_urls[:20],
        "candidate_bundle_urls": candidate_bundle_urls[:10],
        "bundle_api_path_count": len(bundle_api_paths),
        "bundle_api_paths": bundle_api_paths[:50],
        "bundle_text_preview_by_url": bundle_text_preview_by_url,
        "filters": [],
        "detail_probe": probe_detail(session, base_url),
    }
    for portal_type in range(1, 10):
        report["filters"].append(probe_filters(session, base_url, portal_type))
    report["list_probe"] = probe_list(session, base_url, portal_type=1)
    report["list_get_probes"] = [
        probe_list_get(
            session,
            base_url,
            params={"keyword": "", "limit": 10, "offset": 0, "portal_type": 1},
            label="minimal_get",
        ),
        probe_list_get(
            session,
            base_url,
            params={
                "keyword": "",
                "current": 1,
                "limit": 10,
                "portal_type": 1,
                "spread": "probe",
            },
            label="current_limit_get",
        ),
        probe_list_get(
            session,
            base_url,
            params={
                "keyword": "",
                "current": 1,
                "limit": 10,
                "portal_type": 1,
                "location": "CT_11",
                "spread": "probe",
            },
            label="with_location_get",
        ),
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    report = probe_site(args.base_url.rstrip("/"), timeout=args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote probe report: {out_path}")


if __name__ == "__main__":
    main()
