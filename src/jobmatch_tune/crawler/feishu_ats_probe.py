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


def probe_site(base_url: str, *, timeout: float = 20.0) -> dict[str, Any]:
    session = build_session(timeout)
    html = fetch_html(session, f"{base_url}/index")
    website_info = extract_website_info(html)
    script_urls = extract_script_urls(html)
    report: dict[str, Any] = {
        "base_url": base_url,
        "website_info": website_info,
        "script_url_count": len(script_urls),
        "script_urls": script_urls[:20],
        "filters": [],
        "detail_probe": probe_detail(session, base_url),
    }
    for portal_type in range(1, 10):
        report["filters"].append(probe_filters(session, base_url, portal_type))
    report["list_probe"] = probe_list(session, base_url, portal_type=1)
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
