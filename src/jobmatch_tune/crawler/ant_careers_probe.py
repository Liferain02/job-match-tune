from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests


ANT_BASE_URL = "https://hrcareersweb.antgroup.com"
ANT_TALENT_BASE_URL = "https://talent.antgroup.com"
SEARCH_CONDITION_LIST_URL = f"{ANT_BASE_URL}/api/searchCondition/list"
SEARCH_CONDITION_GROUP_URL = f"{ANT_BASE_URL}/api/searchCondition/listPositionGroup"
SEARCH_CONDITION_TALENT_PLAN_URL = f"{ANT_BASE_URL}/api/searchCondition/listTalentPlan"
SOCIAL_SEARCH_URL = f"{ANT_BASE_URL}/api/social/position/search"
POSITION_IDS_URL = f"{ANT_BASE_URL}/api/position/searchPositionIdsByQuery"
SCRIPT_SRC_RE = re.compile(r'<script[^>]+src="([^"]+)"', flags=re.IGNORECASE)
TERN_CONFIG_RE = re.compile(
    r'<script[^>]+type="tern-(?:site|app)-config"[^>]*>(.*?)</script>',
    flags=re.IGNORECASE | re.DOTALL,
)
TALENT_ROUTE_RE = re.compile(r"https://talent\.antgroup\.com(/[A-Za-z0-9\-_/]+)")
SOCIAL_CHUNK_RE = re.compile(r"p__SocialRecruitment__[A-Za-z0-9_]+")
ENTRY_PATHS = ["/social-recruitment", "/social-recruit", "/social", "/"]
TALENT_ENTRY_PATHS = ["/off-campus", "/campus", "/"]


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Content-Type": "application/json;charset=UTF-8",
            "Origin": ANT_BASE_URL,
            "Referer": f"{ANT_BASE_URL}/social-recruitment",
        }
    )
    session.request = _with_timeout(session.request, timeout)
    return session


def _with_timeout(request_fn: Any, timeout: float) -> Any:
    def wrapped(method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", timeout)
        return request_fn(method, url, **kwargs)

    return wrapped


def _extract_search_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    content = payload.get("content") or {}
    if not isinstance(content, dict):
        return []
    search_items = content.get("searchItems") or []
    if not isinstance(search_items, list):
        return []
    return [item for item in search_items if isinstance(item, dict)]


def _extract_id_list(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    content = payload.get("content") or []
    if not isinstance(content, list):
        return []
    values: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        value = item.get("id")
        if value is None:
            continue
        string_value = str(value)
        if string_value not in values:
            values.append(string_value)
    return values


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
            src = f"{ANT_BASE_URL}{src}"
        if src not in urls:
            urls.append(src)
    return urls


def extract_tern_site_config(html: str) -> dict[str, Any] | None:
    match = TERN_CONFIG_RE.search(html)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw[:1000]}
    return parsed if isinstance(parsed, dict) else {"raw": raw[:1000]}


def extract_talent_route_hints(html: str) -> list[str]:
    routes: list[str] = []
    for match in TALENT_ROUTE_RE.finditer(html):
        route = match.group(1)
        if route not in routes:
            routes.append(route)
    return routes


def extract_social_chunk_hints(html: str) -> list[str]:
    hints: list[str] = []
    for match in SOCIAL_CHUNK_RE.finditer(html):
        hint = match.group(0)
        if hint not in hints:
            hints.append(hint)
    return hints


def select_candidate_bundle_urls(script_urls: list[str]) -> list[str]:
    preferred = [
        url for url in script_urls if "index" in url or "main" in url or "app" in url or "umi" in url
    ]
    fallback = [url for url in script_urls if url not in preferred]
    return preferred + fallback


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


def _post(session: requests.Session, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = session.post(url, json=payload)
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
        return result
    result["json"] = parsed
    result["ok"] = isinstance(parsed, dict) and parsed.get("success") is True
    return result


def probe_conditions(session: requests.Session) -> dict[str, Any]:
    list_result = _post(session, SEARCH_CONDITION_LIST_URL, {})
    group_result = _post(session, SEARCH_CONDITION_GROUP_URL, {})
    talent_plan_result = _post(session, SEARCH_CONDITION_TALENT_PLAN_URL, {})
    search_items = _extract_search_items(list_result.get("json"))
    total_positions = ((list_result.get("json") or {}).get("content") or {}).get("totalPositions")
    return {
        "list": list_result,
        "listPositionGroup": group_result,
        "listTalentPlan": talent_plan_result,
        "search_item_count": len(search_items),
        "total_positions": total_positions,
    }


def build_social_search_variants(
    condition_payload: dict[str, Any],
    *,
    position_group_payload: dict[str, Any] | None = None,
    talent_plan_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    search_items = _extract_search_items(condition_payload)
    category_value = ""
    work_city_value = ""
    dept_value = ""
    position_group_ids = _extract_id_list(position_group_payload)
    talent_plan_ids = _extract_id_list(talent_plan_payload)
    for item in search_items:
        item_type = str(item.get("type") or "")
        choices = item.get("items") or []
        if not isinstance(choices, list) or not choices:
            continue
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            continue
        value = first_choice.get("value")
        if item_type == "category" and not category_value:
            category_value = str(value or "")
        elif item_type == "workCity" and not work_city_value:
            work_city_value = str(value or "")
        elif item_type == "dept" and not dept_value:
            dept_value = str(value or "")
    variants = [
        {"name": "empty", "payload": {}},
        {
            "name": "minimal_page",
            "payload": {"keyword": "", "currentPage": 1, "pageSize": 20},
        },
        {
            "name": "array_filters_empty",
            "payload": {
                "keyword": "",
                "currentPage": 1,
                "pageSize": 20,
                "category": [category_value] if category_value else [],
                "workCity": [],
                "dept": [],
                "recruitType": [],
            },
        },
        {
            "name": "scalar_filters_empty_recruit",
            "payload": {
                "keyword": "",
                "currentPage": 1,
                "pageSize": 20,
                "category": category_value,
                "workCity": work_city_value,
                "dept": dept_value,
                "recruitType": "",
            },
        },
        {
            "name": "scalar_filters_social_recruit",
            "payload": {
                "language": "zh_CN",
                "keyword": "",
                "currentPage": 1,
                "pageSize": 20,
                "category": category_value,
                "workCity": work_city_value,
                "dept": dept_value,
                "recruitType": "social_recruit",
            },
        },
        {
            "name": "array_filters_social_recruit",
            "payload": {
                "language": "zh_CN",
                "keyword": "",
                "currentPage": 1,
                "pageSize": 20,
                "category": [category_value] if category_value else [],
                "workCity": [work_city_value] if work_city_value else [],
                "dept": [dept_value] if dept_value else [],
                "recruitType": ["social_recruit"],
            },
        },
        {
            "name": "list_suffix_social_recruit",
            "payload": {
                "language": "zh_CN",
                "keyword": "",
                "currentPage": 1,
                "pageSize": 20,
                "categoryList": [category_value] if category_value else [],
                "workCityList": [work_city_value] if work_city_value else [],
                "deptList": [dept_value] if dept_value else [],
                "recruitTypeList": ["social_recruit"],
            },
        },
        {
            "name": "ids_suffix_social_recruit",
            "payload": {
                "language": "zh_CN",
                "keyword": "",
                "currentPage": 1,
                "pageSize": 20,
                "categoryIds": [category_value] if category_value else [],
                "workCityCodes": [work_city_value] if work_city_value else [],
                "deptIds": [dept_value] if dept_value else [],
                "recruitType": "social_recruit",
            },
        },
        {
            "name": "position_group_social_recruit",
            "payload": {
                "language": "zh_CN",
                "keyword": "",
                "currentPage": 1,
                "pageSize": 20,
                "category": [category_value] if category_value else [],
                "workCity": [work_city_value] if work_city_value else [],
                "dept": [dept_value] if dept_value else [],
                "recruitType": ["social_recruit"],
                "positionGroupId": position_group_ids[0] if position_group_ids else "",
            },
        },
        {
            "name": "talent_plan_social_recruit",
            "payload": {
                "language": "zh_CN",
                "keyword": "",
                "currentPage": 1,
                "pageSize": 20,
                "category": [category_value] if category_value else [],
                "workCity": [work_city_value] if work_city_value else [],
                "dept": [dept_value] if dept_value else [],
                "recruitType": ["social_recruit"],
                "talentPlanId": talent_plan_ids[0] if talent_plan_ids else "",
            },
        },
    ]
    return variants


def build_position_id_search_variants(condition_payload: dict[str, Any]) -> list[dict[str, Any]]:
    search_items = _extract_search_items(condition_payload)
    category_value = ""
    work_city_value = ""
    dept_value = ""
    for item in search_items:
        item_type = str(item.get("type") or "")
        choices = item.get("items") or []
        if not isinstance(choices, list) or not choices:
            continue
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            continue
        value = str(first_choice.get("value") or "")
        if item_type == "category" and not category_value:
            category_value = value
        elif item_type == "workCity" and not work_city_value:
            work_city_value = value
        elif item_type == "dept" and not dept_value:
            dept_value = value
    return [
        {"name": "query_rd", "payload": {"query": "研发", "pageNo": 1, "pageSize": 20}},
        {"name": "query_tech", "payload": {"query": "技术类", "pageNo": 1, "pageSize": 20}},
        {"name": "query_empty", "payload": {"query": "", "pageNo": 1, "pageSize": 20}},
        {
            "name": "category_only",
            "payload": {"query": "", "category": category_value, "pageNo": 1, "pageSize": 20},
        },
        {
            "name": "dept_only",
            "payload": {"query": "", "dept": dept_value, "pageNo": 1, "pageSize": 20},
        },
        {
            "name": "work_city_only",
            "payload": {"query": "", "workCity": work_city_value, "pageNo": 1, "pageSize": 20},
        },
        {
            "name": "combined_scalar",
            "payload": {
                "query": "",
                "category": category_value,
                "dept": dept_value,
                "workCity": work_city_value,
                "pageNo": 1,
                "pageSize": 20,
            },
        },
    ]


def probe_social_search(
    session: requests.Session,
    condition_payload: dict[str, Any],
    *,
    position_group_payload: dict[str, Any] | None = None,
    talent_plan_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    variants = build_social_search_variants(
        condition_payload,
        position_group_payload=position_group_payload,
        talent_plan_payload=talent_plan_payload,
    )
    results: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        result = _post(session, SOCIAL_SEARCH_URL, variant["payload"])
        result["variant"] = idx
        result["variant_name"] = variant["name"]
        results.append(result)
    return results


def probe_position_id_search(
    session: requests.Session,
    condition_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    variants = build_position_id_search_variants(condition_payload)
    results: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        result = _post(session, POSITION_IDS_URL, variant["payload"])
        result["variant"] = idx
        result["variant_name"] = variant["name"]
        results.append(result)
    return results


def probe_ant_site(timeout: float = 20.0, *, include_api: bool = True) -> dict[str, Any]:
    session = build_session(timeout)
    entry_report: dict[str, Any] = {"entry_url": None, "status": "unavailable", "errors": []}
    html = ""
    for path in ENTRY_PATHS:
        entry_url = f"{ANT_BASE_URL}{path}"
        try:
            html = fetch_html(session, entry_url)
        except requests.RequestException as exc:
            entry_report["errors"].append({"url": entry_url, "error": str(exc)})
            continue
        entry_report["entry_url"] = entry_url
        entry_report["status"] = "ok"
        break
    talent_entry_report: dict[str, Any] = {"entry_url": None, "status": "unavailable", "errors": []}
    talent_html = ""
    for path in TALENT_ENTRY_PATHS:
        entry_url = f"{ANT_TALENT_BASE_URL}{path}"
        try:
            talent_html = fetch_html(session, entry_url)
        except requests.RequestException as exc:
            talent_entry_report["errors"].append({"url": entry_url, "error": str(exc)})
            continue
        talent_entry_report["entry_url"] = entry_url
        talent_entry_report["status"] = "ok"
        break
    script_urls = extract_script_urls(html) if html else []
    candidate_bundle_urls = select_candidate_bundle_urls(script_urls) if script_urls else []
    talent_script_urls = extract_script_urls(talent_html) if talent_html else []
    talent_candidate_bundle_urls = select_candidate_bundle_urls(talent_script_urls) if talent_script_urls else []
    talent_tern_config = extract_tern_site_config(talent_html) if talent_html else None
    social_search_snippets: dict[str, list[str]] = {}
    position_id_snippets: dict[str, list[str]] = {}
    for url in candidate_bundle_urls[:4]:
        try:
            bundle_text = fetch_text(session, url)
        except requests.RequestException as exc:
            social_search_snippets[url] = [str(exc)]
            position_id_snippets[url] = [str(exc)]
            continue
        social_hits = extract_endpoint_snippets(bundle_text, "/api/social/position/search")
        position_hits = extract_endpoint_snippets(bundle_text, "/api/position/searchPositionIdsByQuery")
        if social_hits:
            social_search_snippets[url] = social_hits[:3]
        if position_hits:
            position_id_snippets[url] = position_hits[:3]
    report = {
        "base_url": ANT_BASE_URL,
        "entry_probe": entry_report,
        "script_url_count": len(script_urls),
        "script_urls": script_urls[:20],
        "candidate_bundle_urls": candidate_bundle_urls[:10],
        "talent_base_url": ANT_TALENT_BASE_URL,
        "talent_entry_probe": talent_entry_report,
        "talent_script_url_count": len(talent_script_urls),
        "talent_script_urls": talent_script_urls[:20],
        "talent_candidate_bundle_urls": talent_candidate_bundle_urls[:10],
        "talent_tern_site_config": talent_tern_config,
        "talent_route_hints": extract_talent_route_hints(talent_html)[:20] if talent_html else [],
        "talent_social_chunk_hints": extract_social_chunk_hints(talent_html)[:20] if talent_html else [],
    }
    if not include_api:
        report["api_probe_skipped"] = True
        return report
    condition_report = probe_conditions(session)
    condition_payload = condition_report["list"].get("json") or {}
    position_group_payload = condition_report["listPositionGroup"].get("json") or {}
    talent_plan_payload = condition_report["listTalentPlan"].get("json") or {}
    report.update(
        {
            "social_search_snippets": social_search_snippets,
            "position_id_search_snippets": position_id_snippets,
            "conditions": condition_report,
            "social_search_probes": probe_social_search(
                session,
                condition_payload,
                position_group_payload=position_group_payload,
                talent_plan_payload=talent_plan_payload,
            ),
            "position_id_search_probes": probe_position_id_search(session, condition_payload),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--page-only", action="store_true")
    args = parser.parse_args()

    report = probe_ant_site(timeout=args.timeout, include_api=not args.page_only)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote probe report: {out_path}")


if __name__ == "__main__":
    main()
