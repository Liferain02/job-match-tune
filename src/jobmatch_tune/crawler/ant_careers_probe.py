from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


ANT_BASE_URL = "https://hrcareersweb.antgroup.com"
SEARCH_CONDITION_LIST_URL = f"{ANT_BASE_URL}/api/searchCondition/list"
SEARCH_CONDITION_GROUP_URL = f"{ANT_BASE_URL}/api/searchCondition/listPositionGroup"
SEARCH_CONDITION_TALENT_PLAN_URL = f"{ANT_BASE_URL}/api/searchCondition/listTalentPlan"
SOCIAL_SEARCH_URL = f"{ANT_BASE_URL}/api/social/position/search"
POSITION_IDS_URL = f"{ANT_BASE_URL}/api/position/searchPositionIdsByQuery"


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
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


def build_social_search_variants(condition_payload: dict[str, Any]) -> list[dict[str, Any]]:
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
    ]
    return variants


def probe_social_search(session: requests.Session, condition_payload: dict[str, Any]) -> list[dict[str, Any]]:
    variants = build_social_search_variants(condition_payload)
    results: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        result = _post(session, SOCIAL_SEARCH_URL, variant["payload"])
        result["variant"] = idx
        result["variant_name"] = variant["name"]
        results.append(result)
    return results


def probe_position_id_search(session: requests.Session) -> list[dict[str, Any]]:
    variants = [
        {"query": "研发", "pageNo": 1, "pageSize": 20},
        {"query": "技术类", "pageNo": 1, "pageSize": 20},
        {"query": "", "pageNo": 1, "pageSize": 20},
    ]
    results: list[dict[str, Any]] = []
    for idx, payload in enumerate(variants, start=1):
        result = _post(session, POSITION_IDS_URL, payload)
        result["variant"] = idx
        results.append(result)
    return results


def probe_ant_site(timeout: float = 20.0) -> dict[str, Any]:
    session = build_session(timeout)
    condition_report = probe_conditions(session)
    condition_payload = condition_report["list"].get("json") or {}
    return {
        "base_url": ANT_BASE_URL,
        "conditions": condition_report,
        "social_search_probes": probe_social_search(session, condition_payload),
        "position_id_search_probes": probe_position_id_search(session),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    report = probe_ant_site(timeout=args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote probe report: {out_path}")


if __name__ == "__main__":
    main()
