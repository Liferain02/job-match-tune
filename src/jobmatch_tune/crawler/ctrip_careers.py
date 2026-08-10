from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from typing import Any

import requests

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


CTRIP_JOB_LIST_URL = "https://careers.ctrip.com/api/hrrecruit/getJobAd"

TECH_TITLE_KEYWORDS = [
    "开发",
    "研发",
    "算法",
    "测试",
    "前端",
    "后端",
    "客户端",
    "android",
    "ios",
    "java",
    "python",
    "go",
    "golang",
    "c++",
    "infra",
    "sre",
    "ai",
    "agent",
    "大模型",
    "数据",
    "平台",
    "架构",
    "sdk",
    "编译",
    "引擎",
    "安全",
    "运维",
    "机器学习",
    "深度学习",
    "云",
]

TECH_JOB_FAMILY_CODES = {
    "JFG_31",
    "JFG_32",
    "JFG_33",
    "JFG_34",
    "JFG_35",
}

PRODUCT_TITLE_KEYWORDS = {"产品经理", "平台产品", "ai产品", "agent产品"}


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
            "Content-Type": "application/json",
            "Origin": "https://careers.ctrip.com",
            "Referer": "https://careers.ctrip.com/",
            "X-Request-Timeout": str(timeout),
        }
    )
    session.request = _with_timeout(session.request, timeout)
    return session


def _with_timeout(request_fn: Any, timeout: float) -> Any:
    def wrapped(method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", timeout)
        return request_fn(method, url, **kwargs)

    return wrapped


def post_json(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    *,
    retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or data.get("retCode") != "201":
                raise RuntimeError(f"Unexpected Ctrip response: {data}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep_seconds * attempt)
    try:
        data = post_json_via_curl(url, payload, timeout=timeout_from_session(session))
        if not isinstance(data, dict) or data.get("retCode") != "201":
            raise RuntimeError(f"Unexpected Ctrip response via curl: {data}")
        return data
    except Exception as curl_error:
        assert last_error is not None
        raise last_error from curl_error


def timeout_from_session(session: requests.Session) -> float:
    return float(session.headers.get("X-Request-Timeout", "20"))


def post_json_via_curl(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    cmd = [
        "curl",
        "--compressed",
        "-L",
        "--max-time",
        str(int(timeout)),
        url,
        "-H",
        "Content-Type: application/json",
        "-H",
        "Origin: https://careers.ctrip.com",
        "-H",
        "Referer: https://careers.ctrip.com/",
        "--data-raw",
        payload_text,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def is_probably_tech_job(post: dict[str, Any]) -> bool:
    title = str(post.get("jobTitle") or "").strip().lower()
    family_code = str(post.get("jobFamilyGroupCode") or "").strip()
    family_name = str(post.get("jobFamilyGroupName") or "").strip().lower()
    combined = "\n".join(
        [
            title,
            family_name,
            str(post.get("requirements") or "").lower(),
            str(post.get("buName") or "").lower(),
        ]
    )
    if family_code in TECH_JOB_FAMILY_CODES:
        return True
    if family_code == "JFG_41":
        return any(keyword in title for keyword in PRODUCT_TITLE_KEYWORDS)
    return any(keyword in combined for keyword in TECH_TITLE_KEYWORDS)


def build_raw_text(post: dict[str, Any]) -> str:
    title = str(post.get("jobTitle") or "").strip()
    city_name = str(post.get("cityName") or "").strip()
    publish_date = str(post.get("publishDate") or "").strip()
    family_name = str(post.get("jobFamilyGroupName") or "").strip()
    family_code = str(post.get("jobFamilyGroupCode") or "").strip()
    bu_name = str(post.get("buName") or "").strip()
    recruit_kind = str(post.get("kindName") or post.get("kind") or "").strip()
    requirement = str(post.get("requirements") or "").strip()
    from_id = str(post.get("fromId") or "").strip()
    job_id = str(post.get("jobId") or "").strip()
    parts = [
        f"岗位名称：{title}" if title else "",
        "公司名称：携程",
        f"业务线：{bu_name}" if bu_name else "",
        f"工作地点：{city_name}" if city_name else "",
        f"职位类别：{family_name}" if family_name else "",
        f"职位类别编码：{family_code}" if family_code else "",
        f"招聘类型：{recruit_kind}" if recruit_kind else "",
        f"发布时间：{publish_date}" if publish_date else "",
        f"职位外部编号：{from_id}" if from_id else "",
        f"职位内部编号：{job_id}" if job_id else "",
        "职位描述：",
        requirement,
    ]
    return "\n".join(part for part in parts if part)


def convert_ctrip_job(post: dict[str, Any], *, crawl_time: str) -> dict[str, Any]:
    post_id = str(post.get("id") or "").strip()
    if not post_id:
        raise ValueError("Ctrip post missing id")
    raw_text = build_raw_text(post)
    external_job_id = str(post.get("jobId") or post_id).strip()
    return {
        "id": f"ctrip_{post_id}",
        "source": "careers.ctrip.com",
        "url": f"https://careers.ctrip.com/#/experienced/job-detail/{external_job_id}",
        "crawl_time": crawl_time,
        "job_title": str(post.get("jobTitle") or "").strip(),
        "company": "携程",
        "location": str(post.get("cityName") or "").strip(),
        "salary": "",
        "raw_text": raw_text,
        "html": None,
        "meta": {
            "language": "zh",
            "sft_ready": is_probably_tech_job(post),
            "post_id": post_id,
            "from_id": post.get("fromId"),
            "job_id": post.get("jobId"),
            "publish_date": post.get("publishDate"),
            "job_family_group_code": post.get("jobFamilyGroupCode"),
            "job_family_group_name": post.get("jobFamilyGroupName"),
            "bu_code": post.get("buCode"),
            "bu_name": post.get("buName"),
            "requirements": post.get("requirements"),
            "kind": post.get("kind"),
            "kind_name": post.get("kindName"),
            "channel_id": post.get("channelId"),
            "ats_api_type": post.get("atsApiType"),
            "city": post.get("city"),
            "city_name": post.get("cityName"),
        },
    }


def crawl_ctrip_jobs(
    *,
    page_size: int = 100,
    max_pages: int | None = None,
    interval_seconds: float = 0.2,
    timeout: float = 20.0,
    retries: int = 3,
) -> list[dict[str, Any]]:
    session = build_session(timeout)
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    page_index = 1
    total_pages: int | None = None
    while True:
        payload = {
            "head": {"language": "zh-CN", "version": "1"},
            "condition": {"pageIndex": page_index, "pageSize": page_size},
        }
        data = post_json(session, CTRIP_JOB_LIST_URL, payload, retries=retries)
        value = data.get("retValue") or {}
        posts = value.get("recruitJobAdList") or []
        total = int(value.get("total") or 0)
        if total_pages is None:
            total_pages = (total + page_size - 1) // page_size if total else 0
            if max_pages is not None:
                total_pages = min(total_pages, max_pages)
        if not posts:
            break
        for post in posts:
            if not is_probably_tech_job(post):
                continue
            post_id = str(post.get("id") or "").strip()
            if not post_id or post_id in seen_ids:
                continue
            row = convert_ctrip_job(post, crawl_time=crawl_time)
            if not row["raw_text"]:
                continue
            rows.append(row)
            seen_ids.add(post_id)
        page_index += 1
        if total_pages and page_index > total_pages:
            break
        time.sleep(interval_seconds)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/ctrip_jd_raw.jsonl")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    parser.add_argument("--no-merge-existing", action="store_true")
    args = parser.parse_args()

    rows = crawl_ctrip_jobs(
        page_size=args.page_size,
        max_pages=args.max_pages,
        interval_seconds=args.interval_seconds,
        timeout=args.timeout,
        retries=args.retries,
    )
    jsonl_rows = [{key: value for key, value in row.items() if key != "html"} for row in rows]
    merged_rows = jsonl_rows
    if not args.no_merge_existing:
        by_id: dict[str, dict[str, Any]] = {}
        try:
            for row in read_jsonl(args.out):
                row_id = str(row.get("id") or "")
                if row_id:
                    by_id[row_id] = row
        except FileNotFoundError:
            pass
        for row in jsonl_rows:
            by_id[str(row["id"])] = row
        merged_rows = list(by_id.values())
    write_jsonl(args.out, merged_rows)
    init_db(args.db)
    upsert_jd_raw(args.db, rows)
    print(f"crawled {len(rows)} Ctrip tech-like posts")
    print(f"wrote raw JSONL: {args.out} ({len(merged_rows)} rows)")
    print(f"upserted SQLite: {args.db}")


if __name__ == "__main__":
    main()
