from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


JD_JOB_COUNT_URL = "https://zhaopin.jd.com/web/job/job_count"
JD_JOB_LIST_URL = "https://zhaopin.jd.com/web/job/job_list"
JD_JOB_ALLPARAMS_URL = "https://zhaopin.jd.com/web/job/job_allparams"
JD_REFERER = "https://zhaopin.jd.com/web/job/job_info_list/3"

TECH_TITLE_KEYWORDS = [
    "开发",
    "研发",
    "算法",
    "前端",
    "后端",
    "测试",
    "客户端",
    "数据",
    "ai",
    "大模型",
    "模型",
    "java",
    "python",
    "c++",
    "android",
    "ios",
    "机器学习",
    "深度学习",
    "agent",
]


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
            ),
            "Referer": JD_REFERER,
            "Origin": "https://zhaopin.jd.com",
        }
    )
    session.request = _with_timeout(session.request, timeout)
    return session


def _with_timeout(request_fn: Any, timeout: float) -> Any:
    def wrapped(method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", timeout)
        return request_fn(method, url, **kwargs)

    return wrapped


def jd_post_json(session: requests.Session, url: str, data: dict[str, Any]) -> Any:
    response = session.post(url, data=data)
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        return None
    return json.loads(text)


def is_probably_tech_job(post: dict[str, Any]) -> bool:
    title = str(post.get("positionNameOpen") or post.get("positionName") or "").strip().lower()
    if str(post.get("jobTypeCode") or "").strip().upper() == "YANFA":
        return True
    combined = "\n".join(
        [
            title,
            str(post.get("workContent") or ""),
            str(post.get("qualification") or ""),
        ]
    ).lower()
    return any(keyword in combined for keyword in TECH_TITLE_KEYWORDS)


def build_jd_raw_text(post: dict[str, Any]) -> str:
    title = str(post.get("positionNameOpen") or post.get("positionName") or "").strip()
    dept_name = str(post.get("positionDeptName") or "").strip()
    job_type = str(post.get("jobType") or "").strip()
    location = str(post.get("workCity") or "").strip()
    publish_time = str(post.get("formatPublishTime") or "").strip()
    work_content = str(post.get("workContent") or "").strip()
    qualification = str(post.get("qualification") or "").strip()
    req_number = str(post.get("reqNumber") or "").strip()
    parts = [
        f"岗位名称：{title}" if title else "",
        "公司名称：京东",
        f"所属业务：{dept_name}" if dept_name else "",
        f"职位类别：{job_type}" if job_type else "",
        f"工作地点：{location}" if location else "",
        f"职位编号：{req_number}" if req_number else "",
        f"发布日期：{publish_time}" if publish_time else "",
        "岗位职责：",
        work_content,
        "任职要求：",
        qualification,
    ]
    return "\n".join(part for part in parts if part)


def convert_jd_post(post: dict[str, Any], crawl_time: str) -> dict[str, Any]:
    requirement_id = str(post.get("requirementId") or "").strip()
    if not requirement_id:
        raise ValueError("JD post is missing requirementId")
    position_id = str(post.get("positionId") or "").strip()
    title = str(post.get("positionNameOpen") or post.get("positionName") or "").strip()
    return {
        "id": f"jd_careers_{requirement_id}",
        "source": "zhaopin.jd.com",
        "url": f"https://zhaopin.jd.com/web/job_info_detail?recruitType=2&positionId={position_id}&requirementId={requirement_id}",
        "crawl_time": crawl_time,
        "job_title": title,
        "company": "京东",
        "location": str(post.get("workCity") or "").strip(),
        "salary": "",
        "raw_text": build_jd_raw_text(post),
        "html": None,
        "meta": {
            "language": "zh",
            "sft_ready": is_probably_tech_job(post),
            "job_type": post.get("jobType"),
            "job_type_code": post.get("jobTypeCode"),
            "position_id": position_id,
            "position_code": post.get("positionCode"),
            "position_dept_name": post.get("positionDeptName"),
            "position_dept_code": post.get("positionDeptCode"),
            "publish_time": post.get("publishTime"),
            "format_publish_time": post.get("formatPublishTime"),
            "qualification": post.get("qualification"),
            "work_content": post.get("workContent"),
            "work_city_code": post.get("workCityCode"),
            "req_number": post.get("reqNumber"),
        },
    }


def fetch_total_count(
    session: requests.Session,
    *,
    job_search: str = "",
    work_city_json: str = "[]",
    job_type_json: str = "[]",
    dep_type_json: str = "[]",
) -> int:
    payload = {
        "workCityJson": work_city_json,
        "jobTypeJson": job_type_json,
        "jobSearch": job_search,
        "depTypeJson": dep_type_json,
    }
    response = session.post(JD_JOB_COUNT_URL, data=payload)
    response.raise_for_status()
    return int(response.text.strip() or "0")


def fetch_all_params(session: requests.Session) -> dict[str, Any]:
    return jd_post_json(session, JD_JOB_ALLPARAMS_URL, data={}) or {}


def fetch_job_page(
    session: requests.Session,
    *,
    page_index: int,
    page_size: int,
    job_search: str = "",
    work_city_json: str = "[]",
    job_type_json: str = "[]",
    dep_type_json: str = "[]",
) -> list[dict[str, Any]]:
    payload = {
        "pageIndex": page_index,
        "pageSize": page_size,
        "workCityJson": work_city_json,
        "jobTypeJson": job_type_json,
        "jobSearch": job_search,
        "depTypeJson": dep_type_json,
    }
    data = jd_post_json(session, JD_JOB_LIST_URL, data=payload)
    return data if isinstance(data, list) else []


def crawl_jd_jobs(
    *,
    page_size: int = 10,
    interval_seconds: float = 0.2,
    timeout: float = 20.0,
    job_search: str = "",
    job_type_codes: list[str] | None = None,
    dep_codes: list[str] | None = None,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    session = build_session(timeout)
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_type_json = json.dumps(job_type_codes or [], ensure_ascii=False)
    dep_type_json = json.dumps(dep_codes or [], ensure_ascii=False)
    total = fetch_total_count(
        session,
        job_search=job_search,
        job_type_json=job_type_json,
        dep_type_json=dep_type_json,
    )
    page_count = (total + page_size - 1) // page_size
    if max_pages is not None:
        page_count = min(page_count, max_pages)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for page_index in range(1, page_count + 1):
        posts = fetch_job_page(
            session,
            page_index=page_index,
            page_size=page_size,
            job_search=job_search,
            job_type_json=job_type_json,
            dep_type_json=dep_type_json,
        )
        for post in posts:
            try:
                row = convert_jd_post(post, crawl_time)
            except ValueError:
                continue
            if row["id"] in seen_ids or not row["raw_text"]:
                continue
            seen_ids.add(row["id"])
            rows.append(row)
        time.sleep(interval_seconds)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--job-search", default="")
    parser.add_argument("--job-type-code", action="append", default=[])
    parser.add_argument("--dep-code", action="append", default=[])
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--print-allparams", action="store_true")
    parser.add_argument("--out", default="data/raw/jd_careers_raw.jsonl")
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    parser.add_argument("--no-merge-existing", action="store_true")
    args = parser.parse_args()

    if args.print_allparams:
        payload = fetch_all_params(build_session(args.timeout))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    rows = crawl_jd_jobs(
        page_size=args.page_size,
        interval_seconds=args.interval_seconds,
        timeout=args.timeout,
        job_search=args.job_search,
        job_type_codes=args.job_type_code,
        dep_codes=args.dep_code,
        max_pages=args.max_pages,
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
    print(f"crawled {len(rows)} JD posts")
    print(f"wrote raw JSONL: {args.out} ({len(merged_rows)} rows)")
    print(f"upserted SQLite: {args.db}")


if __name__ == "__main__":
    main()
