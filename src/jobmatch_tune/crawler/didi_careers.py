from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Any

import requests

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


DIDI_JOB_LIST_URL = "https://talent.didiglobal.com/recruit-portal-service/api/job/front/list"
DIDI_JOB_DETAIL_URL = "https://talent.didiglobal.com/recruit-portal-service/api/job/front/view/{jd_id}"

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

TECH_JOB_TYPES = {
    "技术",
    "研发",
    "算法",
    "安全",
    "测试",
    "数据",
    "运维",
    "基础平台",
    "客户端",
    "前端",
    "后端",
}


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://talent.didiglobal.com/social/list/1",
        }
    )
    session.request = _with_timeout(session.request, timeout)
    return session


def _with_timeout(request_fn: Any, timeout: float) -> Any:
    def wrapped(method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", timeout)
        return request_fn(method, url, **kwargs)

    return wrapped


def get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or (data.get("meta") or {}).get("code") != 0:
                raise RuntimeError(f"Unexpected Didi response: {data}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep_seconds * attempt)
    assert last_error is not None
    raise last_error


def is_probably_tech_job(post: dict[str, Any], detail: dict[str, Any] | None = None) -> bool:
    title = str(post.get("jobName") or "").strip().lower()
    job_type = str((detail or post).get("jobTypeName") or (detail or post).get("jobType") or "").strip().lower()
    combined = "\n".join(
        [
            title,
            job_type,
            str((detail or post).get("jobDesc") or "").lower(),
            str((detail or post).get("qualification") or "").lower(),
            str((detail or post).get("deptName") or "").lower(),
        ]
    )
    if any(keyword in job_type for keyword in TECH_JOB_TYPES):
        return True
    return any(keyword in combined for keyword in TECH_TITLE_KEYWORDS)


def build_raw_text(detail: dict[str, Any]) -> str:
    title = str(detail.get("jobName") or "").strip()
    department = str(detail.get("deptName") or "").strip()
    work_area = str(detail.get("workArea") or "").strip()
    publish_time = str(detail.get("publishTime") or "").strip()
    refresh_time = str(detail.get("refreshTime") or "").strip()
    recruit_type = str(detail.get("recruitType") or "").strip()
    jd_no = str(detail.get("jdNo") or "").strip()
    recruit_num = str(detail.get("recruitNum") or "").strip()
    job_type = str(detail.get("jobTypeName") or detail.get("jobType") or "").strip()
    duty = str(detail.get("jobDesc") or "").strip()
    requirement = str(detail.get("qualification") or "").strip()
    parts = [
        f"岗位名称：{title}" if title else "",
        "公司名称：滴滴",
        f"所属部门：{department}" if department else "",
        f"工作地点：{work_area}" if work_area else "",
        f"职位类别：{job_type}" if job_type else "",
        f"招聘类型：{recruit_type}" if recruit_type else "",
        f"职位编号：{jd_no}" if jd_no else "",
        f"招聘人数：{recruit_num}" if recruit_num else "",
        f"发布时间：{publish_time}" if publish_time else "",
        f"刷新时间：{refresh_time}" if refresh_time else "",
        "岗位职责：",
        duty,
        "任职要求：",
        requirement,
    ]
    return "\n".join(part for part in parts if part)


def convert_didi_job(detail: dict[str, Any], *, jd_id: str, crawl_time: str) -> dict[str, Any]:
    jd_id = str(jd_id or "").strip()
    if not jd_id:
        raise ValueError("Didi detail missing jdId")
    raw_text = build_raw_text(detail)
    return {
        "id": f"didi_{jd_id}",
        "source": "talent.didiglobal.com",
        "url": f"https://talent.didiglobal.com/social/p/{jd_id}",
        "crawl_time": crawl_time,
        "job_title": str(detail.get("jobName") or "").strip(),
        "company": "滴滴",
        "location": str(detail.get("workArea") or "").strip(),
        "salary": "",
        "raw_text": raw_text,
        "html": None,
        "meta": {
            "language": "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in raw_text) else "en",
            "sft_ready": is_probably_tech_job(detail, detail),
            "jd_id": jd_id,
            "jd_no": detail.get("jdNo"),
            "dept_name": detail.get("deptName"),
            "publish_time": detail.get("publishTime"),
            "refresh_time": detail.get("refreshTime"),
            "recruit_type": detail.get("recruitType"),
            "recruit_num": detail.get("recruitNum"),
            "job_type": detail.get("jobType"),
            "job_desc": detail.get("jobDesc"),
            "qualification": detail.get("qualification"),
            "record_id": detail.get("recordId"),
        },
    }


def crawl_didi_jobs(
    *,
    max_pages: int | None = None,
    interval_seconds: float = 0.1,
    timeout: float = 20.0,
    retries: int = 3,
) -> list[dict[str, Any]]:
    session = build_session(timeout)
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            break
        data = get_json(session, DIDI_JOB_LIST_URL, params={"page": page}, retries=retries)
        items = (data.get("data") or {}).get("items") or []
        if not items:
            break
        for post in items:
            if not is_probably_tech_job(post):
                continue
            jd_id = str(post.get("jdId") or "").strip()
            if not jd_id or jd_id in seen_ids:
                continue
            detail_data = get_json(
                session,
                DIDI_JOB_DETAIL_URL.format(jd_id=jd_id),
                retries=retries,
            )
            detail = detail_data.get("data") or {}
            if not detail or not is_probably_tech_job(post, detail):
                continue
            row = convert_didi_job(detail, jd_id=jd_id, crawl_time=crawl_time)
            if not row["raw_text"]:
                continue
            rows.append(row)
            seen_ids.add(jd_id)
            time.sleep(interval_seconds)
        page += 1
        time.sleep(interval_seconds)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/didi_jd_raw.jsonl")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    parser.add_argument("--no-merge-existing", action="store_true")
    args = parser.parse_args()

    rows = crawl_didi_jobs(
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
    print(f"crawled {len(rows)} Didi tech-like posts")
    print(f"wrote raw JSONL: {args.out} ({len(merged_rows)} rows)")
    print(f"upserted SQLite: {args.db}")


if __name__ == "__main__":
    main()
