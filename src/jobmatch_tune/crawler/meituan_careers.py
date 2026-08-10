from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Any

import requests

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


MEITUAN_JOB_LIST_URL = "https://zhaopin.meituan.com/api/official/job/getJobList"
MEITUAN_JOB_DETAIL_URL = "https://zhaopin.meituan.com/api/official/job/getJobDetail"

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
    "python",
    "java",
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
]

TECH_FAMILIES = {"技术类", "产品类"}
PRODUCT_TITLE_KEYWORDS = ["产品经理", "解决方案产品", "ai", "agent", "平台产品"]


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Content-Type": "application/json",
            "Origin": "https://zhaopin.meituan.com",
            "Referer": "https://zhaopin.meituan.com/",
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
            if not isinstance(data, dict) or data.get("status") != 1:
                raise RuntimeError(f"Unexpected Meituan response: {data}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep_seconds * attempt)
    assert last_error is not None
    raise last_error


def city_names(city_list: list[dict[str, Any]] | None) -> str:
    if not city_list:
        return ""
    names = [str(item.get("name") or "").strip() for item in city_list]
    return " / ".join(name for name in names if name)


def department_names(department_list: list[dict[str, Any]] | None) -> str:
    if not department_list:
        return ""
    names = [str(item.get("name") or "").strip() for item in department_list]
    return " / ".join(name for name in names if name)


def is_probably_tech_job(post: dict[str, Any], detail: dict[str, Any] | None = None) -> bool:
    title = str(post.get("name") or "").strip().lower()
    family = str((detail or post).get("jobFamily") or "").strip()
    family_group = str((detail or post).get("jobFamilyGroup") or "").strip().lower()
    combined = "\n".join(
        [
            title,
            family.lower(),
            family_group,
            str((detail or post).get("jobDuty") or "").lower(),
            str((detail or post).get("jobRequirement") or "").lower(),
            str((detail or post).get("highLight") or "").lower(),
        ]
    )
    if family == "技术类":
        return True
    if family == "产品类":
        return any(keyword in title for keyword in PRODUCT_TITLE_KEYWORDS)
    return any(keyword in combined for keyword in TECH_TITLE_KEYWORDS)


def build_raw_text(detail: dict[str, Any]) -> str:
    title = str(detail.get("name") or "").strip()
    family = str(detail.get("jobFamily") or "").strip()
    family_group = str(detail.get("jobFamilyGroup") or "").strip()
    location = city_names(detail.get("cityList"))
    work_year = str(detail.get("workYear") or "").strip()
    department = department_names(detail.get("department"))
    department_intro = str(detail.get("departmentIntro") or "").strip()
    duty = str(detail.get("jobDuty") or "").strip()
    requirement = str(detail.get("jobRequirement") or "").strip()
    precedence = str(detail.get("precedence") or "").strip()
    highlight = str(detail.get("highLight") or "").strip()
    parts = [
        f"岗位名称：{title}" if title else "",
        "公司名称：美团",
        f"职位类别：{family}" if family else "",
        f"职位子类：{family_group}" if family_group else "",
        f"工作地点：{location}" if location else "",
        f"经验要求：{work_year}" if work_year else "",
        f"所属部门：{department}" if department else "",
        "部门介绍：",
        department_intro,
        "岗位职责：",
        duty,
        "任职要求：",
        requirement,
        "优先条件：",
        precedence,
        "职位亮点：",
        highlight,
    ]
    return "\n".join(part for part in parts if part)


def convert_meituan_job(detail: dict[str, Any], *, crawl_time: str) -> dict[str, Any]:
    job_union_id = str(detail.get("jobUnionId") or "").strip()
    if not job_union_id:
        raise ValueError("Meituan detail missing jobUnionId")
    raw_text = build_raw_text(detail)
    return {
        "id": f"meituan_{job_union_id}",
        "source": "zhaopin.meituan.com",
        "url": f"https://zhaopin.meituan.com/web/position/detail?jobUnionId={job_union_id}",
        "crawl_time": crawl_time,
        "job_title": str(detail.get("name") or "").strip(),
        "company": "美团",
        "location": city_names(detail.get("cityList")),
        "salary": "",
        "raw_text": raw_text,
        "html": None,
        "meta": {
            "language": "zh",
            "sft_ready": is_probably_tech_job(detail, detail),
            "job_union_id": job_union_id,
            "job_family": detail.get("jobFamily"),
            "job_family_group": detail.get("jobFamilyGroup"),
            "work_year": detail.get("workYear"),
            "department_intro": detail.get("departmentIntro"),
            "job_duty": detail.get("jobDuty"),
            "job_requirement": detail.get("jobRequirement"),
            "precedence": detail.get("precedence"),
            "highlight": detail.get("highLight"),
        },
    }


def crawl_meituan_jobs(
    *,
    page_size: int = 20,
    max_pages: int | None = None,
    interval_seconds: float = 0.2,
    timeout: float = 20.0,
    retries: int = 3,
) -> list[dict[str, Any]]:
    session = build_session(timeout)
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    page_no = 1
    total_pages = None
    while True:
        payload = {"pageNo": page_no, "pageSize": page_size}
        data = post_json(session, MEITUAN_JOB_LIST_URL, payload, retries=retries)
        job_data = data.get("data") or {}
        posts = job_data.get("list") or []
        page = job_data.get("page") or {}
        if total_pages is None:
            total_pages = int(page.get("totalPage") or 0)
            if max_pages is not None:
                total_pages = min(total_pages, max_pages)
        if not posts:
            break
        for post in posts:
            if not is_probably_tech_job(post):
                continue
            job_union_id = str(post.get("jobUnionId") or "").strip()
            if not job_union_id or job_union_id in seen_ids:
                continue
            detail_data = post_json(
                session,
                MEITUAN_JOB_DETAIL_URL,
                {"jobUnionId": job_union_id},
                retries=retries,
            )
            detail = detail_data.get("data") or {}
            if not is_probably_tech_job(post, detail):
                continue
            row = convert_meituan_job(detail, crawl_time=crawl_time)
            if not row["raw_text"]:
                continue
            rows.append(row)
            seen_ids.add(job_union_id)
            time.sleep(interval_seconds)
        if total_pages is not None and page_no >= total_pages:
            break
        page_no += 1
        time.sleep(interval_seconds)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--out", default="data/raw/meituan_jd_raw.jsonl")
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    parser.add_argument("--no-merge-existing", action="store_true")
    args = parser.parse_args()

    rows = crawl_meituan_jobs(
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
    print(f"crawled {len(rows)} Meituan tech-like posts")
    print(f"wrote raw JSONL: {args.out} ({len(merged_rows)} rows)")
    print(f"upserted SQLite: {args.db}")


if __name__ == "__main__":
    main()
