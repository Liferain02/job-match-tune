from __future__ import annotations

import argparse
import html
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


MOKA_JOBS_URL_TEMPLATE = "https://api.mokahr.com/api-platform/v1/jobs/{org_id}"

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
    "infra",
    "sre",
    "运维",
    "云",
    "平台",
    "架构",
]

ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def load_sources(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    sources = payload.get("sources") or []
    if not isinstance(sources, list):
        raise ValueError(f"Invalid source manifest: {path}")
    return sources


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


def strip_html_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|ul|ol|h\d)>", "\n", text, flags=re.IGNORECASE)
    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    if not text:
        return ""
    zh_count = len(ZH_CHAR_RE.findall(text))
    ascii_count = sum(1 for char in text if char.isascii() and char.isalpha())
    if zh_count >= max(8, ascii_count // 2):
        return "zh"
    if ascii_count > zh_count:
        return "en"
    return ""


def is_probably_tech_job(job: dict[str, Any], raw_text: str) -> bool:
    title = str(job.get("title") or "").strip().lower()
    combined = "\n".join(
        [
            title,
            raw_text.lower(),
            str(job.get("department", {}).get("name") or "").lower(),
            str(job.get("zhineng", {}).get("name") or "").lower(),
        ]
    )
    return any(keyword in combined for keyword in TECH_TITLE_KEYWORDS)


def format_locations(locations: list[dict[str, Any]] | None) -> str:
    if not locations:
        return ""
    parts: list[str] = []
    for item in locations:
        values = [
            str(item.get("country") or "").strip(),
            str(item.get("province") or "").strip(),
            str(item.get("city") or "").strip(),
            str(item.get("area") or "").strip(),
            str(item.get("address") or "").strip(),
        ]
        text = " / ".join(value for value in values if value)
        if text:
            parts.append(text)
    return " | ".join(dict.fromkeys(parts))


def build_experience_string(job: dict[str, Any]) -> str:
    minimum = job.get("minExperience")
    maximum = job.get("maxExperience")
    if minimum in (None, "") and maximum in (None, ""):
        return ""
    if minimum not in (None, "") and maximum not in (None, ""):
        return f"{minimum}-{maximum}年"
    if minimum not in (None, ""):
        return f"{minimum}年以上"
    return f"{maximum}年以内"


def build_salary_string(job: dict[str, Any]) -> str:
    minimum = job.get("minSalary")
    maximum = job.get("maxSalary")
    if minimum in (None, "") and maximum in (None, ""):
        return ""
    if minimum not in (None, "") and maximum not in (None, ""):
        return f"{minimum}-{maximum}K"
    if minimum not in (None, ""):
        return f"{minimum}K以上"
    return f"{maximum}K以内"


def build_raw_text(job: dict[str, Any], company: str, mode: str) -> str:
    title = str(job.get("title") or "").strip()
    description = strip_html_text(job.get("description"))
    location = format_locations(job.get("locations"))
    salary = build_salary_string(job)
    education = str(job.get("education") or "").strip()
    experience = build_experience_string(job)
    commitment = str(job.get("commitment") or "").strip()
    department = str((job.get("department") or {}).get("name") or "").strip()
    function_name = str((job.get("zhineng") or {}).get("name") or "").strip()
    published_at = str(job.get("publishedAt") or "").strip()
    updated_at = str(job.get("updatedAt") or "").strip()

    sections = [
        f"岗位名称：{title}" if title else "",
        f"公司名称：{company}" if company else "",
        f"招聘模式：{'社招' if mode == 'social' else '校招'}",
        f"工作地点：{location}" if location else "",
        f"薪资范围：{salary}" if salary else "",
        f"学历要求：{education}" if education else "",
        f"经验要求：{experience}" if experience else "",
        f"职位性质：{commitment}" if commitment else "",
        f"所属部门：{department}" if department else "",
        f"职能类别：{function_name}" if function_name else "",
        f"发布时间：{published_at}" if published_at else "",
        f"更新时间：{updated_at}" if updated_at else "",
        "职位描述：",
        description,
    ]
    return "\n".join(part for part in sections if part)


def fetch_jobs_page(
    session: requests.Session,
    *,
    org_id: str,
    mode: str,
    limit: int,
    offset: int,
    keyword: str = "",
    site_id: int | None = None,
    status: str = "open",
    retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    url = MOKA_JOBS_URL_TEMPLATE.format(org_id=org_id)
    params: dict[str, Any] = {
        "mode": mode,
        "limit": limit,
        "offset": offset,
    }
    if keyword:
        params["keyword"] = keyword
    if site_id is not None:
        params["siteId"] = site_id
    if status:
        params["status"] = status

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or "jobs" not in payload:
                raise RuntimeError(f"Unexpected Moka API payload for org={org_id}: {payload}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep_seconds * attempt)
    assert last_error is not None
    raise last_error


def convert_moka_job(
    job: dict[str, Any],
    *,
    org_id: str,
    company: str,
    mode: str,
    site_id: int | None,
    source_name: str,
    source_url: str,
    crawl_time: str,
) -> dict[str, Any]:
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        raise ValueError("Moka job is missing id")
    raw_text = build_raw_text(job, company=company, mode=mode)
    language = detect_language(raw_text)
    location = format_locations(job.get("locations"))
    detail_url = (
        f"https://app.mokahr.com/{'social-recruitment' if mode == 'social' else 'campus-recruitment'}/"
        f"{org_id}/{site_id}#/"
        if site_id is not None
        else f"https://app.mokahr.com/{'social-recruitment' if mode == 'social' else 'campus-recruitment'}/{org_id}#/"
    )
    return {
        "id": f"moka_{org_id}_{mode}_{job_id}",
        "source": source_name,
        "url": detail_url,
        "crawl_time": crawl_time,
        "job_title": str(job.get("title") or "").strip(),
        "company": company,
        "location": location,
        "salary": build_salary_string(job),
        "raw_text": raw_text,
        "html": None,
        "meta": {
            "schema": "moka_jobs_v1",
            "language": language,
            "sft_ready": language == "zh" and is_probably_tech_job(job, raw_text),
            "org_id": org_id,
            "job_id": job_id,
            "mode": mode,
            "site_id": site_id,
            "status": job.get("status"),
            "education": job.get("education"),
            "min_experience": job.get("minExperience"),
            "max_experience": job.get("maxExperience"),
            "commitment": job.get("commitment"),
            "department": job.get("department"),
            "locations": job.get("locations"),
            "zhineng": job.get("zhineng"),
            "published_at": job.get("publishedAt"),
            "updated_at": job.get("updatedAt"),
            "opened_at": job.get("openedAt"),
            "closed_at": job.get("closedAt"),
            "hire_mode": job.get("hireMode"),
            "number": job.get("number"),
            "is_recommendation": job.get("isRecommendation"),
            "custom_fields": job.get("customFields") or [],
            "source_url": source_url,
        },
    }


def crawl_source(
    session: requests.Session,
    source: dict[str, Any],
    *,
    page_limit: int,
    interval_seconds: float,
    retries: int,
    keyword: str = "",
) -> list[dict[str, Any]]:
    org_id = str(source.get("org_id") or "").strip()
    company = str(source.get("company") or org_id).strip()
    source_name = str(source.get("source_name") or f"moka_{org_id}").strip()
    source_url = str(source.get("source_url") or "").strip()
    modes = source.get("modes") or ["social"]
    site_ids = source.get("site_ids") or {}
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, Any]] = []
    for mode in modes:
        site_id = site_ids.get(mode)
        offset = 0
        total = None
        while total is None or offset < total:
            payload = fetch_jobs_page(
                session,
                org_id=org_id,
                mode=mode,
                limit=page_limit,
                offset=offset,
                keyword=keyword,
                site_id=site_id,
                retries=retries,
            )
            total = int(payload.get("total") or 0)
            jobs = payload.get("jobs") or []
            if not jobs:
                break
            for job in jobs:
                try:
                    row = convert_moka_job(
                        job,
                        org_id=org_id,
                        company=company,
                        mode=mode,
                        site_id=site_id,
                        source_name=source_name,
                        source_url=source_url,
                        crawl_time=crawl_time,
                    )
                except ValueError:
                    continue
                if row["raw_text"]:
                    rows.append(row)
            offset += len(jobs)
            if len(jobs) < page_limit:
                break
            time.sleep(interval_seconds)
        time.sleep(interval_seconds)
    return rows


def crawl_moka_sources(
    sources: list[dict[str, Any]],
    *,
    page_limit: int = 30,
    interval_seconds: float = 0.2,
    timeout: float = 20.0,
    retries: int = 3,
    keyword: str = "",
) -> list[dict[str, Any]]:
    session = build_session(timeout)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in sources:
        org_id = str(source.get("org_id") or "").strip()
        if not org_id:
            continue
        source_rows = crawl_source(
            session,
            source,
            page_limit=page_limit,
            interval_seconds=interval_seconds,
            retries=retries,
            keyword=keyword,
        )
        for row in source_rows:
            row_id = str(row.get("id") or "")
            if not row_id or row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="configs/moka_sources.yaml")
    parser.add_argument("--page-limit", type=int, default=30)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--keyword", default="")
    parser.add_argument("--out", default="data/raw/moka_jd_raw.jsonl")
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    parser.add_argument("--no-merge-existing", action="store_true")
    args = parser.parse_args()

    sources = load_sources(args.sources)
    rows = crawl_moka_sources(
        sources=sources,
        page_limit=args.page_limit,
        interval_seconds=args.interval_seconds,
        timeout=args.timeout,
        retries=args.retries,
        keyword=args.keyword,
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
    print(f"crawled {len(rows)} Moka posts from {len(sources)} sources")
    print(f"wrote raw JSONL: {args.out} ({len(merged_rows)} rows)")
    print(f"upserted SQLite: {args.db}")


if __name__ == "__main__":
    main()
