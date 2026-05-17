from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


BAIDU_SOCIAL_LIST_URL = "https://talent.baidu.com/jobs/social-list"
INITIAL_DATA_PATTERN = re.compile(
    r"window\.__INITIAL_DATA__\s*=(.*?);\s*window\.prefix=",
    re.DOTALL,
)


def extract_initial_data_payload(html: str) -> dict[str, Any]:
    match = INITIAL_DATA_PATTERN.search(html)
    if not match:
        raise ValueError("Failed to locate window.__INITIAL_DATA__ payload")
    payload = match.group(1).strip()
    payload = re.sub(r":undefined([,}])", r":null\1", payload)
    return json.loads(payload)


def fetch_baidu_social_page(
    keyword: str,
    timeout: float = 20.0,
    retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    params = {"search": keyword} if keyword else None
    last_error: Exception | None = None
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
        )
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                BAIDU_SOCIAL_LIST_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return extract_initial_data_payload(response.text)
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep_seconds * attempt)
    assert last_error is not None
    raise last_error


def build_baidu_raw_text(post: dict[str, Any]) -> str:
    title = str(post.get("name") or "").strip()
    location = str(post.get("workPlace") or "").strip()
    post_type = str(post.get("postType") or "").strip()
    bg_short_name = str(post.get("bgShortName") or "").strip()
    recruit_num = str(post.get("recruitNum") or "").strip()
    publish_date = str(post.get("publishDate") or "").strip()
    update_date = str(post.get("updateDate") or "").strip()
    service_condition = str(post.get("serviceCondition") or "").strip()
    work_content = str(post.get("workContent") or "").strip()
    parts = [
        f"岗位名称：{title}" if title else "",
        "公司名称：百度",
        f"工作地点：{location}" if location else "",
        f"职位类别：{post_type}" if post_type else "",
        f"业务线：{bg_short_name}" if bg_short_name else "",
        f"招聘人数：{recruit_num}" if recruit_num else "",
        f"发布日期：{publish_date}" if publish_date else "",
        f"更新时间：{update_date}" if update_date else "",
        "岗位职责：",
        work_content,
        "任职要求：",
        service_condition,
    ]
    return "\n".join(part for part in parts if part)


def convert_baidu_post(
    post: dict[str, Any],
    *,
    keyword: str,
    crawl_time: str,
    recruit_type: str = "SOCIAL",
) -> dict[str, Any]:
    post_id = str(post.get("postId") or "").strip()
    if not post_id:
        raise ValueError("Baidu post is missing postId")
    title = str(post.get("name") or "").strip()
    url = f"https://talent.baidu.com/jobs/detail/{recruit_type}/{post_id}"
    return {
        "id": f"baidu_{post_id}",
        "source": "talent.baidu.com",
        "url": url,
        "crawl_time": crawl_time,
        "job_title": title,
        "company": "百度",
        "location": str(post.get("workPlace") or "").strip(),
        "salary": "",
        "raw_text": build_baidu_raw_text(post),
        "html": None,
        "meta": {
            "language": "zh",
            "sft_ready": True,
            "keyword": keyword,
            "post_id": post_id,
            "job_id": post.get("jobId"),
            "recruit_type": recruit_type,
            "post_type": post.get("postType"),
            "publish_date": post.get("publishDate"),
            "update_date": post.get("updateDate"),
            "recruit_num": post.get("recruitNum"),
            "service_condition": post.get("serviceCondition"),
            "work_content": post.get("workContent"),
            "work_years": post.get("workYears"),
            "education": post.get("education"),
            "project_type": post.get("projectType"),
            "bg_short_name": post.get("bg_short_name") or post.get("bgShortName"),
            "hot_flag": bool(post.get("hotFlag")),
            "favorite_flag": bool(post.get("favoriteFlag")),
        },
    }


def crawl_baidu_social(
    keywords: list[str],
    interval_seconds: float = 0.5,
    retries: int = 3,
    include_empty_search: bool = True,
) -> list[dict[str, Any]]:
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    queries = [""] if include_empty_search else []
    queries.extend(keywords)
    for keyword in queries:
        if keyword in seen_queries:
            continue
        seen_queries.add(keyword)
        payload = fetch_baidu_social_page(keyword=keyword, retries=retries)
        list_data = (payload.get("listData") or {}) if isinstance(payload, dict) else {}
        recruit_type = str(list_data.get("recruitType") or "SOCIAL")
        posts = list_data.get("listDetailData") or []
        for post in posts:
            try:
                row = convert_baidu_post(
                    post,
                    keyword=keyword,
                    crawl_time=crawl_time,
                    recruit_type=recruit_type,
                )
            except ValueError:
                continue
            if row["id"] in seen_ids or not row["raw_text"]:
                continue
            seen_ids.add(row["id"])
            rows.append(row)
        time.sleep(interval_seconds)
    return rows


def build_search_url(keyword: str) -> str:
    if not keyword:
        return BAIDU_SOCIAL_LIST_URL
    return f"{BAIDU_SOCIAL_LIST_URL}?{urlencode({'search': keyword})}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--keywords-file", default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--out", default="data/raw/baidu_jd_raw.jsonl")
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    parser.add_argument("--no-merge-existing", action="store_true")
    parser.add_argument("--no-empty-search", action="store_true")
    args = parser.parse_args()

    keywords = args.keyword[:]
    if args.keywords_file:
        with Path(args.keywords_file).open("r", encoding="utf-8") as f:
            keywords.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    keywords = list(dict.fromkeys(keywords))

    rows = crawl_baidu_social(
        keywords=keywords,
        interval_seconds=args.interval_seconds,
        retries=args.retries,
        include_empty_search=not args.no_empty_search,
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
    print(f"crawled {len(rows)} Baidu posts for keywords={keywords}")
    print(f"wrote raw JSONL: {args.out} ({len(merged_rows)} rows)")
    print(f"upserted SQLite: {args.db}")


if __name__ == "__main__":
    main()
