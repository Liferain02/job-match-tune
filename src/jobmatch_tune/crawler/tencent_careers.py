from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Any

import requests

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


TENCENT_QUERY_URL = "https://careers.tencent.com/tencentcareer/api/post/Query"


def fetch_tencent_posts(keyword: str, page_size: int = 20, page_index: int = 1) -> tuple[int, list[dict[str, Any]]]:
    params = {
        "timestamp": int(datetime.now().timestamp() * 1000),
        "countryId": "",
        "cityId": "",
        "bgIds": "",
        "productId": "",
        "categoryId": "",
        "parentCategoryId": "",
        "attrId": "",
        "keyword": keyword,
        "pageIndex": page_index,
        "pageSize": page_size,
        "language": "zh-cn",
        "area": "cn",
    }
    response = requests.get(TENCENT_QUERY_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("Code") != 200:
        raise RuntimeError(f"Tencent API error: {payload}")
    data = payload.get("Data", {})
    return int(data.get("Count") or 0), data.get("Posts", [])


def convert_post(post: dict[str, Any]) -> dict[str, Any]:
    post_id = str(post.get("PostId") or post.get("RecruitPostId"))
    title = post.get("RecruitPostName") or ""
    location = post.get("LocationName") or ""
    company = post.get("ComName") or "腾讯"
    raw_text = "\n".join(
        part
        for part in [
            f"岗位名称：{title}",
            f"工作地点：{location}" if location else "",
            f"事业群：{post.get('BGName')}" if post.get("BGName") else "",
            "岗位职责：",
            post.get("Responsibility") or "",
            f"经验要求：{post.get('RequireWorkYearsName')}" if post.get("RequireWorkYearsName") else "",
        ]
        if part
    )
    return {
        "id": f"tencent_{post_id}",
        "source": "careers.tencent.com",
        "url": post.get("PostURL") or f"https://careers.tencent.com/jobdesc.html?postId={post_id}",
        "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "job_title": title,
        "company": company,
        "location": location,
        "salary": "",
        "raw_text": raw_text,
        "html": None,
        "meta": {
            "post_id": post_id,
            "category": post.get("CategoryName"),
            "product": post.get("ProductName"),
            "last_update_time": post.get("LastUpdateTime"),
        },
    }


def crawl_tencent(
    keywords: list[str],
    limit: int,
    page_size: int,
    max_pages: int,
    interval_seconds: float,
    category_allowlist: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for keyword in keywords:
        for page_index in range(1, max_pages + 1):
            total, posts = fetch_tencent_posts(
                keyword=keyword,
                page_size=page_size,
                page_index=page_index,
            )
            if not posts:
                break
            for post in posts:
                category = post.get("CategoryName") or ""
                if category_allowlist and category not in category_allowlist:
                    continue
                post_id = str(post.get("PostId") or post.get("RecruitPostId"))
                if not post_id or post_id in seen:
                    continue
                seen.add(post_id)
                rows.append(convert_post(post))
                if len(rows) >= limit:
                    return rows
            if page_index * page_size >= total:
                break
            time.sleep(interval_seconds)
        time.sleep(interval_seconds)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--keywords-file", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    parser.add_argument("--category", action="append", default=["技术"])
    parser.add_argument("--out", default="data/raw/tencent_jd_raw.jsonl")
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    parser.add_argument("--no-merge-existing", action="store_true")
    args = parser.parse_args()

    keywords = args.keyword[:]
    if args.keywords_file:
        with open(args.keywords_file, "r", encoding="utf-8") as f:
            keywords.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    if not keywords:
        keywords = ["大模型"]

    rows = crawl_tencent(
        keywords=list(dict.fromkeys(keywords)),
        limit=args.limit,
        page_size=args.page_size,
        max_pages=args.max_pages,
        interval_seconds=args.interval_seconds,
        category_allowlist=set(args.category) if args.category else None,
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
    print(f"crawled {len(rows)} Tencent posts for keywords={keywords}")
    print(f"wrote raw JSONL: {args.out} ({len(merged_rows)} rows)")
    print(f"upserted SQLite: {args.db}")


if __name__ == "__main__":
    main()
