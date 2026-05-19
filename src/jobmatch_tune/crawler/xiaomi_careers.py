from __future__ import annotations

import argparse
import html
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


XIAOMI_LIST_URL_TEMPLATE = "https://hr.xiaomi.com/job/list/{path}"
LIST_ROW_RE = re.compile(
    r"<tr>\s*"
    r'<td class="first"><a href="(?P<href>https://hr\.xiaomi\.com/job/view/(?P<job_id>\d+))">'
    r"(?P<title>.*?)"
    r"(?:<span class=\"hot-tip\"></span>)?</a></td>\s*"
    r"<td>(?P<category>.*?)</td>\s*"
    r"<td>(?P<location>.*?)</td>\s*"
    r"<td>(?P<publish_date>.*?)</td>\s*"
    r"</tr>",
    re.DOTALL,
)
DETAIL_FIELD_RE = re.compile(
    r'<td class="details-title(?: require)?">(?P<label>[^<：]+)：</td>\s*'
    r'<td class="(?P<class_name>job-details|details-list)"(?:\s+colspan="\d+")?>(?P<value>.*?)</td>',
    re.DOTALL,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")

TECH_TITLE_KEYWORDS = [
    "开发",
    "研发",
    "算法",
    "前端",
    "后端",
    "测试",
    "客户端",
    "android",
    "ios",
    "python",
    "java",
    "c++",
    "音视频",
    "sdk",
    "infra",
    "sre",
    "平台",
    "编译",
    "数据",
    "ai",
    "大模型",
    "模型",
    "服务器",
    "中间件",
]

LOCATION_ALIASES = {
    "peking": "北京",
    "beijing": "北京",
    "wuhan": "武汉",
    "shanghai": "上海",
    "nanjing": "南京",
    "guangzhou": "广州",
    "shenzhen": "深圳",
}


def clean_html_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|ul|ol)>", "\n", text, flags=re.IGNORECASE)
    text = HTML_TAG_RE.sub("", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_location(value: str) -> str:
    text = clean_html_text(value).strip()
    return LOCATION_ALIASES.get(text.lower(), text)


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    session.request = _with_timeout(session.request, timeout)
    return session


def _with_timeout(request_fn: Any, timeout: float) -> Any:
    def wrapped(method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", timeout)
        return request_fn(method, url, **kwargs)

    return wrapped


def fetch_html(
    session: requests.Session,
    url: str,
    *,
    retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep_seconds * attempt)
    assert last_error is not None
    raise last_error


def parse_list_rows(html_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for match in LIST_ROW_RE.finditer(html_text):
        rows.append(
            {
                "job_id": match.group("job_id").strip(),
                "href": match.group("href").strip(),
                "title": clean_html_text(match.group("title")),
                "category": clean_html_text(match.group("category")),
                "location": normalize_location(match.group("location")),
                "publish_date": clean_html_text(match.group("publish_date")),
            }
        )
    return rows


def parse_detail_fields(html_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in DETAIL_FIELD_RE.finditer(html_text):
        label = clean_html_text(match.group("label"))
        value = clean_html_text(match.group("value"))
        if label and value:
            fields[label] = value
    return fields


def is_probably_tech_job(title: str, raw_text: str) -> bool:
    combined = f"{title}\n{raw_text}".lower()
    return any(keyword in combined for keyword in TECH_TITLE_KEYWORDS)


def build_raw_text(
    *,
    title: str,
    location: str,
    category: str,
    publish_date: str,
    channel: str,
    duties: str,
    requirements: str,
) -> str:
    parts = [
        f"岗位名称：{title}" if title else "",
        "公司名称：小米",
        f"工作地点：{location}" if location else "",
        f"职位类别：{category}" if category else "",
        f"招聘渠道：{channel}" if channel else "",
        f"发布日期：{publish_date}" if publish_date else "",
        "岗位职责：",
        duties,
        "任职要求：",
        requirements,
    ]
    return "\n".join(part for part in parts if part)


def build_page_path(list_path: str, page: int) -> str:
    if page <= 1:
        return list_path
    return f"{list_path}-0-{page}"


def convert_xiaomi_job(
    row: dict[str, str],
    detail_fields: dict[str, str],
    *,
    crawl_time: str,
    list_path: str,
) -> dict[str, Any]:
    title = detail_fields.get("职位名称") or row["title"]
    location = normalize_location(detail_fields.get("工作地点") or row["location"])
    category = detail_fields.get("职位类别") or row["category"]
    channel = detail_fields.get("招聘渠道") or ""
    duties = detail_fields.get("工作职责") or detail_fields.get("岗位职责") or ""
    requirements = (
        detail_fields.get("工作要求")
        or detail_fields.get("任职要求")
        or detail_fields.get("任职资格")
        or ""
    )
    raw_text = build_raw_text(
        title=title,
        location=location,
        category=category,
        publish_date=row["publish_date"],
        channel=channel,
        duties=duties,
        requirements=requirements,
    )
    return {
        "id": f"xiaomi_{row['job_id']}",
        "source": "hr.xiaomi.com",
        "url": row["href"],
        "crawl_time": crawl_time,
        "job_title": title,
        "company": "小米",
        "location": location,
        "salary": "",
        "raw_text": raw_text,
        "html": None,
        "meta": {
            "language": "zh",
            "sft_ready": is_probably_tech_job(title, raw_text),
            "job_id": row["job_id"],
            "category": category,
            "publish_date": row["publish_date"],
            "channel": channel,
            "duties": duties,
            "requirements": requirements,
            "list_path": list_path,
            "detail_fields": detail_fields,
        },
    }


def crawl_xiaomi_jobs(
    *,
    list_paths: list[str],
    max_pages: int = 20,
    interval_seconds: float = 0.2,
    timeout: float = 20.0,
    retries: int = 3,
) -> list[dict[str, Any]]:
    session = build_session(timeout)
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for list_path in list_paths:
        for page in range(1, max_pages + 1):
            list_url = XIAOMI_LIST_URL_TEMPLATE.format(path=build_page_path(list_path, page))
            html_text = fetch_html(session, list_url, retries=retries)
            list_rows = parse_list_rows(html_text)
            if not list_rows:
                break
            new_count = 0
            for item in list_rows:
                if item["job_id"] in seen_ids:
                    continue
                detail_html = fetch_html(session, item["href"], retries=retries)
                detail_fields = parse_detail_fields(detail_html)
                row = convert_xiaomi_job(item, detail_fields, crawl_time=crawl_time, list_path=list_path)
                if not row["raw_text"]:
                    continue
                seen_ids.add(item["job_id"])
                rows.append(row)
                new_count += 1
                time.sleep(interval_seconds)
            if new_count == 0:
                break
            time.sleep(interval_seconds)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-path", action="append", default=[])
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--out", default="data/raw/xiaomi_jd_raw.jsonl")
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    parser.add_argument("--no-merge-existing", action="store_true")
    args = parser.parse_args()

    list_paths = args.list_path or ["8-0-2"]
    rows = crawl_xiaomi_jobs(
        list_paths=list_paths,
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
    print(f"crawled {len(rows)} Xiaomi posts for list_paths={list_paths}")
    print(f"wrote raw JSONL: {args.out} ({len(merged_rows)} rows)")
    print(f"upserted SQLite: {args.db}")


if __name__ == "__main__":
    main()
