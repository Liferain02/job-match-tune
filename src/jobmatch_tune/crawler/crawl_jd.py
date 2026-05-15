from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.preprocess.clean_text import clean_text
from jobmatch_tune.utils.io import write_jsonl


def stable_id(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"jd_{digest}"


def fetch_url(url: str, timeout: int, user_agent: str) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": user_agent, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7"},
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def extract_title(html: str) -> str:
    if BeautifulSoup is None:
        match = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
        return clean_text(match.group(1)) if match else ""
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" "))
    if soup.title:
        return clean_text(soup.title.get_text(" "))
    return ""


def extract_job_posting(html: str) -> dict[str, Any] | None:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "lxml")
        scripts = [script.string or script.get_text() for script in soup.find_all("script", attrs={"type": "application/ld+json"})]
    else:
        scripts = re.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html,
            flags=re.I | re.S,
        )
    for raw in scripts:
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                return item
    return None


def html_fragment_to_text(fragment: str) -> str:
    return clean_text(fragment, is_html=True)


def parse_job_posting(html: str) -> dict[str, str] | None:
    posting = extract_job_posting(html)
    if not posting:
        return None
    org = posting.get("hiringOrganization") or {}
    location = posting.get("jobLocation") or {}
    address = location.get("address") if isinstance(location, dict) else {}
    return {
        "job_title": clean_text(str(posting.get("title") or "")),
        "company": clean_text(str(org.get("name") or "")) if isinstance(org, dict) else "",
        "location": clean_text(str(address.get("addressCountry") or "")) if isinstance(address, dict) else "",
        "salary": "",
        "raw_text": html_fragment_to_text(str(posting.get("description") or "")),
    }


def extract_main_text(html: str) -> str:
    if trafilatura is not None:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted and len(extracted) >= 80:
            return clean_text(extracted)
    return clean_text(html, is_html=True)


def is_valid_job_page(html: str, title: str, text: str) -> bool:
    lowered_title = title.lower()
    if "not found" in lowered_title or "no encontrada" in lowered_title or "404" in lowered_title:
        return False
    if 'name="robots"' in html.lower() and "noindex" in html.lower():
        return False
    return len(text) >= 120 and any(keyword in text for keyword in ["岗位", "职位", "任职", "职责"])


def infer_source(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc or "public_web"


def crawl_urls(urls: list[str], request_config: dict[str, Any]) -> list[dict[str, Any]]:
    timeout = int(request_config.get("timeout", 20))
    interval = float(request_config.get("interval_seconds", 2))
    user_agent = request_config.get("user_agent", "JobMatchTuneBot/0.1")
    rows = []
    for idx, url in enumerate(urls):
        if idx:
            time.sleep(interval)
        html = fetch_url(url, timeout=timeout, user_agent=user_agent)
        structured = parse_job_posting(html) or {}
        raw_text = structured.get("raw_text") or extract_main_text(html)
        title = structured.get("job_title") or extract_title(html)
        if not is_valid_job_page(html, title, raw_text):
            print(f"skip non-job page: {url}")
            continue
        rows.append(
            {
                "id": stable_id(url),
                "source": infer_source(url),
                "url": url,
                "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "job_title": title,
                "company": structured.get("company", ""),
                "location": structured.get("location", ""),
                "salary": structured.get("salary", ""),
                "raw_text": raw_text,
                "html": html,
                "meta": {"title": title},
            }
        )
    return rows


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crawl.yaml")
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--out", default=None)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    urls = [*config.get("seed_urls", []), *args.url]
    urls = [url for url in dict.fromkeys(urls) if url]
    if not urls:
        raise SystemExit("No URLs provided. Add seed_urls in configs/crawl.yaml or pass --url.")

    rows = crawl_urls(urls, config.get("request", {}))
    out = args.out or config.get("output_raw_jsonl", "data/raw/jd_raw.jsonl")
    jsonl_rows = [{key: value for key, value in row.items() if key != "html"} for row in rows]
    write_jsonl(out, jsonl_rows)

    db_path = args.db or config.get("db_path")
    if db_path:
        init_db(db_path)
        upsert_jd_raw(db_path, rows)

    print(f"crawled {len(rows)} URLs")
    print(f"wrote raw JSONL: {out}")
    if db_path:
        print(f"upserted SQLite: {db_path}")


if __name__ == "__main__":
    main()
