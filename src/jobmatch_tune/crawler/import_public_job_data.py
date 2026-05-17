from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def load_sources(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    sources = payload.get("sources") or []
    if not isinstance(sources, list):
        raise ValueError(f"Invalid source manifest: {path}")
    return sources


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() != "none" else ""


def build_job_area(location: str) -> tuple[str, str]:
    if "·" not in location:
        return location, ""
    parts = [part.strip() for part in location.split("·") if part.strip()]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " / ".join(parts[1:])


def build_raw_text_from_bosszp(row: dict[str, Any]) -> str:
    job_name = normalize_text(row.get("job_name"))
    job_area = normalize_text(row.get("job_area"))
    job_salary = normalize_text(row.get("job_salary"))
    com_name = normalize_text(row.get("com_name"))
    com_type = normalize_text(row.get("com_type"))
    com_size = normalize_text(row.get("com_size"))
    finance_stage = normalize_text(row.get("finance_stage"))
    work_year = normalize_text(row.get("work_year"))
    education = normalize_text(row.get("education"))
    job_benefits = normalize_text(row.get("job_benefits"))

    sections = [
        f"岗位名称：{job_name}" if job_name else "",
        f"公司名称：{com_name}" if com_name else "",
        f"工作地点：{job_area}" if job_area else "",
        f"薪资范围：{job_salary}" if job_salary else "",
        f"公司行业：{com_type}" if com_type else "",
        f"公司规模：{com_size}" if com_size else "",
        f"融资阶段：{finance_stage}" if finance_stage else "",
        f"经验要求：{work_year}" if work_year else "",
        f"学历要求：{education}" if education else "",
        f"福利标签：{job_benefits}" if job_benefits else "",
    ]
    return "\n".join(part for part in sections if part)


def strip_html_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|ul|ol|h\d)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_tagged_text(value: Any, tag: str) -> str:
    text = str(value or "")
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not match:
        return ""
    return normalize_text(match.group(1))


def build_salary_string(
    salary_min: Any,
    salary_max: Any,
    salary_currency: Any,
    salary_period: Any,
) -> str:
    minimum = salary_min if salary_min not in (None, "") else None
    maximum = salary_max if salary_max not in (None, "") else None
    if minimum is None and maximum is None:
        return ""
    currency = normalize_text(salary_currency)
    period = normalize_text(salary_period)
    if minimum is not None and maximum is not None:
        body = f"{minimum}-{maximum}"
    else:
        body = str(minimum if minimum is not None else maximum)
    suffix = f" {currency}".strip()
    if period:
        suffix = f"{suffix}/{period}".strip("/")
    return f"{body} {suffix}".strip()


def convert_bosszp_csv_row(
    row: dict[str, Any],
    source_name: str,
    source_url: str,
    crawl_time: str,
) -> dict[str, Any]:
    job_title = normalize_text(row.get("job_name"))
    company = normalize_text(row.get("com_name"))
    location = normalize_text(row.get("job_area"))
    salary = normalize_text(row.get("job_salary"))
    raw_text = build_raw_text_from_bosszp(row)
    stable_payload = {
        "job_title": job_title,
        "company": company,
        "location": location,
        "salary": salary,
        "work_year": normalize_text(row.get("work_year")),
        "education": normalize_text(row.get("education")),
    }
    digest = hashlib.sha1(
        json.dumps(stable_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "id": f"{source_name}_{digest}",
        "source": source_name,
        "url": source_url,
        "crawl_time": crawl_time,
        "job_title": job_title,
        "company": company,
        "location": location,
        "salary": salary,
        "raw_text": raw_text,
        "html": None,
        "meta": {
            "schema": "bosszp_csv_v1",
            "language": "zh",
            "sft_ready": False,
            "job_area_city": build_job_area(location)[0],
            "job_area_detail": build_job_area(location)[1],
            "company_type": normalize_text(row.get("com_type")),
            "company_size": normalize_text(row.get("com_size")),
            "finance_stage": normalize_text(row.get("finance_stage")),
            "work_year": normalize_text(row.get("work_year")),
            "education": normalize_text(row.get("education")),
            "job_benefits": normalize_text(row.get("job_benefits")),
            "source_file": source_url,
            "raw_columns": row,
        },
    }


def import_bosszp_csv(
    csv_path: str | Path,
    source_name: str,
    source_url: str,
    crawl_time: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = convert_bosszp_csv_row(
                row=row,
                source_name=source_name,
                source_url=source_url,
                crawl_time=crawl_time,
            )
            if converted["job_title"] and converted["raw_text"]:
                rows.append(converted)
    return rows


def build_raw_text_from_workaggregation(row: dict[str, Any]) -> str:
    title = normalize_text(row.get("title"))
    place = normalize_text(row.get("place"))
    salary = normalize_text(row.get("salary"))
    experience = normalize_text(row.get("experience"))
    education = normalize_text(row.get("education"))
    company_type = normalize_text(row.get("companytype"))
    industry = normalize_text(row.get("industry"))
    provider = normalize_text(row.get("provider"))
    keyword = normalize_text(row.get("keyword"))
    description = normalize_text(row.get("description"))
    sections = [
        f"岗位名称：{title}" if title else "",
        f"来源站点：{provider}" if provider else "",
        f"关键词：{keyword}" if keyword else "",
        f"工作地点：{place}" if place else "",
        f"薪资范围：{salary}" if salary else "",
        f"经验要求：{experience}" if experience else "",
        f"学历要求：{education}" if education else "",
        f"公司性质：{company_type}" if company_type else "",
        f"行业标签：{industry}" if industry else "",
        f"职位描述：{description}" if description else "",
    ]
    return "\n".join(part for part in sections if part)


def convert_workaggregation_csv_row(
    row: dict[str, Any],
    source_name: str,
    source_url: str,
    crawl_time: str,
) -> dict[str, Any]:
    job_title = normalize_text(row.get("title"))
    location = normalize_text(row.get("place"))
    salary = normalize_text(row.get("salary"))
    provider = normalize_text(row.get("provider"))
    raw_text = build_raw_text_from_workaggregation(row)
    stable_payload = {
        "job_title": job_title,
        "provider": provider,
        "location": location,
        "salary": salary,
        "experience": normalize_text(row.get("experience")),
        "education": normalize_text(row.get("education")),
    }
    digest = hashlib.sha1(
        json.dumps(stable_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "id": f"{source_name}_{digest}",
        "source": source_name,
        "url": source_url,
        "crawl_time": crawl_time,
        "job_title": job_title,
        "company": provider,
        "location": location,
        "salary": salary,
        "raw_text": raw_text,
        "html": None,
        "meta": {
            "schema": "workaggregation_csv_v1",
            "language": "zh",
            "sft_ready": False,
            "keyword": normalize_text(row.get("keyword")),
            "experience": normalize_text(row.get("experience")),
            "education": normalize_text(row.get("education")),
            "company_type": normalize_text(row.get("companytype")),
            "industry": normalize_text(row.get("industry")),
            "description": normalize_text(row.get("description")),
            "source_file": source_url,
            "raw_columns": row,
        },
    }


def import_workaggregation_csv(
    csv_path: str | Path,
    source_name: str,
    source_url: str,
    crawl_time: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = convert_workaggregation_csv_row(
                row=row,
                source_name=source_name,
                source_url=source_url,
                crawl_time=crawl_time,
            )
            if converted["job_title"] and converted["raw_text"]:
                rows.append(converted)
    return rows


def build_raw_text_from_job_educational_row(row: dict[str, Any]) -> str:
    title = extract_tagged_text(row.get("user"), "岗位名称")
    description = extract_tagged_text(row.get("user"), "岗位描述")
    education_desc = extract_tagged_text(row.get("user"), "学历描述")
    sections = [
        f"岗位名称：{title}" if title else "",
        "任务类型：从岗位中提取学历",
        "岗位描述：",
        description,
        f"学历提示：{education_desc}" if education_desc else "",
    ]
    return "\n".join(part for part in sections if part)


def convert_job_educational_row(
    row: dict[str, Any],
    source_name: str,
    source_url: str,
    crawl_time: str,
) -> dict[str, Any]:
    job_id = normalize_text(row.get("job_id"))
    title = extract_tagged_text(row.get("user"), "岗位名称")
    description = extract_tagged_text(row.get("user"), "岗位描述")
    education_desc = extract_tagged_text(row.get("user"), "学历描述")
    label = normalize_text(row.get("assistant"))
    digest = hashlib.sha1(
        json.dumps(
            {
                "job_id": job_id,
                "title": title,
                "description": description[:500],
                "assistant": label,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    row_id = f"{source_name}_{job_id or digest}"
    return {
        "id": row_id,
        "source": source_name,
        "url": source_url,
        "crawl_time": crawl_time,
        "job_title": title,
        "company": "",
        "location": "",
        "salary": "",
        "raw_text": build_raw_text_from_job_educational_row(row),
        "html": None,
        "meta": {
            "schema": "job_educational_parquet_v1",
            "language": "zh",
            "sft_ready": False,
            "task_type": "education_extraction",
            "system_prompt": normalize_text(row.get("system")),
            "education_label": label,
            "education_desc": education_desc,
            "user_short": normalize_text(row.get("user_short")),
            "user_short2": normalize_text(row.get("user_short2")),
            "ai_prediction": normalize_text(row.get("ai")),
            "diff": bool(row.get("diff")),
            "source_file": source_url,
        },
    }


def import_job_educational_parquet(
    parquet_path: str | Path,
    source_name: str,
    source_url: str,
    crawl_time: str,
    *,
    min_description_chars: int = 40,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    rows: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            description = extract_tagged_text(row.get("user"), "岗位描述")
            label = normalize_text(row.get("assistant"))
            if len(description) < min_description_chars or not label:
                continue
            converted = convert_job_educational_row(
                row=row,
                source_name=source_name,
                source_url=source_url,
                crawl_time=crawl_time,
            )
            if converted["job_title"] and converted["raw_text"]:
                rows.append(converted)
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


DEFAULT_OPEN_APPLY_KEYWORDS = [
    "software",
    "engineer",
    "engineering",
    "developer",
    "backend",
    "back-end",
    "frontend",
    "front-end",
    "full stack",
    "fullstack",
    "android",
    "ios",
    "mobile",
    "platform",
    "devops",
    "sre",
    "site reliability",
    "cloud",
    "infrastructure",
    "data",
    "machine learning",
    "ml",
    "ai ",
    " ai",
    "llm",
    "genai",
    "security",
    "qa",
    "test",
    "automation",
]

DEFAULT_OPEN_APPLY_EXCLUDE_KEYWORDS = [
    "sales",
    "account executive",
    "recruiter",
    "marketing",
    "designer",
    "finance",
    "legal",
    "operations",
    "customer support",
    "talent",
    "hr ",
    "human resources",
]


def matches_keyword_filters(
    text: str,
    include_keywords: list[str],
    exclude_keywords: list[str],
) -> bool:
    lowered = f" {text.lower()} "
    if exclude_keywords and any(keyword.lower() in lowered for keyword in exclude_keywords):
        return False
    if not include_keywords:
        return True
    return any(keyword.lower() in lowered for keyword in include_keywords)


def convert_open_apply_jobs_row(
    row: dict[str, Any],
    source_name: str,
    source_url: str,
    crawl_time: str,
) -> dict[str, Any]:
    locations = row.get("locations") or []
    if not isinstance(locations, list):
        locations = [str(locations)]
    location = " | ".join(str(item).strip() for item in locations if str(item).strip())
    title = normalize_text(row.get("title"))
    description_text = strip_html_text(row.get("description_html"))
    salary = build_salary_string(
        row.get("salary_min"),
        row.get("salary_max"),
        row.get("salary_currency"),
        row.get("salary_period"),
    )
    department = normalize_text(row.get("department"))
    employment_type = normalize_text(row.get("employment_type"))
    source_slug = normalize_text(row.get("source_slug"))
    remote_text = "是" if bool(row.get("remote")) else "否"
    sections = [
        f"岗位名称：{title}" if title else "",
        f"公司标识：{source_slug}" if source_slug else "",
        f"部门：{department}" if department else "",
        f"工作地点：{location}" if location else "",
        f"雇佣类型：{employment_type}" if employment_type else "",
        f"远程：{remote_text}",
        f"薪资范围：{salary}" if salary else "",
        "职位描述：",
        description_text,
    ]
    return {
        "id": normalize_text(row.get("id")) or hashlib.sha1(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "source": source_name,
        "url": normalize_text(row.get("apply_url")) or source_url,
        "crawl_time": crawl_time,
        "job_title": title,
        "company": source_slug,
        "location": location,
        "salary": salary,
        "raw_text": "\n".join(part for part in sections if part),
        "html": None,
        "meta": {
            "schema": "open_apply_jobs_parquet_v1",
            "language": "en",
            "sft_ready": False,
            "platform": normalize_text(row.get("platform")),
            "source_slug": source_slug,
            "department": department,
            "employment_type": employment_type,
            "remote": bool(row.get("remote")),
            "posted_at": normalize_text(row.get("posted_at")),
            "updated_at": normalize_text(row.get("updated_at")),
            "locations": locations,
            "source_file": source_url,
        },
    }


def import_open_apply_jobs_parquet(
    parquet_path: str | Path,
    source_name: str,
    source_url: str,
    crawl_time: str,
    *,
    include_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
    min_description_chars: int = 600,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    include_keywords = include_keywords or DEFAULT_OPEN_APPLY_KEYWORDS
    exclude_keywords = exclude_keywords or DEFAULT_OPEN_APPLY_EXCLUDE_KEYWORDS
    parquet = pq.ParquetFile(parquet_path)
    rows: list[dict[str, Any]] = []
    platform = Path(parquet_path).stem.lower()
    for batch in parquet.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            title = normalize_text(row.get("title"))
            department = normalize_text(row.get("department"))
            description_text = strip_html_text(row.get("description_html"))
            combined = "\n".join([title, department, description_text[:400]])
            if len(description_text) < min_description_chars:
                continue
            if not matches_keyword_filters(combined, include_keywords, exclude_keywords):
                continue
            row["platform"] = platform
            converted = convert_open_apply_jobs_row(
                row=row,
                source_name=source_name,
                source_url=source_url,
                crawl_time=crawl_time,
            )
            if converted["job_title"] and converted["raw_text"]:
                rows.append(converted)
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def import_sources(source_specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    imported: list[dict[str, Any]] = []
    for spec in source_specs:
        enabled = bool(spec.get("enabled", True))
        if not enabled:
            continue
        source_type = spec.get("type")
        local_path = spec.get("local_path")
        if not local_path:
            raise ValueError(f"Missing local_path in source spec: {spec}")
        source_name = str(spec.get("name") or "public_dataset")
        source_url = str(spec.get("source_url") or local_path)
        if source_type == "bosszp_csv":
            imported.extend(
                import_bosszp_csv(
                    csv_path=local_path,
                    source_name=source_name,
                    source_url=source_url,
                    crawl_time=crawl_time,
                )
            )
            continue
        if source_type == "workaggregation_csv":
            imported.extend(
                import_workaggregation_csv(
                    csv_path=local_path,
                    source_name=source_name,
                    source_url=source_url,
                    crawl_time=crawl_time,
                )
            )
            continue
        if source_type == "open_apply_jobs_parquet":
            imported.extend(
                import_open_apply_jobs_parquet(
                    parquet_path=local_path,
                    source_name=source_name,
                    source_url=source_url,
                    crawl_time=crawl_time,
                    include_keywords=list(spec.get("include_keywords") or DEFAULT_OPEN_APPLY_KEYWORDS),
                    exclude_keywords=list(spec.get("exclude_keywords") or DEFAULT_OPEN_APPLY_EXCLUDE_KEYWORDS),
                    min_description_chars=int(spec.get("min_description_chars") or 600),
                    max_rows=int(spec["max_rows"]) if spec.get("max_rows") else None,
                )
            )
            continue
        if source_type == "job_educational_parquet":
            imported.extend(
                import_job_educational_parquet(
                    parquet_path=local_path,
                    source_name=source_name,
                    source_url=source_url,
                    crawl_time=crawl_time,
                    min_description_chars=int(spec.get("min_description_chars") or 40),
                    max_rows=int(spec["max_rows"]) if spec.get("max_rows") else None,
                )
            )
            continue
        raise ValueError(f"Unsupported public source type: {source_type}")
    return imported


def merge_rows(
    existing_rows: Iterable[dict[str, Any]],
    new_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        row_id = str(row.get("id") or "")
        if row_id:
            by_id[row_id] = dict(row)
    for row in new_rows:
        by_id[str(row["id"])] = {key: value for key, value in row.items() if key != "html"}
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="configs/public_job_sources.yaml")
    parser.add_argument("--out", default="data/raw/public_job_datasets_raw.jsonl")
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    parser.add_argument("--no-merge-existing", action="store_true")
    args = parser.parse_args()

    source_specs = load_sources(args.sources)
    rows = import_sources(source_specs)
    jsonl_rows = [{key: value for key, value in row.items() if key != "html"} for row in rows]

    merged_rows = jsonl_rows
    if not args.no_merge_existing:
        try:
            merged_rows = merge_rows(read_jsonl(args.out), rows)
        except FileNotFoundError:
            pass

    write_jsonl(args.out, merged_rows)
    init_db(args.db)
    upsert_jd_raw(args.db, rows)
    print(f"imported {len(rows)} public job rows from {args.sources}")
    print(f"wrote raw JSONL: {args.out} ({len(merged_rows)} rows)")
    print(f"upserted SQLite: {args.db}")


if __name__ == "__main__":
    main()
