from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS jd_raw (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  url TEXT NOT NULL,
  crawl_time TEXT NOT NULL,
  job_title TEXT,
  company TEXT,
  location TEXT,
  salary TEXT,
  raw_text TEXT NOT NULL,
  html TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS jd_clean (
  id TEXT PRIMARY KEY,
  raw_id TEXT NOT NULL,
  job_title TEXT,
  company TEXT,
  location TEXT,
  clean_text TEXT NOT NULL,
  sections_json TEXT NOT NULL DEFAULT '{}',
  labels_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY(raw_id) REFERENCES jd_raw(id)
);

CREATE TABLE IF NOT EXISTS resume_clean (
  id TEXT PRIMARY KEY,
  target_role TEXT,
  clean_text TEXT NOT NULL,
  labels_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sft_samples (
  id TEXT PRIMARY KEY,
  task_type TEXT NOT NULL,
  split TEXT NOT NULL,
  messages_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jd_raw_url ON jd_raw(url);
CREATE INDEX IF NOT EXISTS idx_sft_split_task ON sft_samples(split, task_type);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_jd_raw(db_path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    sql = """
    INSERT INTO jd_raw (
      id, source, url, crawl_time, job_title, company, location, salary, raw_text, html, meta_json
    )
    VALUES (:id, :source, :url, :crawl_time, :job_title, :company, :location, :salary,
            :raw_text, :html, :meta_json)
    ON CONFLICT(id) DO UPDATE SET
      source=excluded.source,
      url=excluded.url,
      crawl_time=excluded.crawl_time,
      job_title=excluded.job_title,
      company=excluded.company,
      location=excluded.location,
      salary=excluded.salary,
      raw_text=excluded.raw_text,
      html=excluded.html,
      meta_json=excluded.meta_json
    """
    prepared = []
    for row in rows:
        item = dict(row)
        item["meta_json"] = json.dumps(item.get("meta", {}), ensure_ascii=False)
        prepared.append(item)
    with connect(db_path) as conn:
        conn.executemany(sql, prepared)


def upsert_jd_clean(db_path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    sql = """
    INSERT INTO jd_clean (
      id, raw_id, job_title, company, location, clean_text, sections_json, labels_json
    )
    VALUES (:id, :raw_id, :job_title, :company, :location, :clean_text, :sections_json,
            :labels_json)
    ON CONFLICT(id) DO UPDATE SET
      raw_id=excluded.raw_id,
      job_title=excluded.job_title,
      company=excluded.company,
      location=excluded.location,
      clean_text=excluded.clean_text,
      sections_json=excluded.sections_json,
      labels_json=excluded.labels_json
    """
    prepared = []
    for row in rows:
        item = dict(row)
        item["sections_json"] = json.dumps(item.get("sections", {}), ensure_ascii=False)
        item["labels_json"] = json.dumps(item.get("labels", {}), ensure_ascii=False)
        prepared.append(item)
    with connect(db_path) as conn:
        conn.executemany(sql, prepared)


def fetch_table(db_path: str | Path, table: str) -> list[dict[str, Any]]:
    allowed = {"jd_raw", "jd_clean", "resume_clean", "sft_samples"}
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
