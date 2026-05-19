#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.crawler.xiaomi_careers \
  --list-path 8-0-2 \
  --out data/raw/xiaomi_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
