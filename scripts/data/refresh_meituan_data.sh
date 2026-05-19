#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.crawler.meituan_careers \
  --out data/raw/meituan_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
