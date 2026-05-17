#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.crawler.baidu_talent \
  --keywords-file configs/baidu_keywords.txt \
  --out data/raw/baidu_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3 \
  "$@"
