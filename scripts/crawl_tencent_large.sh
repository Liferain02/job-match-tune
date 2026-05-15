#!/usr/bin/env bash
set -euo pipefail
python -m jobmatch_tune.crawler.tencent_careers \
  --keywords-file configs/tencent_keywords.txt \
  --limit 3000 \
  --page-size 50 \
  --max-pages 30 \
  --interval-seconds 0.5 \
  --category 技术 \
  --out data/raw/tencent_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
