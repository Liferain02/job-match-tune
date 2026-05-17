#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.crawler.jd_careers \
  --out data/raw/jd_careers_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
