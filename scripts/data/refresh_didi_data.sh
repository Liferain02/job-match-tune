#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.crawler.didi_careers \
  --out data/raw/didi_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
