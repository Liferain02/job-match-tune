#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHONPATH=src python -m jobmatch_tune.crawler.moka_careers \
  --sources configs/moka_sources.yaml \
  --page-limit 30 \
  --interval-seconds 0.2 \
  --out data/raw/moka_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
