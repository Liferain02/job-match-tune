#!/usr/bin/env bash
set -euo pipefail

source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo

cd /share/home/lifr/workspace/code/job-match-tune

PYTHONPATH=src python -m jobmatch_tune.crawler.moka_careers \
  --sources configs/moka_sources.yaml \
  --page-limit 30 \
  --interval-seconds 0.2 \
  --out data/raw/moka_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
