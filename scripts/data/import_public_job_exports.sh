#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHONPATH=src python -m jobmatch_tune.dataset.download_verified_sources \
  --manifest configs/public_job_sources.yaml \
  --report-out outputs/eval_reports/public_job_download_verification.json

PYTHONPATH=src python -m jobmatch_tune.crawler.import_public_job_data \
  --sources configs/public_job_sources.yaml \
  --out data/raw/public_job_datasets_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
