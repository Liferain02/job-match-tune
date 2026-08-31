#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MANIFEST="${1:-configs/public_job_sources.yaml}"

PYTHONPATH=src python -m jobmatch_tune.dataset.download_verified_sources \
  --manifest "$MANIFEST" \
  --report-out outputs/eval_reports/public_job_download_verification.json
PYTHONPATH=src python -m jobmatch_tune.crawler.import_public_job_data \
  --sources "$MANIFEST" \
  --out data/raw/public_job_datasets_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
PYTHONPATH=src python -m jobmatch_tune.eval.audit_public_jd_data \
  --input data/raw/public_job_datasets_raw.jsonl \
  --out outputs/eval_reports/public_jd_import_audit.json
PYTHONPATH=src python -m jobmatch_tune.dataset.build_public_jd_candidate_pool
PYTHONPATH=src python -m jobmatch_tune.dataset.build_jd_train_pool_combined
