#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

bash scripts/data/audit_public_jd_data.sh \
  --input data/raw/public_job_datasets_raw.jsonl \
  --out outputs/eval_reports/public_jd_import_audit.json
bash scripts/data/build_public_jd_candidate_pool.sh
bash scripts/data/build_jd_train_pool_combined.sh
