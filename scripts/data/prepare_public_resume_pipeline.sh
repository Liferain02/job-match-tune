#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

bash scripts/data/import_public_resume_exports.sh "$@"
bash scripts/data/audit_public_resume_data.sh \
  --input data/external/public_resume_imports.jsonl \
  --out outputs/eval_reports/public_resume_import_audit.json
bash scripts/data/build_resume_train_pool_combined.sh
