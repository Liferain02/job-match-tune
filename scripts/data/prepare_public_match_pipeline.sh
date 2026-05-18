#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

bash scripts/data/import_public_match_exports.sh "$@"
bash scripts/data/audit_public_match_data.sh \
  --input data/external/public_match_imports.jsonl \
  --out outputs/eval_reports/public_match_import_audit.json
bash scripts/data/build_match_train_pool_combined.sh
