#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.preprocess.normalize_jd \
  --db data/jobmatch_tune.sqlite3 \
  --out data/interim/jd_clean.jsonl \
  --schema configs/label_schema.yaml \
  --skip-clean-table-sync

PYTHONPATH=src python -m jobmatch_tune.preprocess.deduplicate \
  --input data/interim/jd_clean.jsonl \
  --out data/interim/jd_clean_dedup.jsonl \
  --text-key clean_text

bash scripts/data/build_current_data_pools.sh
