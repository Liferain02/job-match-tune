#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.preprocess.normalize_jd \
  --db data/jobmatch_tune.sqlite3 \
  --out data/interim/jd_clean.jsonl \
  --schema configs/label_schema.yaml

PYTHONPATH=src python -m jobmatch_tune.preprocess.deduplicate \
  --input data/interim/jd_clean.jsonl \
  --out data/interim/jd_clean_dedup.jsonl \
  --text-key clean_text

PYTHONPATH=src python -m jobmatch_tune.dataset.build_sft_dataset \
  --jd data/interim/jd_clean_dedup.jsonl \
  --out-dir data/sft \
  --quality-profile strict

PYTHONPATH=src python -m jobmatch_tune.dataset.build_sft_dataset \
  --jd data/interim/jd_clean_dedup.jsonl \
  --out-dir data/sft_expanded \
  --include-weak-tech \
  --quality-profile expanded \
  --target-total 20000
