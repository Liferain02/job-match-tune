#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.dataset.build_multilingual_weak_sft_dataset \
  --jd data/interim/jd_clean_dedup.jsonl \
  --schema configs/label_schema.yaml \
  --out-dir data/sft_multilingual_weak \
  "$@"
