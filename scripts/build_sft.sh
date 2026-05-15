#!/usr/bin/env bash
set -euo pipefail
python -m jobmatch_tune.dataset.build_sft_dataset \
  --jd data/interim/jd_clean.jsonl \
  --out-dir data/sft
