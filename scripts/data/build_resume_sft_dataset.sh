#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHONPATH=src python -m jobmatch_tune.dataset.build_resume_sft_dataset \
  --input data/eval/resume_train_pool_combined.jsonl \
  --out-dir data/sft_resume \
  "$@"
