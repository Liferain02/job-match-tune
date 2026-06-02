#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.dataset.build_preference_bootstrap_dataset \
  --train-input data/sft_multitask/train.jsonl \
  --valid-input data/sft_multitask/valid.jsonl \
  --train-out data/preference_product_bootstrap/train.jsonl \
  --valid-out data/preference_product_bootstrap/valid.jsonl \
  "$@"
