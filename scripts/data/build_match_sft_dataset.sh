#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/data/build_match_eval_dataset.sh
bash scripts/data/build_match_train_pool_synthetic.sh
bash scripts/data/build_match_train_pool_combined.sh
PYTHONPATH=src python -m jobmatch_tune.dataset.build_match_sft_dataset
