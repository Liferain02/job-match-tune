#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/data/build_match_eval_dataset.sh
PYTHONPATH=src python -m jobmatch_tune.dataset.build_match_sft_dataset
