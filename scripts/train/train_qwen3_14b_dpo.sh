#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source scripts/train/_dpo_pause_gate.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source scripts/train/_training_readiness_gate.sh
run_training_readiness_gate jd_dpo

PYTHONPATH=src python -m jobmatch_tune.train.train_dpo \
  --config configs/train_qwen3_14b_dpo.yaml
