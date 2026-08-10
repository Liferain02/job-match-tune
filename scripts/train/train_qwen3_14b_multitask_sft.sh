#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash scripts/data/build_multitask_sft_dataset.sh
source scripts/train/_training_readiness_gate.sh
run_training_readiness_gate sft

PYTHONPATH=src python -m jobmatch_tune.train.train_lora \
  --config configs/train_qwen3_14b_qlora.yaml \
  --train_file data/sft_multitask/train.jsonl \
  --valid_file data/sft_multitask/valid.jsonl
