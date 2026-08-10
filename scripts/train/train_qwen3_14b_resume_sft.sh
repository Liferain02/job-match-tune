#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source scripts/train/_training_readiness_gate.sh
run_training_readiness_gate sft

PYTHONPATH=src python -m jobmatch_tune.train.train_lora \
  --config configs/train_qwen3_14b_qlora.yaml \
  --adapter_path outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601 \
  --train_file data/sft_resume/train.jsonl \
  --valid_file data/sft_resume/valid.jsonl \
  --output_dir outputs/checkpoints/qwen3-14b-resume-qlora \
  --learning_rate 5e-5 \
  --num_train_epochs 2
