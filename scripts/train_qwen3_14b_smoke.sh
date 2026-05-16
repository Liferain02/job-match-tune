#!/usr/bin/env bash
set -euo pipefail

source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo

cd /share/home/lifr/workspace/code/job-match-tune
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHONPATH=src python -m jobmatch_tune.train.train_lora \
  --config configs/train_qwen3_14b_qlora.yaml \
  --output_dir outputs/checkpoints/qwen3-14b-smoke \
  --max_train_samples 32 \
  --max_eval_samples 8 \
  --num_train_epochs 0.2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4
