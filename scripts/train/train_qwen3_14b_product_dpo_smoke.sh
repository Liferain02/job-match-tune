#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHONPATH=src python -m jobmatch_tune.train.train_dpo \
  --config configs/train_qwen3_14b_product_dpo.yaml \
  --output_dir outputs/checkpoints/qwen3-14b-jobmatch-product-dpo-smoke \
  --max_train_samples "${MAX_TRAIN_SAMPLES:-8}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-4}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  "$@"
