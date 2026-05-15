#!/usr/bin/env bash
set -euo pipefail
python -m jobmatch_tune.train.train_lora \
  --config configs/train_qlora.yaml \
  --model_name_or_path models/Qwen3-1.7B \
  --output_dir outputs/checkpoints/qwen3-1.7b-jobmatch-smoke \
  --max_train_samples 4 \
  --max_eval_samples 2
