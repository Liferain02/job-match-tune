#!/usr/bin/env bash
set -euo pipefail

source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo

cd /share/home/lifr/workspace/code/job-match-tune
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHONPATH=src python -m jobmatch_tune.train.train_lora \
  --config configs/train_qlora.yaml \
  --model_name_or_path models/Qwen3-14B \
  --output_dir outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 768 \
  --lora_r 8 \
  --lora_alpha 16

