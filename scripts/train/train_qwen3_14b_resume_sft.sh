#!/usr/bin/env bash
set -euo pipefail

source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo

cd /share/home/lifr/workspace/code/job-match-tune
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHONPATH=src python -m jobmatch_tune.train.train_lora \
  --config configs/train_qwen3_14b_qlora.yaml \
  --adapter_path outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --train_file data/sft_resume/train.jsonl \
  --valid_file data/sft_resume/valid.jsonl \
  --output_dir outputs/checkpoints/qwen3-14b-resume-qlora \
  --learning_rate 5e-5 \
  --num_train_epochs 2
