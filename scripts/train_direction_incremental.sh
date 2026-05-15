#!/usr/bin/env bash
set -euo pipefail

source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo

cd /share/home/lifr/workspace/code/job-match-tune

PYTHONPATH=src python -m jobmatch_tune.dataset.build_direction_hardcase_sft
PYTHONPATH=src python -m jobmatch_tune.dataset.build_incremental_sft_dataset

PYTHONPATH=src python -m jobmatch_tune.train.train_lora \
  --config configs/train_qlora.yaml \
  --model_name_or_path models/Qwen3-1.7B \
  --adapter_path outputs/checkpoints/qwen3-1.7b-dft-lr1e-4 \
  --train_file data/sft/train_incremental.jsonl \
  --valid_file data/sft/valid_incremental.jsonl \
  --output_dir outputs/checkpoints/qwen3-1.7b-direction-incremental \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 16
