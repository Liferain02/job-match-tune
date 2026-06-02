#!/usr/bin/env bash
set -euo pipefail

source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo

cd /share/home/lifr/workspace/code/job-match-tune
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source scripts/train/_training_readiness_gate.sh
run_training_readiness_gate

PYTHONPATH=src python -m jobmatch_tune.train.train_lora \
  --config configs/train_qwen3_14b_qlora.yaml
