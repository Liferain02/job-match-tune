#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source scripts/train/_training_readiness_gate.sh
run_training_readiness_gate sft

CONFIG="${JOBMATCH_SFT_CONFIG:-configs/train_qwen3_14b_qlora.yaml}"
PYTHONPATH=src python -m jobmatch_tune.train.train_lora --config "$CONFIG" "$@"

