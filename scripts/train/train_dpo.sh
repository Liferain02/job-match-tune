#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
if [[ "${JOBMATCH_ALLOW_DPO:-0}" != "1" ]]; then
  echo "DPO 默认暂停：当前 preference 未达到非合成样本数量与质量门槛。" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source scripts/train/_training_readiness_gate.sh
run_training_readiness_gate dpo

CONFIG="${JOBMATCH_DPO_CONFIG:-configs/train_qwen3_14b_dpo.yaml}"
PYTHONPATH=src python -m jobmatch_tune.train.train_dpo --config "$CONFIG" "$@"

