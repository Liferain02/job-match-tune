#!/usr/bin/env bash
set -euo pipefail

export JOBMATCH_MODEL_PATH="${JOBMATCH_MODEL_PATH:-models/Qwen3-14B}"
export JOBMATCH_ADAPTER_PATH="${JOBMATCH_ADAPTER_PATH:-outputs/checkpoints/qwen3-14b-jobmatch-qlora}"
export JOBMATCH_VLLM_HOST="${JOBMATCH_VLLM_HOST:-0.0.0.0}"
export JOBMATCH_VLLM_PORT="${JOBMATCH_VLLM_PORT:-8010}"
export JOBMATCH_VLLM_SERVED_MODEL_NAME="${JOBMATCH_VLLM_SERVED_MODEL_NAME:-jobmatch-qwen3-14b}"
export JOBMATCH_VLLM_LORA_NAME="${JOBMATCH_VLLM_LORA_NAME:-jobmatch-lora}"
export JOBMATCH_VLLM_MAX_MODEL_LEN="${JOBMATCH_VLLM_MAX_MODEL_LEN:-8192}"

vllm serve "${JOBMATCH_MODEL_PATH}" \
  --host "${JOBMATCH_VLLM_HOST}" \
  --port "${JOBMATCH_VLLM_PORT}" \
  --served-model-name "${JOBMATCH_VLLM_SERVED_MODEL_NAME}" \
  --enable-lora \
  --max-model-len "${JOBMATCH_VLLM_MAX_MODEL_LEN}" \
  --lora-modules "${JOBMATCH_VLLM_LORA_NAME}=${JOBMATCH_ADAPTER_PATH}"
