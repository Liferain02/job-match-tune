#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export JOBMATCH_MODEL_PATH="${JOBMATCH_MODEL_PATH:-models/Qwen3-14B}"
export JOBMATCH_ADAPTER_PATH="${JOBMATCH_ADAPTER_PATH:-outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601}"
export JOBMATCH_LOAD_4BIT="${JOBMATCH_LOAD_4BIT:-1}"
export JOBMATCH_INFERENCE_BACKEND="${JOBMATCH_INFERENCE_BACKEND:-transformers}"

uvicorn jobmatch_tune.api.server:app \
  --host "${JOBMATCH_API_HOST:-127.0.0.1}" \
  --port "${JOBMATCH_API_PORT:-8000}"
