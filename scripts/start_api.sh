#!/usr/bin/env bash
set -euo pipefail

export JOBMATCH_MODEL_PATH="${JOBMATCH_MODEL_PATH:-models/Qwen3-14B}"
export JOBMATCH_ADAPTER_PATH="${JOBMATCH_ADAPTER_PATH:-outputs/checkpoints/qwen3-14b-jobmatch-qlora}"
export JOBMATCH_LOAD_4BIT="${JOBMATCH_LOAD_4BIT:-1}"

uvicorn jobmatch_tune.api.server:app --host 0.0.0.0 --port "${JOBMATCH_API_PORT:-8000}"
