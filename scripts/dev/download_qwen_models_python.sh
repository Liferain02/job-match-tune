#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:-14B}"

case "$MODEL_SIZE" in
  14B)
    MODEL_ID="Qwen/Qwen3-14B"
    LOCAL_DIR="models/Qwen3-14B"
    ;;
  1.7B)
    MODEL_ID="Qwen/Qwen3-1.7B"
    LOCAL_DIR="models/Qwen3-1.7B"
    ;;
  *)
    echo "Unsupported size: $MODEL_SIZE. Use 14B or 1.7B." >&2
    exit 1
    ;;
esac

cd "$(dirname "$0")/../.."
PYTHONPATH=src python -m jobmatch_tune.utils.download_hf_snapshot --repo-id "$MODEL_ID" --local-dir "$LOCAL_DIR"
