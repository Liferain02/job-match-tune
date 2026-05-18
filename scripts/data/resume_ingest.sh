#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/data/resume_ingest.sh <resume-file-or-dir> [out-jsonl] [ocr-dir]" >&2
  exit 1
fi

cd "$(dirname "$0")/../.."
OUT_PATH="${2:-data/resume_raw/resume_ingest.jsonl}"
OCR_DIR="${3:-}"
if [[ -n "$OCR_DIR" ]]; then
  PYTHONPATH=src python -m jobmatch_tune.resume.ingest --input "$1" --out "$OUT_PATH" --ocr-dir "$OCR_DIR"
else
  PYTHONPATH=src python -m jobmatch_tune.resume.ingest --input "$1" --out "$OUT_PATH"
fi
