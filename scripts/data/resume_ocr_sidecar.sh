#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/data/resume_ocr_sidecar.sh <image-or-pdf-file-or-dir> [out-dir]" >&2
  exit 1
fi

cd "$(dirname "$0")/../.."
OUT_DIR="${2:-data/resume_ocr_text}"
PYTHONPATH=src python -m jobmatch_tune.resume.ocr --input "$1" --out-dir "$OUT_DIR"
