#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/data/probe_feishu_ats.sh <base-url> [out-json]" >&2
  exit 1
fi

BASE_URL="$1"
OUT_PATH="${2:-outputs/eval_reports/feishu_ats_probe.json}"

PYTHONPATH=src python -m jobmatch_tune.crawler.feishu_ats_probe \
  --base-url "$BASE_URL" \
  --out "$OUT_PATH"
