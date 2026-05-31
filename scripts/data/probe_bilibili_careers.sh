#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_PATH="${1:-outputs/eval_reports/bilibili_probe.json}"

PYTHONPATH=src python -m jobmatch_tune.crawler.bilibili_careers_probe --out "$OUT_PATH"
